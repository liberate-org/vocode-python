from copy import deepcopy
import re
import time
from typing import (
    Dict,
    Any,
    AsyncGenerator,
    AsyncIterable,
    Callable,
    List,
    Literal,
    Optional,
    TypeVar,
    Union,
)
import logging

from vocode.streaming.models.actions import FunctionCall, FunctionFragment
from vocode.streaming.models.events import Sender
from vocode.streaming.models.transcript import (
    ActionFinish,
    ActionStart,
    EventLog,
    Message,
    Transcript,
)

SENTENCE_ENDINGS = [".", "!", "?", "\n"]


async def collate_response_async(
    gen: AsyncIterable[Union[str, FunctionFragment]],
    sentence_endings: List[str] = SENTENCE_ENDINGS,
    get_functions: Literal[True, False] = False,
    logger: Optional[logging.Logger] = None,
    start_token_processing: Optional[float] = time.time()
) -> AsyncGenerator[Union[str, FunctionCall], None]:
    sentence_endings_pattern = "|".join(map(re.escape, sentence_endings))
    list_item_ending_pattern = r"\n"
    buffer = ""
    function_name_buffer = ""
    function_args_buffer = ""
    tool_calls = []
    has_seen_tool = False
    prev_ends_with_money = False
    async for token in gen:
        if not token:
            continue
        if isinstance(token, str):
            if prev_ends_with_money and token.startswith(" "):
                if logger:
                    logger.debug("Took %s to generate [%s]", 
                                 time.time() - start_token_processing, 
                                 buffer.strip())
                yield buffer.strip()
                buffer = ""

            buffer += token
            possible_list_item = bool(re.match(r"^\d+[ .]", buffer))
            ends_with_money = bool(re.findall(r"\$\d+.$", buffer))
            if re.findall(
                list_item_ending_pattern
                if possible_list_item
                else sentence_endings_pattern,
                token,
            ):
                if not ends_with_money:
                    to_return = buffer.strip()
                    if to_return:
                        if logger:
                            logger.debug("Took %s to generate [%s]",
                              time.time() - start_token_processing,
                              to_return)
                        yield to_return
                    buffer = ""
            prev_ends_with_money = ends_with_money
        elif isinstance(token, FunctionFragment):
            if len(token.name) > 0:
                has_seen_tool = True
            if has_seen_tool and len(token.name) > 0 and len(function_name_buffer) > 0 and len(function_args_buffer) > 0:
                if logger:
                    logger.info(f"Second tool seen {token.name}")
                tool_calls.append(FunctionCall(name=function_name_buffer, arguments=function_args_buffer))
                function_name_buffer = ""
                function_args_buffer = ""                
            function_name_buffer += token.name
            function_args_buffer += token.arguments
    to_return = buffer.strip()
    if to_return:
        if logger:
            logger.debug("Took %s to generate [%s]",
                time.time() - start_token_processing,
                to_return)
        yield to_return
    if has_seen_tool:
        if logger:
            logger.info(f"Adding tool {function_name_buffer}")
        tool_calls.append(FunctionCall(name=function_name_buffer, arguments=function_args_buffer))
    if len(tool_calls) > 0 and get_functions:
        yield tool_calls[0]


async def openai_get_tokens(gen) -> AsyncGenerator[Union[str, FunctionFragment], None]:
    async for event in gen:
        choices = event.choices or []
        if len(choices) == 0:
            break
        choice = choices[0]
        if choice.finish_reason:
            break
        delta = choice.delta or {}
        if hasattr(delta, "text") and delta.text:
            token = delta.text
            yield token
        if hasattr(delta, "content") and delta.content:
            token = delta.content
            yield token
            
        elif hasattr(delta, "tool_calls") and delta.tool_calls:
            for tool_call in delta.tool_calls:
                if tool_call.function is not None:
                    function = tool_call.function
                    yield FunctionFragment(
                        name =(
                            function.name
                            if hasattr(function, "name") and function.name
                            else ""
                        ),
                        arguments=(
                            function.arguments
                            if hasattr(function, "arguments") and function.arguments
                            else ""
                        )
                    )


def find_last_punctuation(buffer: str) -> Optional[int]:
    indices = [buffer.rfind(ending) for ending in SENTENCE_ENDINGS]
    if not indices:
        return None
    return max(indices)


def get_sentence_from_buffer(buffer: str):
    last_punctuation = find_last_punctuation(buffer)
    if last_punctuation:
        return buffer[: last_punctuation + 1], buffer[last_punctuation + 1 :]
    else:
        return None, None


def format_openai_chat_messages_from_transcript(
    transcript: Transcript, prompt_preamble: Optional[str] = None
) -> List[dict]:
    chat_messages: List[Dict[str, Optional[Any]]] = (
        [{"role": "system", "content": prompt_preamble}] if prompt_preamble else []
    )

    # merge consecutive bot messages
    new_event_logs: List[EventLog] = []
    idx = 0
    while idx < len(transcript.event_logs):
        bot_messages_buffer: List[Message] = []
        current_log = transcript.event_logs[idx]
        while isinstance(current_log, Message) and current_log.sender == Sender.BOT:
            bot_messages_buffer.append(current_log)
            idx += 1
            try:
                current_log = transcript.event_logs[idx]
            except IndexError:
                break
        if bot_messages_buffer:
            merged_bot_message = deepcopy(bot_messages_buffer[-1])
            merged_bot_message.text = " ".join(
                event_log.text for event_log in bot_messages_buffer
            )
            new_event_logs.append(merged_bot_message)
        else:
            new_event_logs.append(current_log)
            idx += 1

    for event_log in new_event_logs:
        if isinstance(event_log, Message):
            chat_messages.append(
                {
                    "role": "assistant" if event_log.sender == Sender.BOT else "user",
                    "content": event_log.text,
                }
            )
        elif isinstance(event_log, ActionStart):
            chat_messages.append(
                {
                    "role": "assistant",
                    "content": None,
                    "function_call": {
                        "name": event_log.action_type,
                        "arguments": event_log.action_input.params.json(),
                    },
                }
            )
        elif isinstance(event_log, ActionFinish):
            chat_messages.append(
                {
                    "role": "function",
                    "name": event_log.action_type,
                    "content": event_log.action_output.response.json(),
                }
            )
    return chat_messages


def vector_db_result_to_openai_chat_message(vector_db_result):
    return {"role": "user", "content": vector_db_result}
