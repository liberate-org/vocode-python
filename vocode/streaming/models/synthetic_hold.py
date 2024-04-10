from typing import (
  List,
  Optional
)
from pydantic import BaseModel

class SyntheticHoldConfig(BaseModel):
  enabled: bool = True
  audio_url: Optional[str]
  preflight_messages:  List[str] = []
  resume_messages: List[str] = []