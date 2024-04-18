import os
from jinja2 import Environment, FileSystemLoader
from fastapi import Response
from typing import Optional

class Templater:
    def __init__(self):
        self.templates = Environment(
            loader=FileSystemLoader("%s/templates/" % os.path.dirname(__file__))
        )

    def render_template(self, template_name: str, **kwargs):
        template = self.templates.get_template(template_name)
        return template.render(**kwargs)

    def get_connection_twiml(self, call_id: str, base_url: str, answer_delay: Optional[int] = 0):
        return Response(
            self.render_template("connect_call.xml", base_url=base_url, id=call_id, answer_delay=answer_delay),
            media_type="application/xml",
        )
