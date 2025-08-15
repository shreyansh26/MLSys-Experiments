#!/usr/bin/env python3
# render_chat_template.py
#
# Usage examples:
#   python render_chat_template.py --system "You are helpful." --user "How do I sort a list?"
#   python render_chat_template.py --system "You are helpful." --user "Search the web" --tools-json '[{"name":"search","description":"web search","parameters":{"q":"string"}}]'
#   python render_chat_template.py --system "Be terse." --user "Hi" --bos-token "<|begin_of_text|>" --date-string "15 Aug 2025"

import argparse
import json
import datetime as _dt
from jinja2 import Environment, StrictUndefined

CHAT_TEMPLATE = r"""{{- bos_token }}
{%- if custom_tools is defined %}
    {%- set tools = custom_tools %}
{%- endif %}
{%- if not tools_in_user_message is defined %}
    {%- set tools_in_user_message = true %}
{%- endif %}
{%- if not date_string is defined %}
    {%- if strftime_now is defined %}
        {%- set date_string = strftime_now("%d %b %Y") %}
    {%- else %}
        {%- set date_string = "26 Jul 2024" %}
    {%- endif %}
{%- endif %}
{%- if not tools is defined %}
    {%- set tools = none %}
{%- endif %}

{#- This block extracts the system message, so we can slot it into the right place. #}
{%- if messages[0]['role'] == 'system' %}
    {%- set system_message = messages[0]['content']|trim %}
    {%- set messages = messages[1:] %}
{%- else %}
    {%- set system_message = "" %}
{%- endif %}

{#- System message #}
{{- "<|start_header_id|>system<|end_header_id|>\n\n" }}
{%- if tools is not none %}
    {{- "Environment: ipython\n" }}
{%- endif %}
{{- "Cutting Knowledge Date: December 2023\n" }}
{{- "Today Date: " + date_string + "\n\n" }}
{%- if tools is not none and not tools_in_user_message %}
    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
{%- endif %}
{{- system_message }}
{{- "<|eot_id|>" }}

{#- Custom tools are passed in a user message with some extra guidance #}
{%- if tools_in_user_message and not tools is none %}
    {#- Extract the first user message so we can plug it in here #}
    {%- if messages | length != 0 %}
        {%- set first_user_message = messages[0]['content']|trim %}
        {%- set messages = messages[1:] %}
    {%- else %}
        {{- raise_exception("Cannot put tools in the first user message when there's no first user message!") }}
{%- endif %}
    {{- '<|start_header_id|>user<|end_header_id|>\n\n' -}}
    {{- "Given the following functions, please respond with a JSON for a function call " }}
    {{- "with its proper arguments that best answers the given prompt.\n\n" }}
    {{- 'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.' }}
    {{- "Do not use variables.\n\n" }}
    {%- for t in tools %}
        {{- t | tojson(indent=4) }}
        {{- "\n\n" }}
    {%- endfor %}
    {{- first_user_message + "<|eot_id|>"}}
{%- endif %}

{%- for message in messages %}
    {%- if not (message.role == 'ipython' or message.role == 'tool' or 'tool_calls' in message) %}
        {{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' }}
    {%- elif 'tool_calls' in message %}
        {%- if not message.tool_calls|length == 1 %}
            {{- raise_exception("This model only supports single tool-calls at once!") }}
        {%- endif %}
        {%- set tool_call = message.tool_calls[0].function %}
        {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' -}}
        {{- '{"name": "' + tool_call.name + '", ' }}
        {{- '"parameters": ' }}
        {{- tool_call.arguments | tojson }}
        {{- "}" }}
        {{- "<|eot_id|>" }}
    {%- elif message.role == "tool" or message.role == "ipython" %}
        {{- "<|start_header_id|>ipython<|end_header_id|>\n\n" }}
        {%- if message.content is mapping or message.content is iterable %}
            {{- message.content | tojson }}
        {%- else %}
            {{- message.content }}
        {%- endif %}
        {{- "<|eot_id|>" }}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
    {{- '<|start_header_id|>assistant<|end_header_id|>\n\n' }}
{%- endif %}
"""

def _tojson(obj, indent=None):
    # Jinja2 usually has a built-in `tojson`, but we provide one to be safe.
    return json.dumps(obj, indent=indent, ensure_ascii=False)

class TemplateAbort(Exception):
    pass

def raise_exception(msg: str):
    raise TemplateAbort(str(msg))

def strftime_now(fmt: str) -> str:
    return _dt.datetime.now().strftime(fmt)

def build_env():
    env = Environment(
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    env.filters["tojson"] = _tojson
    env.globals["raise_exception"] = raise_exception
    env.globals["strftime_now"] = strftime_now
    return env

def render(
    system: str,
    user: str,
    tools=None,
    bos_token: str = "<|begin_of_text|>",
    date_string: str | None = None,
    tools_in_user_message: bool | None = None,
    add_generation_prompt: bool = True,
    extra_messages: list[dict] | None = None,
    template_str: str = CHAT_TEMPLATE,
) -> str:
    env = build_env()
    tmpl = env.from_string(template_str)

    # Build the baseline messages list (system + user)
    messages = [{"role": "system", "content": system}]
    if user:
        messages.append({"role": "user", "content": user})
    # Optionally append any additional conversation turns
    if extra_messages:
        messages.extend(extra_messages)

    # Build context carefully: only set vars that should be "defined"
    ctx = {"messages": messages, "bos_token": bos_token}

    if tools is not None:
        # The template looks for `custom_tools` to copy into `tools`
        ctx["custom_tools"] = tools

    if date_string is not None:
        # If we provide date_string, it becomes "defined" and overrides the default
        ctx["date_string"] = date_string

    if tools_in_user_message is not None:
        # If omitted, the template defaults this to true
        ctx["tools_in_user_message"] = bool(tools_in_user_message)

    # Whether to add the assistant header at the end
    ctx["add_generation_prompt"] = bool(add_generation_prompt)

    return tmpl.render(**ctx)

def main():
    ap = argparse.ArgumentParser(description="Render a Jinja2 chat template.")
    ap.add_argument("--system", required=True, help="System prompt content")
    ap.add_argument("--user", required=True, help="User prompt content")
    ap.add_argument("--bos-token", default="<|begin_of_text|>", help="Value for {{ bos_token }}")
    ap.add_argument("--date-string", default=None, help="e.g., '15 Aug 2025'; omit to use template default")
    ap.add_argument("--tools-json", default=None, help="JSON array of tool specs")
    ap.add_argument("--tools-in-user-message", choices=["true", "false"], default=None,
                    help="If set, overrides the template default (true)")
    ap.add_argument("--no-add-generation-prompt", action="store_true",
                    help="If set, do not append assistant header at the end")
    ap.add_argument("--template-path", default=None, help="Optional path to a template file to use instead")
    args = ap.parse_args()

    tools = None
    if args.tools_json:
        tools = json.loads(args.tools_json)

    add_generation_prompt = not args.no_add_generation_prompt
    tools_in_user_message = None
    if args.tools_in_user_message is not None:
        tools_in_user_message = (args.tools_in_user_message.lower() == "true")

    template_str = CHAT_TEMPLATE
    if args.template_path:
        with open(args.template_path, "r", encoding="utf-8") as f:
            template_str = f.read()

    out = render(
        system=args.system,
        user=args.user,
        tools=tools,
        bos_token=args.bos_token,
        date_string=args.date_string,
        tools_in_user_message=tools_in_user_message,
        add_generation_prompt=add_generation_prompt,
        template_str=template_str,
    )
    print(out)

if __name__ == "__main__":
    main()
