# payload.py

`backend/open_webui/utils/payload.py` defines a set of small helpers that massage
chat payloads before they are sent to an upstream model.

These routines operate on the request body in place. They convert parameter
types, inject system prompts and translate OpenAI style payloads to
Ollama's format.  The goal is to accept flexible input from Open WebUI's HTTP
API while keeping the model specific schemas consistent.

The routers invoke these utilities when shaping a chat request. Model defaults
are merged into the payload and the dictionary is gradually adapted to match the
target backend.

All helpers **mutate** the dictionary passed to them and return the same
object.  Callers typically construct `form_data` from the HTTP request and then
apply these functions in sequence.

## System prompt injection

`apply_model_system_prompt_to_body` inserts a system message at the beginning of
the conversation. It handles the legacy `user` template variables as well as the
new metadata driven placeholders.

```python
form_data = {"messages": [{"role": "user", "content": "Hello"}]}
params = {"system": "Hello {user_name}"}
metadata = {"variables": {"user_name": "Alice"}}
apply_model_system_prompt_to_body(params, form_data, metadata)
# form_data['messages'][0] becomes the injected system message
```

Both templating helpers can be used together. Passing a `user` object provides
`{{USER_NAME}}` and `{{USER_LOCATION}}` values while `metadata["variables"]`
supplies arbitrary placeholders:

```python
from types import SimpleNamespace

form_data = {"messages": []}
params = {"system": "Hi {{USER_NAME}} from {planet}"}
metadata = {"variables": {"planet": "Mars"}}
user = SimpleNamespace(name="Bob", info={"location": "US"})

apply_model_system_prompt_to_body(params, form_data, metadata, user=user)
# -> form_data['messages'][0]['content'] == 'Hi Bob from Mars'
```

The function uses `prompt_template` from `task.py` and
`add_or_update_system_message` from `misc.py` to perform the merge.

## Parameter normalization

Models expose slightly different parameter names.  Two helpers convert common
OpenAI fields to the appropriate types and names:

- `apply_model_params_to_body_openai`
- `apply_model_params_to_body_ollama`

They iterate over a mapping of expected keys and cast functions.  For example:

```python
form = {}
params = {"temperature": "0.7", "max_tokens": "128"}
apply_model_params_to_body_openai(params, form)
# form == {"temperature": 0.7, "max_tokens": 128}
```
Behind the scenes `apply_model_params_to_body` iterates over the mapping and
casts each recognised parameter in place.  Custom mappings can be supplied for
additional backends.

The Ollama variant additionally renames `max_tokens` to `num_predict` and moves
`keep_alive` and `format` from the `options` section if present.  It also knows
about advanced options like `num_ctx`, `mirostat_eta` and `num_thread`.

## Converting message formats

Ollama expects a slightly different message structure than OpenAI.  The
`convert_messages_openai_to_ollama` helper walks through each chat entry and
splits text, images and tool calls into the layout expected by Ollama's API.

```python
messages = [
    {"role": "user", "content": [
        {"type": "text", "text": "Hi"},
        {"type": "image_url", "image_url": {"url": "img.png"}}
    ]},
]
convert_messages_openai_to_ollama(messages)
```

Tool calls are converted to Ollama's schema with an empty `content` field.  Base64
image URLs are trimmed so only the raw data remains.  A more complex example:

```python
messages = [
    {
        "role": "assistant",
        "tool_calls": [
            {"id": "call_0", "index": 0, "function": {"name": "echo", "arguments": "{}"}}
        ]
    }
]
convert_messages_openai_to_ollama(messages)
# -> [{'role': 'assistant', 'tool_calls': [{'index': 0, 'id': 'call_0',
#      'function': {'name': 'echo', 'arguments': {}}}], 'content': ''}]
```

`convert_payload_openai_to_ollama` wraps this function and copies the remaining
fields (`model`, `stream`, etc.) producing a final dictionary that can be posted
to `/api/chat` on an Ollama server.

```python
openai_payload = {
    "model": "llama2",
    "messages": messages,
    "stream": True,
    "options": {"max_tokens": 32, "system": "hello"}
}
ollama_payload = convert_payload_openai_to_ollama(openai_payload)
```

When translating the payload the helper remaps several fields. `max_tokens`
becomes `num_predict` under `options`, any `system` prompt is copied to the
top level and values like `keep_alive` or `format` are mirrored so Ollama
accepts them.

### Base64 image trimming

`convert_messages_openai_to_ollama` shortens data URLs embedded in image
messages. Only the raw base64 string is kept which greatly reduces payload size
when posting directly to an Ollama server.

## Integration with the chat flow

These helpers are called by the request routers before forwarding a chat
completion to the upstream API. When `routers/openai.py` prepares a request the
model parameters stored in the database are merged into the payload:

```python
params = model_info.params.model_dump()
payload = apply_model_params_to_body_openai(params, payload)
payload = apply_model_system_prompt_to_body(params, payload, metadata, user)
```

If the selected model is an Ollama model `utils.chat.generate_chat_completion`
converts the request right before dispatching it:

```python
if model.get("owned_by") == "ollama":
    form_data = convert_payload_openai_to_ollama(form_data)
    response = await generate_ollama_chat_completion(...)
```

This step‑wise normalisation allows the incoming OpenAI style schema to be
gradually shaped into the specific format required by each backend.

