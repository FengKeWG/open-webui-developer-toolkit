# chat.py

`backend/open_webui/utils/chat.py` coordinates chat completions across multiple
sources.

It acts as the main entry point for the middleware once the request payload has
been normalised.  The helpers in this file decide **which backend** to invoke
and stream results back to the client.

Key points:
* Chooses between OpenAI, Ollama or a custom pipe based on the selected model.
* Applies inlet/outlet filters through `routers.pipelines` when configured.
* Exposes `generate_direct_chat_completion`, `generate_chat_completion`,
  `chat_completed` and `chat_action` used by the HTTP routes.

The rest of this document focuses on `generate_chat_completion` which performs
the main routing logic.

## `generate_chat_completion`

This asynchronous function receives the processed form data and determines how
the chat should be executed.  When the request originated from a "direct"
connection (typically from another WebUI instance) it forwards the payload over
the websocket channel.  Otherwise it performs a number of checks and invokes the
appropriate backend.

Pseudo-code outline:

```
if request.state.direct:
    return generate_direct_chat_completion(...)

if not bypass_filter and user.role == "user":
    check_model_access(user, model)

if model.owned_by == "arena":
    # pick a random model and re-enter generate_chat_completion
    ...
elif model.pipe:
    return generate_function_chat_completion(...)
elif model.owned_by == "ollama":
    convert_payload_openai_to_ollama()
    return generate_ollama_chat_completion(...)
else:
    return generate_openai_chat_completion(...)
```

### Streaming direct completions

When a direct connection is used with `stream=True` the function registers a
socket listener and yields chunks as they arrive.  The relevant section is shown
below (lines 81‑136 in the upstream file):

```python
q = asyncio.Queue()

async def message_listener(sid, data):
    await q.put(data)

sio.on(channel, message_listener)

res = await event_caller({
    "type": "request:chat:completion",
    "data": {
        "form_data": form_data,
        "model": models[form_data["model"]],
        "channel": channel,
        "session_id": session_id,
    },
})

if res.get("status", False):
    async def event_generator():
        while True:
            data = await q.get()
            if isinstance(data, dict) and data.get("done"):
                break
            yield f"data: {json.dumps(data)}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

### Arena model selection

Models owned by `"arena"` act as wrappers that randomly choose another model at
runtime.  After selecting the model id the function re-invokes itself so the
normal logic applies.  A simplified excerpt (lines 203‑249) illustrates the
approach:

```python
model_ids = model.get("info", {}).get("meta", {}).get("model_ids")
if model_ids and filter_mode == "exclude":
    model_ids = [m["id"] for m in request.app.state.MODELS.values()
                 if m.get("owned_by") != "arena" and m["id"] not in model_ids]

selected_model_id = random.choice(model_ids)
form_data["model"] = selected_model_id

if form_data.get("stream"):
    async def stream_wrapper(stream):
        yield f"data: {json.dumps({'selected_model_id': selected_model_id})}\n\n"
        async for chunk in stream:
            yield chunk
    response = await generate_chat_completion(request, form_data, user, bypass_filter=True)
    return StreamingResponse(stream_wrapper(response.body_iterator), media_type="text/event-stream")
else:
    result = await generate_chat_completion(request, form_data, user, bypass_filter=True)
    return {**result, "selected_model_id": selected_model_id}
```

### Outlet filters

After a model has produced a response `chat_completed` is called.  It gathers the
configured filter functions for the model and executes them via
`process_filter_functions`:

```python
filter_functions = [Functions.get_function_by_id(fid)
                    for fid in get_sorted_filter_ids(model)]
result, _ = await process_filter_functions(
    request=request,
    filter_functions=filter_functions,
    filter_type="outlet",
    form_data=data,
    extra_params=extra_params,
)
```

This mechanism lets extensions modify the final message before it is emitted to
the client.

### Chat actions

`chat_action` runs extension-defined actions on a given chat message. The caller
sends an action identifier along with the normal form data. The function loads
the corresponding module, injects context parameters and then calls its
`action` function.

The first part deals with locating the module and building an event helper pair:

```python
if "." in action_id:
    action_id, sub_action_id = action_id.split(".")
else:
    sub_action_id = None

if action_id in request.app.state.FUNCTIONS:
    function_module = request.app.state.FUNCTIONS[action_id]
else:
    function_module, _, _ = load_function_module_by_id(action_id)
    request.app.state.FUNCTIONS[action_id] = function_module
```

Each call is associated with a message so an event emitter and event caller are
constructed using its metadata (lines 375‑390) so the action can dispatch socket
events when necessary.

Before invocation the loader hydrates any `valves` attached to the action and
prepares a parameter dictionary. Only arguments present in the function
signature are added:

```python
sig = inspect.signature(action)
params = {"body": data}
extra_params = {
    "__model__": model,
    "__id__": sub_action_id if sub_action_id is not None else action_id,
    "__event_emitter__": __event_emitter__,
    "__event_call__": __event_call__,
    "__request__": request,
}
for key, value in extra_params.items():
    if key in sig.parameters:
        params[key] = value
```

If `__user__` appears in the signature the user's info and per-user valve
settings are merged in. The handler may be a coroutine or a regular function—the
wrapper handles both cases transparently.

In practice a custom action can look like:

```python
class Action:
    def action(self, body, __model__, __event_emitter__, __user__):
        __event_emitter__("log", {"msg": f"Using model {__model__['name']}"})
        return {"echo": body["messages"][-1]["content"]}
```

The module can then be triggered from the API by posting `{"actionId": "foo",
"model": "gpt-4", ...}` and `chat_action` will load `foo`, pass in the extra
context and return the resulting dictionary.

## Chat history persistence

`backend/open_webui/models/chats.py` stores each conversation in a JSON field.
A chat record roughly looks like:

```json
{
  "title": "New Chat",
  "history": {
    "currentId": "m1",
    "messages": {
      "m1": {"role": "user", "content": "hi"}
    }
  }
}
```

`upsert_message_to_chat_by_id_and_message_id(id, message_id, data)` merges
`data` into the existing message if present or inserts it otherwise.  There is
**no schema enforcement**, so any keys provided are stored as is. The builtin
middleware writes plain strings under `content`, but custom pipes can store lists
or additional metadata. For example:

```python
Chats.upsert_message_to_chat_by_id_and_message_id(
    chat_id,
    "msg-123",
    {
        "role": "assistant",
        "content": "hi",
        "custom": True,
        "processed_at": 1712345678,
    },
)
```

During streaming the middleware may call this helper on every delta when
`ENABLE_REALTIME_CHAT_SAVE` is enabled. Otherwise a final `chat:completion`
event triggers one write with the complete content. Existing fields are merged
so any metadata added earlier remains intact.

Rendering helpers such as `get_content_from_message` only inspect the `content`
field, meaning custom keys (like tool metadata) are ignored by the default UI
unless additional code handles them.

## Tool and code interpreter blocks

`backend/open_webui/utils/middleware.py` inserts `<details>` elements into the
message `content` whenever tool calls or code interpreter blocks are processed.
The relevant helper converts `content_blocks` to messages and produces HTML such
as:

```html
<details type="tool_calls" done="true" id="123" name="my_tool" arguments="{}" result="\"ok\"">
<summary>Tool Executed</summary>
</details>
```

These tags allow the built‑in renderer to show tool execution progress.
