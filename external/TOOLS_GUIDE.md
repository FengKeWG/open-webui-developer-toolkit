# tools.py

`backend/open_webui/utils/tools.py` discovers tool functions by id and prepares them for execution.

It handles:
- Loading tool modules via `plugin.load_tool_module_by_id`.
- Converting Python callables or tool server endpoints into async functions.
- Parsing type hints and docstrings with `convert_function_to_pydantic_model`.
- Building OpenAI style JSON specs from Pydantic models.
- Integrating with **OpenAPI tool servers** for remote actions.

## Local tool loading

`get_tools(request, tool_ids, user, extra_params)` retrieves tool modules from
the database and returns a dictionary mapping **function names** to callables and
metadata. It transparently handles per‑user valve configuration and converts
each function into an awaitable:

```python
tool_map = get_tools(request, ["calculator"], user, {"__user__": user.model_dump()})

add_spec = tool_map["add"]["spec"]  # OpenAI style function schema
result = await tool_map["add"]["callable"](a=1, b=2)
```

The helper loads the module with
`plugin.load_tool_module_by_id`, sets any `Valves` or `UserValves` objects, and
filters out internal parameters such as `__id__` before returning the spec.

## Pydantic conversion deep dive

`convert_function_to_pydantic_model` inspects a Python function and converts its
signature plus docstring into a Pydantic model. Parameter descriptions defined
with `:param` are copied over to the schema:

```python
def greet(name: str, excited: bool = False):
    """Return a greeting.

    :param name: User name
    :param excited: Add an exclamation mark
    """
    suffix = "!" if excited else "."
    return f"Hello {name}{suffix}"

Model = convert_function_to_pydantic_model(greet)
print(Model.model_json_schema())
```

This schema is then fed through `convert_pydantic_model_to_openai_function_spec`
so tools appear as ChatGPT compatible definitions.

## Tool server integration

Tool servers expose functions over HTTP by serving an OpenAPI document. The
utilities below download the spec, convert each operation into a ChatGPT style
tool payload and later proxy the call:

```python
from open_webui.utils.tools import (
    get_tool_servers_data,
    execute_tool_server,
)

servers = [
    {"url": "https://example.com/api", "config": {"enable": True}},
]

tool_servers = await get_tool_servers_data(servers)
spec = tool_servers[0]["specs"][0]  # first operation as a tool payload

result = await execute_tool_server(
    token="",
    url=tool_servers[0]["url"],
    name=spec["name"],
    params={"foo": "bar"},
    server_data=tool_servers[0],
)
```

`convert_openapi_to_tool_payload` translates each `operationId` to the JSON
structure expected by the OpenAI API. The snippet below shows the core loop
around lines 365–438 where parameters are collected and request bodies are
flattened into a single schema:

```python
for path, methods in openapi_spec.get("paths", {}).items():
    for method, operation in methods.items():
        if operation.get("operationId"):
            tool = {
                "type": "function",
                "name": operation.get("operationId"),
                "description": operation.get(
                    "description",
                    operation.get("summary", "No description available."),
                ),
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
            # path and query parameters are mapped to the tool schema
```

These lines continue by resolving request body schemas and appending the final
tool object to the list returned by the function.

### Configuration and caching

Connections to tool servers are defined in the `TOOL_SERVER_CONNECTIONS`
environment variable. `config.py` loads this JSON list at startup and stores it
as a persistent setting【F:external/open-webui/backend/open_webui/config.py†L884-L897】.
When the `/tools` endpoint is first called the router fetches each server's
OpenAPI spec via `get_tool_servers_data` and caches the result in
`app.state.TOOL_SERVERS`【F:external/open-webui/backend/open_webui/routers/tools.py†L38-L45】.
The returned list of tools includes these remote operations by enumerating each
server with an index before merging the specs with local tools【F:external/open-webui/backend/open_webui/routers/tools.py†L48-L71】.
