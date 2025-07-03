# File Handler Reference

Open WebUI reads uploaded files and injects their contents into the system prompt. Extensions can skip this behavior when they manage files themselves.

Set `self.file_handler = True` in the filter or tool `__init__` to signal that you will process file uploads. WebUI's [filter loader](https://github.com/open-webui/open-webui/blob/main/backend/open_webui/utils/filter.py) and [middleware](https://github.com/open-webui/open-webui/blob/main/backend/open_webui/utils/middleware.py) inspect this flag and delete the files from the payload once your handler runs:

```python
# Check if the function has a file_handler variable
if filter_type == "inlet" and hasattr(function_module, "file_handler"):
    skip_files = function_module.file_handler
...
if skip_files and "files" in form_data.get("metadata", {}):
    del form_data["files"]
    del form_data["metadata"]["files"]
```

When a tool declares `file_handler = True` the middleware performs a similar
cleanup:

```python
if (
    tools[tool_function_name]
    .get("metadata", {})
    .get("file_handler", False)
):
    skip_files = True

...

if skip_files and "files" in body.get("metadata", {}):
    del body["metadata"]["files"]
```

When `file_handler` is **False** (the default), the middleware collects context
from each attached file using `get_sources_from_files` and then injects that
context into a system message:

```python
context_string = ""
for source in sources:
    for doc_context, doc_meta in zip(source["document"], source["metadata"]):
        context_string += (
            f'<source id="{citation_idx[citation_id]}">' f"{doc_context}</source>\n"
        )

form_data["messages"] = add_or_update_system_message(
    rag_template(request.app.state.config.RAG_TEMPLATE, context_string, prompt),
    form_data["messages"],
)
```

See [middleware.py lines 922–980](https://github.com/open-webui/open-webui/blob/main/backend/open_webui/utils/middleware.py#L922-L980) for the full logic.

## Filters

Example:

```python
class Filter:
    def __init__(self) -> None:
        # Let the loader know we will process uploaded files ourselves
        self.file_handler = True  # disable built-in file injection
        self.citation = False  # optional, manage citations yourself
        self.valves = self.Valves()

    async def inlet(self, body: dict) -> dict:
        # Uploaded files are available under body["files"]
        return body
```

## Tools

Example:

```python
# image_tagging_tool.py

class Tools:
    def __init__(self):
        self.file_handler = True  # prevent default file injection
        self.citation = False  # disable auto citations (optional)
        self.valves = self.Valves()

```

## Pipes

Pipes do not support a `file_handler` flag. If it's not disabled via any of the enabled tools or filters, you will see the the file contents injected in the system prompt.
