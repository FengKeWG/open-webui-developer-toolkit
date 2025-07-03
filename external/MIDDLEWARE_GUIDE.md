# Middleware Overview

`backend/open_webui/utils/middleware.py` defines the heart of Open WebUI's chat pipeline.  It wires the chat REST endpoints to a collection of helpers that augment a request, invoke the model and then stream results back to the browser.  The code is long but can be decomposed into a handful of cooperating routines.

## Core concepts

The middleware coordinates several building blocks:

- **Tasks** – discrete actions such as title generation or code execution that may run in the background using a dedicated task model.
- **Features** – optional capabilities requested per chat like `memory`,
  `web_search`, `image_generation` or `code_interpreter`.
- **Pipelines & Filters** – inlet and outlet filters let extensions mutate requests or streaming responses.
- **Tools & function calls** – tools can be triggered natively via the model's function calling API or by parsing JSON and invoking Python functions directly.
- **Memory & retrieval** – conversation history, uploaded files and web search results can be merged into the prompt as retrieval‑augmented context.

## Request lifecycle

1. **Incoming chat request** arrives with model id, message list and optional files or tool specs.
2. `process_chat_payload` is called to enrich and validate the payload. It initializes helper callbacks (`get_event_emitter`, `get_event_call`) and assembles an `extra_params` dict used throughout the rest of the flow.
3. The payload is run through pipeline inlet filters. Tools, image generation, web search and retrieval are triggered as needed.
4. The final payload along with metadata and any events are passed to `generate_chat_completion`.
5. `process_chat_response` handles the streaming or non streaming response. It updates the database, fires websocket events and executes tool calls or the code interpreter where requested.

All of these helpers rely on a small set of utility functions that live in the same module.

## Function reference

### `chat_completion_tools_handler`
Handles function calling when the selected model does not support it natively.

Pseudo-code outline:
```
1. Build a prompt containing JSON tool specs using `tools_function_calling_generation_template`.
2. Call `generate_chat_completion` on the task model.
3. Parse the JSON result; for each returned tool:
   a. Validate parameters against the allowed spec.
   b. Execute the tool either directly or via event emitter.
   c. Capture string results as context snippets or citation sources.
4. Remove any file metadata if a tool handles its own uploads.
5. Return the modified body and a list of citation sources.
```

### `chat_web_search_handler`
Adds web search results to the payload.

1. Emits a `status` event that search is starting.
2. Generates search queries from the latest user message via `generate_queries`.
3. Invokes `process_web_search` which performs the search and stores documents under temporary filenames.
4. Appends file metadata for each search result and emits progress events.
5. Returns the updated form data so later steps can inject the snippets.

### `chat_image_generation_handler`
Optionally produces an image when the client requests it.

1. Emits a `status` event.
2. Optionally calls `generate_image_prompt` to craft a stable diffusion prompt from the conversation.
3. Uses `image_generations` to create the image and emits a `files` event containing its URL.
4. Adds a system message summarizing that an image was generated (or that an error occurred).

### `chat_completion_files_handler`
Retrieves context from user uploaded files or search results.

1. Generates RAG queries from the current conversation.
2. Calls `get_sources_from_files` inside a thread to avoid blocking.
3. The function returns any extracted snippets which later become citation sources.

### `apply_params_to_form_data`
Extracts optional parameters supplied by the client and merges them into the
top‑level payload. The helper understands both OpenAI and Ollama style models.

```python
form_data = {
    "model": "gpt-3.5-turbo",
    "params": {"temperature": 0.5, "max_tokens": 100},
}
model = {"id": "gpt-3.5-turbo"}

apply_params_to_form_data(form_data, model)
print(form_data)
# {
#   "model": "gpt-3.5-turbo",
#   "temperature": 0.5,
#   "max_tokens": 100
# }
```

For Ollama models the parameters are stored under `options` and a few keys
(`format`, `keep_alive`) are mirrored at the root:

```python
form_data = {
    "model": "llama2",
    "params": {"temperature": 0.7, "format": "json"},
}
model = {"id": "llama2", "ollama": True}

apply_params_to_form_data(form_data, model)
print(form_data)
# {
#   "model": "llama2",
#   "options": {"temperature": 0.7, "format": "json"},
#   "format": "json"
# }
```

Any keys reserved for Open WebUI itself (`stream_response`, `function_calling`,
`system`) are stripped from the `params` dict before the rest of the values are
merged.  Additional options may be supplied under a nested `custom_params`
object; string values there are parsed as JSON when possible and combined with
the other parameters.

Any `logit_bias` value is normalized through
`convert_logit_bias_input_to_json` so callers may provide shorthand strings.

### `process_chat_payload`
Orchestrates the full inbound flow.

Steps performed:
1. Merge `params` into the form data using `apply_params_to_form_data`.
2. Prepare the `extra_params` dictionary with user info, metadata and event helpers.
3. Choose which models are available (direct single model or global list).
4. If the model defines built‑in knowledge collections, add them to the file list so retrieval can reference them.
5. Pass the payload through `process_pipeline_inlet_filter` so extensions can modify it.
6. Resolve filter functions from the model settings and execute them via `process_filter_functions`.
7. When tools are present and the model cannot handle function calling, `chat_completion_tools_handler` is invoked.
8. Apply feature handlers in order: memory retrieval, web search, image generation and code interpreter prompts.
9. Collect context from uploaded files or search results via `chat_completion_files_handler`.
10. Construct RAG context and insert it into the system message using `rag_template` and `add_or_update_system_message`.
11. Return the final form data, metadata and any accumulated events.

### `process_chat_response`
Wraps the model response and streams it back to the client.

1. When the response is non streaming, it schedules `post_response_handler` as a background task.  This task updates chat messages, generates titles or tags and triggers webhook notifications if the user is inactive.
2. For streaming responses the function wraps the generator so each chunk is filtered and forwarded as a websocket event.
3. As tool call blocks are received they are stored, executed and their results inserted back into the conversation. Failed calls retry up to ten times.
4. If code interpreter blocks are enabled they are sent to `execute_code_jupyter` and the resulting output (including generated images) is embedded in the stream.
5. Once the stream finishes a final `chat:completion` event is emitted and the message is persisted.

This multi stage processing allows Open WebUI to offer web search, code execution, retrieval augmented generation and arbitrary tool calls all within a single chat endpoint.

## Event system

`middleware.py` relies heavily on the websocket event helpers `get_event_emitter` and `get_event_call`. These wrap asynchronous queues connected to the user's browser. Each major step (searching, executing a tool, streaming model output) emits structured events so the client can update the UI in real time.

Only a few event types trigger immediate database writes (`status`, `message`, `replace`). Others like `citation` merely update the UI. Pipelines that want to persist custom citations must update the chat record themselves, e.g. via `Chats.upsert_message_to_chat_by_id_and_message_id({"sources": [...]})`.

## Background tasks

Longer operations such as database updates, title generation or tag extraction are offloaded using `create_task`. This keeps the HTTP response snappy while ensuring chat history and metadata are stored reliably.

## Deep dive: `process_chat_payload`
`process_chat_payload` prepares the incoming chat payload for the model.  It
collects knowledge sources, runs inlet filters and handles features like web
search and tool execution before any model call is made.

At the start of the function WebUI builds an `extra_params` dictionary that is
passed to every helper:

```python
event_emitter = get_event_emitter(metadata)
event_call = get_event_call(metadata)

extra_params = {
    "__event_emitter__": event_emitter,
    "__event_call__": event_call,
    "__user__": {"id": user.id, "email": user.email, "name": user.name, "role": user.role},
    "__metadata__": metadata,
    "__request__": request,
    "__model__": model,
}
```

This structure is threaded through the pipeline so other functions can emit
events or access user and request details.

When `features` are present the handler triggers optional helpers.  For example
web search results are appended to the file list so retrieval can use them later:

```python
features = form_data.pop("features", None)
if features:
    if "memory" in features and features["memory"]:
        form_data = await chat_memory_handler(request, form_data, extra_params, user)
    if "web_search" in features and features["web_search"]:
        form_data = await chat_web_search_handler(request, form_data, extra_params, user)
    if "image_generation" in features and features["image_generation"]:
        form_data = await chat_image_generation_handler(request, form_data, extra_params, user)
    if "code_interpreter" in features and features["code_interpreter"]:
        form_data["messages"] = add_or_update_user_message(
            request.app.state.config.CODE_INTERPRETER_PROMPT_TEMPLATE
            if request.app.state.config.CODE_INTERPRETER_PROMPT_TEMPLATE != ""
            else DEFAULT_CODE_INTERPRETER_PROMPT,
            form_data["messages"],
        )
```

Finally the function constructs retrieval context from any collected sources and
inserts it as a system message using `rag_template`:

```python
if len(sources) > 0:
    context_string = ""
    for source in sources:
        for doc_context, _ in zip(source["document"], source["metadata"]):
            context_string += f"{doc_context}\n"
    form_data["messages"] = add_or_update_system_message(
        rag_template(request.app.state.config.RAG_TEMPLATE, context_string, get_last_user_message(form_data["messages"])),
        form_data["messages"],
    )
```

The returned tuple `(form_data, metadata, events)` feeds directly into
`generate_chat_completion`.

## Deep dive: `process_chat_response`
`process_chat_response` receives the raw model output and turns it into events that the browser understands.  The function distinguishes between streaming and non‑streaming replies, spawns background tasks for post‑processing and wraps streaming generators so every chunk can be filtered and emitted to the websocket.

The inner `post_response_handler` consolidates streamed blocks into full messages:

```python
  1188	        def split_content_and_whitespace(content):
  1189	            content_stripped = content.rstrip()
  1190	            original_whitespace = (
  1191	                content[len(content_stripped) :]
  1192	                if len(content) > len(content_stripped)
  1193	                else ""
  1194	            )
  1195	            return content_stripped, original_whitespace
  1196	
  1197	        def is_opening_code_block(content):
  1198	            backtick_segments = content.split("```")
  1199	            # Even number of segments means the last backticks are opening a new block
  1200	            return len(backtick_segments) > 1 and len(backtick_segments) % 2 == 0
  1201	
  1202	        # Handle as a background task
  1203	        async def post_response_handler(response, events):
  1204	            def serialize_content_blocks(content_blocks, raw=False):
  1205	                content = ""
  1206	
  1207	                for block in content_blocks:
  1208	                    if block["type"] == "text":
```

This handler assembles the streamed `tool_calls`, reasoning blocks and code interpreter output into final messages before persisting them.

`serialize_content_blocks` walks over a list of block dictionaries and builds a
single string for storage or display. Plain `text` blocks are appended as-is.
`tool_calls` blocks become `<details>` tags showing the tool name, arguments and
an "Executing..." placeholder until results are attached. `reasoning` and
`code_interpreter` sections also expand into `<details>` wrappers with optional
duration or output attributes. When `raw=True` the underlying tags are emitted
directly; otherwise they are converted to human-friendly HTML.

After the model produces a list of tool calls, the handler appends a
`{"type": "tool_calls"}` block to `content_blocks` and immediately emits a
`chat:completion` event using the serialized content. This lets the UI display
"Executing..." while the tool is invoked via `event_caller` or a direct
call. The tool's result is stored in the same block's `results` field and
another `chat:completion` update follows.

When streaming, the original generator is wrapped so extra events can be injected and each chunk is run through any outlet filters:

```python
    else:
        # Fallback to the original response
        async def stream_wrapper(original_generator, events):
            def wrap_item(item):
                return f"data: {item}\n\n"

            for event in events:
                event, _ = await process_filter_functions(
                    request=request,
                    filter_functions=filter_functions,
                    filter_type="stream",
                    form_data=event,
                    extra_params=extra_params,
                )

                if event:
                    yield wrap_item(json.dumps(event))

            async for data in original_generator:
                data, _ = await process_filter_functions(
                    request=request,
                    filter_functions=filter_functions,
                    filter_type="stream",
                    form_data=data,
                    extra_params=extra_params,
                )

                if data:
                    yield data

        return StreamingResponse(
            stream_wrapper(response.body_iterator, events),
            headers=dict(response.headers),
            background=response.background,
        )
```

The wrapper yields any queued events before forwarding chunks from the language model.  Each payload is also passed through configured outlet filters so extensions can modify the stream.
### Async streaming strategy

`stream_body_handler` iterates over `response.body_iterator` using `async for` so tokens can be parsed and forwarded the moment they arrive. Each piece of data is filtered then emitted to the websocket via `event_emitter`, ensuring the UI updates without blocking the main event loop. The surrounding `stream_wrapper` is itself an async generator passed to `StreamingResponse`, which preserves backpressure when the client is slow to consume updates.

When the request does not require streaming or no websocket session is active, the middleware returns the raw `Response` and schedules any heavy post-processing with `create_task`. Skipping the async wrapper in these cases keeps the code path lean and avoids unnecessary overhead.

#### Avoiding async overhead

Only the streaming path needs to read and emit tokens one by one. When no websocket is attached or the model response is non-streaming, `post_response_handler` is scheduled via `create_task` so the HTTP request finishes immediately while tool execution and message updates continue in the background.

Wrapping a plain `Response` in an async generator would add unnecessary context switching and memory allocations. By reserving `stream_wrapper` and `stream_body_handler` for true streaming scenarios, the middleware minimizes CPU usage and keeps latency low.

## Deep dive: `chat_completion_files_handler`
`chat_completion_files_handler` collects retrieval context from uploaded files or search results. It is called after any tool or search handlers and augments the chat payload with snippets found in those files.

Steps performed:
1. Look for a `files` list in `body['metadata']`. If absent the function simply returns.
2. Ask the model to generate RAG queries via `generate_queries`. The JSON response is parsed, falling back to the raw text if needed.
3. If no queries are produced, the latest user message is used instead.
4. To avoid blocking the event loop, `get_sources_from_files` is executed in a `ThreadPoolExecutor`. This helper performs vector search and optional reranking to produce relevant snippets.
5. The gathered `sources` are returned to be injected later by `process_chat_payload`.

The offload to a worker thread is shown below:

```python
loop = asyncio.get_running_loop()
with ThreadPoolExecutor() as executor:
    sources = await loop.run_in_executor(
        executor,
        lambda: get_sources_from_files(
            request=request,
            files=files,
            queries=queries,
            embedding_function=lambda q, prefix: request.app.state.EMBEDDING_FUNCTION(
                q, prefix=prefix, user=user
            ),
            k=request.app.state.config.TOP_K,
            reranking_function=request.app.state.rf,
            k_reranker=request.app.state.config.TOP_K_RERANKER,
            r=request.app.state.config.RELEVANCE_THRESHOLD,
            hybrid_bm25_weight=request.app.state.config.HYBRID_BM25_WEIGHT,
            hybrid_search=request.app.state.config.ENABLE_RAG_HYBRID_SEARCH,
            full_context=request.app.state.config.RAG_FULL_CONTEXT,
        ),
    )
```

By delegating the retrieval work to a thread, the main FastAPI server remains responsive while still making use of the synchronous `get_sources_from_files` implementation.
## Deep dive: `chat_completion_tools_handler`
`chat_completion_tools_handler` orchestrates manual tool invocation when the underlying model lacks native function calling. It crafts a tool request prompt, parses the result and executes each selected tool.

The helper first builds a secondary payload aimed at the task model:
```python
def get_tools_function_calling_payload(messages, task_model_id, content):
        user_message = get_last_user_message(messages)
        history = "\n".join(
            f"{message['role'].upper()}: \"\"\"{message['content']}\"\"\""
            for message in messages[::-1][:4]
        )

        prompt = f"History:\n{history}\nQuery: {user_message}"

        return {
            "model": task_model_id,
            "messages": [
                {"role": "system", "content": content},
                {"role": "user", "content": f"Query: {prompt}"},
            ],
            "stream": False,
            "metadata": {"task": str(TASKS.FUNCTION_CALLING)},
        }

```
This payload contains a history summary and instructs the task model to return JSON describing which tools to call.

When the response arrives each tool is executed in turn. The logic validates parameters, dispatches direct tools via the event system and captures outputs for use as context or citations:
```python
            async def tool_call_handler(tool_call):
                nonlocal skip_files

                log.debug(f"{tool_call=}")

                tool_function_name = tool_call.get("name", None)
                if tool_function_name not in tools:
                    return body, {}

                tool_function_params = tool_call.get("parameters", {})

                try:
                    tool = tools[tool_function_name]

                    spec = tool.get("spec", {})
                    allowed_params = (
                        spec.get("parameters", {}).get("properties", {}).keys()
                    )
                    tool_function_params = {
                        k: v
                        for k, v in tool_function_params.items()
                        if k in allowed_params
                    }

                    if tool.get("direct", False):
                        tool_result = await event_caller(
                            {
                                "type": "execute:tool",
                                "data": {
                                    "id": str(uuid4()),
                                    "name": tool_function_name,
                                    "params": tool_function_params,
                                    "server": tool.get("server", {}),
                                    "session_id": metadata.get("session_id", None),
                                },
                            }
                        )
                    else:
                        tool_function = tool["callable"]
                        tool_result = await tool_function(**tool_function_params)

                except Exception as e:
                    tool_result = str(e)

                tool_result_files = []
                if isinstance(tool_result, list):
                    for item in tool_result:
                        # check if string
                        if isinstance(item, str) and item.startswith("data:"):
                            tool_result_files.append(item)
                            tool_result.remove(item)

                if isinstance(tool_result, dict) or isinstance(tool_result, list):
                    tool_result = json.dumps(tool_result, indent=2)

                if isinstance(tool_result, str):
                    tool = tools[tool_function_name]
                    tool_id = tool.get("tool_id", "")

                    tool_name = (
                        f"{tool_id}\/{tool_function_name}"
                        if tool_id
                        else f"{tool_function_name}"
                    )
                    if tool.get("metadata", {}).get("citation", False) or tool.get(
                        "direct", False
                    ):
                        # Citation is enabled for this tool
                        sources.append(
                            {
                                "source": {
                                    "name": (f"TOOL:{tool_name}"),
                                },
                                "document": [tool_result],
                                "metadata": [{"source": (f"TOOL:{tool_name}")}],
                            }
                        )
                    else:
                        # Citation is not enabled for this tool
                        body["messages"] = add_or_update_user_message(
                            f"\\nTool `{tool_name}` Output: {tool_result}",
                            body["messages"],
                        )

                    if (
                        tools[tool_function_name]
                        .get("metadata", {})
                        .get("file_handler", False)
                    ):
                        skip_files = True

            # check if "tool_calls" in result
            if result.get("tool_calls"):
                for tool_call in result.get("tool_calls"):
                    await tool_call_handler(tool_call)
            else:
                await tool_call_handler(result)

        except Exception as e:
            log.debug(f"Error: {e}")
            content = None
    except Exception as e:
        log.debug(f"Error: {e}")
        content = None

    log.debug(f"tool_contexts: {sources}")

    if skip_files and "files" in body.get("metadata", {}):
        del body["metadata"]["files"]

    return body, {"sources": sources}


```
The handler also removes `metadata['files']` when a tool reports it handled uploads itself. The function returns the updated chat body and any source snippets so later stages can inject them into the conversation.

## Deep dive: `chat_web_search_handler`
`chat_web_search_handler` performs on‑the‑fly web searches and attaches the results to the chat payload.  It emits progress updates so the client knows when search is happening and stores retrieved documents under temporary filenames so later steps can fetch the snippets.

Steps performed:
1. Send an initial `status` event marking that a search query is being generated.
2. Use `generate_queries` to craft search terms from the latest user message. The JSON response is parsed or the raw text is used as a fallback.
3. If query generation fails the user's message becomes the sole query. When the resulting list is empty a completion event is sent and the handler returns.
4. When queries exist, `process_web_search` is called which downloads pages and stores them in the retrieval system.
5. Each returned collection or document is appended to `form_data['files']` with the originating `queries` list so RAG can read them later. A summary event reports the visited URLs.
6. Errors are caught and reported back to the client via `status` events.

Below is the core logic showing how queries are built and search results attached:

```python
async def chat_web_search_handler(request: Request, form_data: dict, extra_params: dict, user):
    event_emitter = extra_params["__event_emitter__"]
    await event_emitter({
        "type": "status",
        "data": {"action": "web_search", "description": "Generating search query", "done": False},
    })

    messages = form_data["messages"]
    user_message = get_last_user_message(messages)
    queries = []
    try:
        res = await generate_queries(
            request,
            {"model": form_data["model"], "messages": messages, "prompt": user_message, "type": "web_search"},
            user,
        )
        response = res["choices"][0]["message"]["content"]
        try:
            bracket_start = response.find("{")
            bracket_end = response.rfind("}") + 1
            if bracket_start == -1 or bracket_end == -1:
                raise Exception("No JSON object found in the response")
            response = response[bracket_start:bracket_end]
            queries = json.loads(response).get("queries", [])
        except Exception:
            queries = [response]
    except Exception:
        log.exception("query generation failed")
        queries = [user_message]

    if len(queries) == 0:
        await event_emitter({
            "type": "status",
            "data": {"action": "web_search", "description": "No search query generated", "done": True},
        })
        return form_data

    await event_emitter({
        "type": "status",
        "data": {"action": "web_search", "description": "Searching the web", "done": False},
    })

    try:
        results = await process_web_search(request, SearchForm(queries=queries), user=user)
        if results:
            files = form_data.get("files", [])
            if results.get("collection_names"):
                for collection_name in results.get("collection_names"):
                    files.append({
                        "collection_name": collection_name,
                        "name": ", ".join(queries),
                        "type": "web_search",
                        "urls": results["filenames"],
                    })
            elif results.get("docs"):
                files.append({
                    "docs": results["docs"],
                    "name": ", ".join(queries),
                    "type": "web_search",
                    "urls": results["filenames"],
                })

            form_data["files"] = files
            await event_emitter({
                "type": "status",
                "data": {"action": "web_search", "description": "Searched {{count}} sites", "urls": results["filenames"], "done": True},
            })
        else:
            await event_emitter({
                "type": "status",
                "data": {"action": "web_search", "description": "No search results found", "done": True, "error": True},
            })
    except Exception:
        log.exception("search failed")
        await event_emitter({
            "type": "status",
            "data": {"action": "web_search", "description": "An error occurred while searching the web", "queries": queries, "done": True, "error": True},
        })

    return form_data
```

The function's extensive event reporting keeps the user informed while retrieval happens in the background. The returned `form_data` now contains file records referencing the downloaded pages so `chat_completion_files_handler` can later embed their contents.
