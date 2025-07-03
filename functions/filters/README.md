# Filters Guide

Filters are lightweight plugins that run before, during, and after a pipe. Each file can define `inlet`, `stream`, and `outlet` handlers to modify chat data or emit events.

## Basic Structure

```python
from pydantic import BaseModel

class Filter:
    class Valves(BaseModel):
        priority: int = 0  # Determines the execution order for multiple filters.

    async def inlet(self, body, __event_emitter__=None):
        # Handle the incoming request body (user input and context).
        return body

    async def stream(self, event):
        # Handle streaming events (such as partial responses or intermediate actions).
        return event

    async def outlet(self, body):
        # Handle the outgoing response body before it is sent to the UI.
        return body
```

Only the methods you implement are called. `priority` controls execution order when multiple filters are active.

## Important Notes

* **Event emitters** (`__event_emitter__`) are available in the `inlet` handler but **not** passed to the `outlet` handler. Thus, you cannot emit events from the outlet.
* The `body` object passed to the `inlet` and the `body` passed to the `outlet` are **different**:

  * **Inlet Body**: Contains the initial incoming request data, user messages, model information, features, metadata, and other context.
  * **Outlet Body**: Contains the final response data, including completed messages with their responses, metadata, usage details, and sources.

### JSON Example

**Inlet Body Example**:

```python
body = {
  "stream": true,
  "model": "openai_responses.o4-mini-high",
  "messages": [
    {
      "role": "user",
      "content": "test"
    }
  ],
  "features": {
    "image_generation": false,
    "code_interpreter": false,
    "web_search": false,
    "memory": false
  },
  "metadata": {
    "user_id": "a28e8ee4-1f0c-48da-a21f-a6b869fef275",
    "chat_id": "b2673a82-998a-489f-9789-6b8831988e5b",
    "message_id": "471acf86-c233-4a28-a665-dae9d762ebbf",
    "session_id": "hQHlyQy7hOhwsdvLAABJ",
    "filter_ids": ["reason_filter"]
  },
  "reasoning_effort": "high"
}
```

**Outlet Body Example**:

```python
body = {
  "model": "openai_responses.gpt-4o",
  "messages": [
    {
      "id": "9301d638-d8c0-4a68-976a-51ffdc88c83b",
      "role": "user",
      "content": "test",
      "timestamp": 1749579867
    },
    {
      "id": "471acf86-c233-4a28-a665-dae9d762ebbf",
      "role": "assistant",
      "content": "### Hello!\nIt looks like you’re running a quick test.",
      "timestamp": 1749581306,
      "usage": {
        "input_tokens": 510,
        "output_tokens": 173,
        "total_tokens": 683,
      },
      "sources": [],
      "model": "openai_responses.o4-mini-high",
      "modelName": "openai_responses.o4-mini-high"
    }
  ],
  "filter_ids": ["reason_filter"],
  "chat_id": "b2673a82-998a-489f-9789-6b8831988e5b",
  "session_id": "hQHlyQy7hOhwsdvLAABJ",
  "id": "471acf86-c233-4a28-a665-dae9d762ebbf"
}
```

## Loading

Upload the file via the Functions API. Optional packages can be listed in a frontmatter block and will be installed automatically:

```python
"""
requirements: httpx
"""
```

The loader caches each module under `request.app.state.FUNCTIONS`.

## Valves and User Settings

Filters may expose extra options via `Valves` and `UserValves` classes. These are hydrated from the database on each call so settings can be tweaked without re-uploading the code.

Set `file_handler = True` if the filter consumes uploaded files itself. Only parameters declared in a handler's signature are provided (e.g., `body`, `event`, `__user__`).

