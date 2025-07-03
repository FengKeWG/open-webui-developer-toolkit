# Citations Example Pipe

## Overview

The Citations Example demonstrates how an Open WebUI pipeline can attach **inline citations** to an assistant's response.

### Incremental Citation Events (Streaming)

The example demonstrates emitting citations **incrementally** during response streaming. First, yield the citation placeholder `[1]`, then immediately emit the corresponding citation event.

```python
        # 1️⃣  Yield the source number (must start at one!)
        yield "[1]"

        # 2️⃣  Immediately emit the matching citation
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "source",        # "source" (preferred) or "citation"
                    "data": {
                        "source": {"name": "NASA"},
                        "document": [
                            "299 792 458 metres per second is the exact speed of light in vacuum."
                        ],
                        "metadata": [
                            {
                                "source": "https://science.nasa.gov/ems/03_movinglight/",
                                "date_accessed": "2025-06-24"
                            }
                        ],
                    },
                }
            )
```

### Single Event Emission (All at Once)

As an alternative, Open WebUI also allows sending all the citations in one go, usually at the end of the answer. Instead of emitting multiple small events, the pipeline can send a **single consolidated event** containing the entire answer and all sources. This is done using the `chat:completion` event type:

* **One final event:** The pipe would construct the full answer with citations and then call `__event_emitter__` once, with `type: "chat:completion"`. In the `data`, it includes the complete answer content (with `[n]` markers included) and a `sources` list containing all citation entries. For example, the event data might look like:

  ```python
  {
    "type": "chat:completion",
    "data": {
        "content": "This example cites two references [1][2].",
        "done": True,
        "sources": [
            { "source": {"name": "Example Source 1"},
              "document": ["Example document snippet one."],
              "metadata": [{"source": "https://example.com/1"}]
            },
            { "source": {"name": "Example Source 2"},
              "document": ["Another snippet from a second source."],
              "metadata": [{"source": "https://example.com/2"}]
            }
        ]
    }
  }
  ```

  (Here `"done": True` signifies the completion of the response stream.) The toolkit documentation shows this exact structure being used in the citations example.

* **Frontend handling:** When this single event is received, the frontend will treat it similarly: it will display the provided `content` as the assistant message and attach the `sources` list to that message. The placeholders `[1]` and `[2]` in the content will be recognized and rendered as citations (just as with the incremental approach).

* **Streaming Incrementally vs all at once** If you use a one-time `chat:completion` event without any prior yielding, the user will see the entire answer appear at once when the event arrives (rather than gradual word-by-word streaming). On the other hand, you can also combine approaches: for instance, *yield* the answer text normally for streaming, and then emit a final `chat:completion` event at the end that contains only the `sources` array. In that hybrid case, you would set the `data.content` to an empty string (`""`) in the final event so that it **does not overwrite** the text already shown, while still delivering the citations list. (Always including a `content` field – even if empty – otherwise the UI will become unresponsive.).

Both methods achieve the same end result: the message content contains inline \[number] references, and the message's `sources` data holds the details for each citation. The choice between incremental vs single emission depends on the use case – incremental updates can be more interactive, whereas a single batched update might be simpler for static responses.

## Frontend Handling of Citations

Once the backend emits the citation events (by either method), the Open WebUI frontend takes care of integrating them into the chat display:

* **Merging Citation Events:** The chat client listens for incoming events. The Open WebUI frontend is designed to accept events of type `"source"` or `"citation"` and merge their data into the current assistant message. Essentially, as each citation event arrives, the frontend code appends the provided data entry to the message’s `sources` list. If no sources list exists yet on that message, it will create one. This happens transparently during streaming, before the message rendering is finalized.

* **Rendering Inline Markers:** When the assistant's message is finally rendered, the frontend scans the message text for citation patterns. Any occurrence of **`[n]`** (where *n* is 1, 2, 3, ...) that appears outside of code blocks is identified as a citation marker. The renderer will replace each such token with a special `<source_id>` element that links to the corresponding source data. For example, the text "`46.5 hours [1]`" will be rendered with a clickable reference in place of the literal "\[1]". The mapping is based on the order of the sources: the first source in the list becomes \[1], the second becomes \[2], etc. This is why it’s crucial that numbering in the text and the order of emitted sources align. The frontend explicitly looks for bracketed numbers starting at 1 and increments upward sequentially.

* **Citations Modal and Metadata:** In the Citations modal (which opens when a reference is clicked), the UI will show the **document snippet** (the content of what was cited) and some metadata fields like the source URL (`metadata.source`) or any other standard fields (e.g., page number or relevance score if present). The example above would show the full sentence from AAA guidance or NHTSA tips, along with the URLs `https://www.aaa.com/safety-tips` or `https://www.nhtsa.gov/road-safety` if those were included. Any metadata keys that the UI doesn't specifically recognize for display will be stored but not shown to the user – this allows developers to attach internal info to citations without cluttering the UI.

## Persistence and Best Practices

When implementing citation events, there are a few additional considerations to ensure everything works smoothly:

* **Use Sequential Numeric Markers:** Always label citations in the text as `[1]`, `[2]`, etc., starting at 1. The Open WebUI renderer looks for increasing integers in square brackets to identify citations. Non-numeric or out-of-order labels will not be recognized as citations.

* **Event Type Compatibility:** Open WebUI now prefers the `"source"` event type name for adding citation data, but it still accepts `"citation"` for backward compatibility. In practice, you can use either.

* **Automatic vs Manual Save:** Open WebUI front end will normally save the final state of the assistant message (including sources and other events emitted to it) once the response is done. However, to be safe, you can manually save the sources in the pipe itself as shown in the example (using the `Chats.upsert_message...` call at the end of the pipe). This covers cases where the user exits the chat mid-stream and therefore the frontend never persists the final message.

```python
chat_id = __metadata__.get("chat_id") if __metadata__ else None
msg_id  = __metadata__.get("message_id") if __metadata__ else None
if chat_id and msg_id:
    Chats.upsert_message_to_chat_by_id_and_message_id(
        chat_id, msg_id, {"sources": emitted}
    )
```
