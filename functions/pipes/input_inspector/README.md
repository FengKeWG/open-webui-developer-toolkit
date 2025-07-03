# Input Inspector
A simple demonstration pipe that shows the contents of each pipe input argument.
It emits a citation block for `body`, `__metadata__`, `__user__`, `__request__`,
`__files__` and `__tools__` so you can inspect the data WebUI passes to a
pipeline.

Sensitive headers like `Authorization` and `Cookie` are redacted from
`__request__` to avoid leaking private information. The `REDACT_REQUEST`
valve controls this behavior and defaults to `True`.

Values that can't be serialized to JSON are replaced with a placeholder
like `<UNSERIALIZABLE ClassName>` so the overall structure of the input
remains visible.

## Usage
1. Copy `input_inspector.py` to your WebUI under **Admin ▸ Pipelines**.
2. Enable the pipe and run a chat. The pipe will emit citation blocks containing
the JSON for each argument.
