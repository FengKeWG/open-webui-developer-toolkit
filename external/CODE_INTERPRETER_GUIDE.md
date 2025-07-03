# code_interpreter.py

`backend/open_webui/utils/code_interpreter.py` runs the optional **code interpreter** feature. It sends Python snippets to a Jupyter kernel and streams the result back to the caller.

The module exposes a single public helper `execute_code_jupyter` which wraps the `JupyterCodeExecuter` context manager. The executer takes care of authenticating with the Jupyter server, creating a kernel and relaying messages over a WebSocket connection.

```python
async def execute_code_jupyter(base_url: str, code: str, token: str = "", password: str = "", timeout: int = 60) -> dict:
    async with JupyterCodeExecuter(base_url, code, token, password, timeout) as executor:
        result = await executor.run()
        return result.model_dump()
```

Internally `execute_in_jupyter` sends an `execute_request` and waits for output events. The loop stops once the kernel reports an idle state or the timeout is reached:

```python
while True:
    message = await asyncio.wait_for(ws.recv(), self.timeout)
    message_data = json.loads(message)
    if message_data.get("parent_header", {}).get("msg_id") != msg_id:
        continue
    msg_type = message_data.get("msg_type")
    match msg_type:
        case "stream":
            if message_data["content"]["name"] == "stdout":
                stdout += message_data["content"]["text"]
            elif message_data["content"]["name"] == "stderr":
                stderr += message_data["content"]["text"]
        case "execute_result" | "display_data":
            data = message_data["content"]["data"]
            if "image/png" in data:
                result.append(f"data:image/png;base64,{data['image/png']}")
            elif "text/plain" in data:
                result.append(data["text/plain"])
        case "error":
            stderr += "\n".join(message_data["content"]["traceback"])
        case "status":
            if message_data["content"]["execution_state"] == "idle":
                break
```

Each call returns a dictionary with `stdout`, `stderr` and `result` fields. Image data is encoded as a data URL so the frontend can display charts or figures without writing to disk.
