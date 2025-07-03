# task.py

`backend/open_webui/utils/task.py` centralizes helpers that build prompts for background tasks such as title generation or retrieval queries. It also contains utilities for selecting the model used for these asynchronous steps.

## Task model selection

`get_task_model_id` chooses which model should handle a task. The default model id can be overridden with user settings depending on whether the current model is an Ollama model:
```python
def get_task_model_id(
    default_model_id: str, task_model: str, task_model_external: str, models
) -> str:
    # Set the task model
    task_model_id = default_model_id
    # Check if the user has a custom task model and use that model
    if models[task_model_id].get("owned_by") == "ollama":
        if task_model and task_model in models:
            task_model_id = task_model
    else:
        if task_model_external and task_model_external in models:
            task_model_id = task_model_external

    return task_model_id
```

## Prompt templating (deep dive)

Most helpers rely on `prompt_template` which expands date and user placeholders. The function inserts the current date/time and substitutes unknown values with "Unknown":
```python
def prompt_template(
    template: str, user_name: Optional[str] = None, user_location: Optional[str] = None
) -> str:
    # Get the current date
    current_date = datetime.now()

    # Format the date to YYYY-MM-DD
    formatted_date = current_date.strftime("%Y-%m-%d")
    formatted_time = current_date.strftime("%I:%M:%S %p")
    formatted_weekday = current_date.strftime("%A")

    template = template.replace("{{CURRENT_DATE}}", formatted_date)
    template = template.replace("{{CURRENT_TIME}}", formatted_time)
    template = template.replace(
        "{{CURRENT_DATETIME}}", f"{formatted_date} {formatted_time}"
    )
    template = template.replace("{{CURRENT_WEEKDAY}}", formatted_weekday)

    if user_name:
        # Replace {{USER_NAME}} in the template with the user's name
        template = template.replace("{{USER_NAME}}", user_name)
    else:
        # Replace {{USER_NAME}} in the template with "Unknown"
        template = template.replace("{{USER_NAME}}", "Unknown")

    if user_location:
        # Replace {{USER_LOCATION}} in the template with the current location
        template = template.replace("{{USER_LOCATION}}", user_location)
    else:
        # Replace {{USER_LOCATION}} in the template with "Unknown"
        template = template.replace("{{USER_LOCATION}}", "Unknown")

    return template
```
Supported placeholders include `{{CURRENT_DATE}}`, `{{CURRENT_TIME}}`, `{{CURRENT_DATETIME}}`, `{{CURRENT_WEEKDAY}}`, `{{USER_NAME}}` and `{{USER_LOCATION}}`. Additional variables can be replaced with `prompt_variables_template` using a dictionary of substitutions.

### Prompt truncation helpers

`replace_prompt_variable` lets a template reference a user prompt in different ways. The regex looks for `{{prompt}}` plus variants to grab the start, end or a middle-truncated slice:
```python
def replace_prompt_variable(template: str, prompt: str) -> str:
    def replacement_function(match):
        full_match = match.group(
            0
        ).lower()  # Normalize to lowercase for consistent handling
        start_length = match.group(1)
        end_length = match.group(2)
        middle_length = match.group(3)

        if full_match == "{{prompt}}":
            return prompt
        elif start_length is not None:
            return prompt[: int(start_length)]
        elif end_length is not None:
            return prompt[-int(end_length) :]
        elif middle_length is not None:
            middle_length = int(middle_length)
            if len(prompt) <= middle_length:
                return prompt
            start = prompt[: math.ceil(middle_length / 2)]
            end = prompt[-math.floor(middle_length / 2) :]
            return f"{start}...{end}"
        return ""

    # Updated regex pattern to make it case-insensitive with the `(?i)` flag
    pattern = r"(?i){{prompt}}|{{prompt:start:(\d+)}}|{{prompt:end:(\d+)}}|{{prompt:middletruncate:(\d+)}}"
    template = re.sub(pattern, replacement_function, template)
    return template
```
`replace_messages_variable` mirrors this for the conversation history with `{{MESSAGES}}` placeholders.

### Example usage
```python
from open_webui.utils import task

raw = "Hello {{USER_NAME}}, today is {{CURRENT_DATE}}"
print(task.prompt_template(raw, user_name="Alice"))
# -> "Hello Alice, today is 2024-05-30" (date will vary)

snippet = "{{prompt:start:5}} ... {{prompt:end:5}}"
print(task.replace_prompt_variable(snippet, "This is a long user message"))
# -> "This  ... sage"
```

## Generation helpers

A series of small helpers combine these utilities to craft prompts for specific subtasks:

- `rag_template` inserts retrieved context and the original query while guarding against prompt injection.
- `title_generation_template`, `tags_generation_template` and `image_prompt_generation_template` build prompts from recent messages.
- `emoji_generation_template` and `autocomplete_generation_template` handle lighter features like emoji responses or command completion.
- `moa_response_generation_template` merges results from multiple models.
- `tools_function_calling_generation_template` injects OpenAI-style tool specs.

These helpers typically call `replace_prompt_variable`, `replace_messages_variable` and `prompt_template` in sequence. An example is `title_generation_template`:
```python
def title_generation_template(
    template: str, messages: list[dict], user: Optional[dict] = None
) -> str:
    prompt = get_last_user_message(messages)
    template = replace_prompt_variable(template, prompt)
    template = replace_messages_variable(template, messages)

    template = prompt_template(
        template,
        **(
            {"user_name": user.get("name"), "user_location": user.get("location")}
            if user
            else {}
        ),
    )

    return template
```
Together these utilities standardize how background tasks are prompted across the codebase.

## `rag_template` in depth

`rag_template` crafts the prompt used when performing Retrieval Augmented
Generation (RAG).  If the provided template is empty the helper falls back to
`DEFAULT_RAG_TEMPLATE` from `open_webui.config`.  The function expands any date
placeholders via `prompt_template` and performs several safety checks:

- Logs a warning when the template does not include `[context]` or
  `{{CONTEXT}}` – these markers are required so the retrieved documents are
  injected into the prompt.
- Warns if the supplied `context` already contains `<context>` and `</context>`
  tags which might indicate a prompt injection attempt.

To avoid conflicts when the context itself contains `[query]` or `{{QUERY}}`,
the original placeholder in the template is temporarily replaced with a unique
token using `uuid.uuid4()`.  Once the context and query text have been inserted
the token is swapped back to the actual query string.

Example usage:

```python
from open_webui.utils import task

template = "[context]\n\nQ: [query]"
context = "Info about [query] <context> tags</context>"
query = "How does it work?"

print(task.rag_template(template, context, query))
```

Which prints something similar to:

```
Info about How does it work? <context> tags</context>

Q: How does it work?
```

Even when the context already contains `[query]` the placeholder in the template
is expanded correctly without losing the original context text.
