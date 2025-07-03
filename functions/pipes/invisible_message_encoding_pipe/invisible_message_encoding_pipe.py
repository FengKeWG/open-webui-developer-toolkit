"""
title: Invisible Message Encoder
id: invisible_message_encoder
description: Embed a secret message in markdown comments, keeping it invisible from the UI.
author: Justin Kropp
version: 2.5.1
license: MIT

Notes:
    To encode, simply place your secret between brackets followed by ': #'
    (e.g., `[your secret message]: #`). Previous formats used complex encodings, but this
    approach is straightforward and intuitive.
"""

import re
from typing import Any, AsyncGenerator, Awaitable, Callable

# ——— Helper Functions ————————————————————————

HIDDEN_MESSAGE_REGEX = re.compile(r"\[([^\]]+)\]: #")


def hide_message(secret: str) -> str:
    """Wraps the secret message in a markdown comment."""
    return f"\n[{secret}]: #\n"


def reveal_message(markdown: str) -> str | None:
    """Extracts the first hidden message from markdown, if present."""
    match = HIDDEN_MESSAGE_REGEX.search(markdown)
    return match.group(1) if match else None


def find_latest_hidden_message(messages: list[dict[str, Any]]) -> str | None:
    """Finds the most recent hidden message from message history."""
    for message in reversed(messages):
        content = message.get("content", "")
        if hidden_message := reveal_message(content):
            return hidden_message
    return None


# ——— Main Pipe Logic ——————————————————————————


class Pipe:
    async def pipe(
        self,
        body: dict[str, Any],
        __metadata__: dict[str, Any],
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]] | None,
        __event_call__: Callable[[dict[str, Any]], Awaitable[Any]] | None,
        *_,
    ) -> AsyncGenerator[str, None]:

        # Step 1: Check if there's a hidden message to reveal
        previous_messages = body.get("messages", [])
        hidden_message = find_latest_hidden_message(previous_messages)

        if hidden_message:
            yield f"🔓 **Your hidden message:** `{hidden_message}`"
            return

        # Step 2: Prompt user to input a new secret message
        user_input = await __event_call__(
            {
                "type": "input",
                "data": {
                    "title": "Encode a Hidden Message",
                    "message": "Enter the message you'd like to hide:",
                    "placeholder": "Your secret here…",
                },
            }
        )

        if not user_input:
            yield "⚠️ **No message provided.** Please try again."
            return

        # Step 3: Embed the user's secret message invisibly
        hidden_markdown_comment = hide_message(user_input)
        yield (
            "✨ **Message encoded successfully!** It is now hidden in this response. "
            "To decode it, send another message.\n"
            f"{hidden_markdown_comment}"
        )
