"""
title: Citations Example
id: citations_example
author: OpenAI Codex
description: Demonstrates realistic streaming example where inline citations are emitted incrementally as the response is streamed.
version: 1.0.0
license: MIT
"""
from __future__ import annotations

import asyncio
import datetime
from typing import Any, AsyncGenerator, Awaitable, Callable
# Import Chats model to allow manual saving of messages to the database
from open_webui.models.chats import Chats

class Pipe:
    async def pipe(
        self,
        body: dict[str, Any],
        __event_emitter__: Callable[[dict[str, Any]], Awaitable[None]] | None,
        __metadata__: dict[str, Any] | None = None,
        *_,
    ) -> AsyncGenerator[any, None]:
        
        # Example User Message:
        #   I'm considering driving from Los Angeles to New York City.
        #   Could you calculate how long the trip would take if I drive at an average speed of 60 mph?
        #   Also, can you share some reliable information about the recommended number of breaks during such long trips and safety tips?


        # Assistant response text with three citations clearly marked [1], [2], [3]
        response_text = (
            "Certainly! At an average speed of 60 mph, driving the approximately 2,790 miles from Los Angeles to New York City would take roughly 46.5 hours [1]. "
            "The American Automobile Association (AAA) recommends taking a break every two hours of continuous driving to prevent fatigue. Additionally, AAA advises stopping at least every 100 miles or two hours to stretch your legs and remain alert [2]. "
            "Further safety tips provided by the National Highway Traffic Safety Administration (NHTSA) include getting sufficient rest before long trips, avoiding driving during peak fatigue times (midnight to 6 a.m.), and planning your route carefully in advance [3]."
        )
        
        # Define citation sources
        sources = {
            # [1]: Most complex (function_call + function_call_output clearly separate)
            "[1]": {
                "source": {"name": f"Tool Call(s)"},
                "document": [
                    "Calculator tool executed to evaluate '2790 miles / 60 mph'.",
                    "Calculator tool returned result: '46.5 hours'."
                ],
                "metadata": [
                    {   # Custom Metadata for original function call in OpenAI Responses API format
                        "tool_name": "calculator",
                        "execution_time": "0.2s",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "payload": {
                            "type": "function_call",
                            "id": "fc_123456789abcdef",
                            "call_id": "call_calc_001",
                            "name": "calculator",
                            "arguments": "{\"expression\":\"2790/60\"}",
                            "status": "completed"
                        },
                        "internal_notes": "Evaluated user-provided distance/speed equation"
                    },
                    {   # Custom Metadata for corresponding function call output in OpenAI Responses API format
                        "tool_name": "calculator",
                        "execution_time": "0.2s",
                        "timestamp": datetime.datetime.now().isoformat(),
                        "payload": {
                            "type": "function_call_output",
                            "call_id": "call_calc_001",
                            "output": "2790 miles / 60 mph = 46.5 hours"
                        },
                    }
                ],
            },

            # [2]: Intermediate Complexity (multiple documents under one shared source URL/name)
            "[2]": {
                "source": {"name": "American Automobile Association (AAA)"},
                "document": [
                    "AAA recommends taking a break every two hours of continuous driving.",
                    "AAA advises stopping at least every 100 miles to stretch and remain alert."
                ],
                "metadata": [
                    {
                        "source": "https://www.aaa.com/safety-tips",
                        "date_accessed": datetime.date.today().isoformat()
                    },
                    {
                        "source": "https://www.aaa.com/safety-tips",
                        "date_accessed": datetime.date.today().isoformat()
                    }
                ],
            },

            # [3]: Basic citation example (single document, simple metadata)
            "[3]": {
                "source": {"name": "National Highway Traffic Safety Administration (NHTSA)"},
                "document": ["Always ensure you get sufficient rest before long drives."],
                "metadata": [
                    {
                        "source": "https://www.nhtsa.gov/road-safety",
                        "date_accessed": datetime.date.today().isoformat()
                    }
                ],
            },
        }

        # Stream the response word-by-word, emitting citations as encountered
        emitted_citations: list[dict[str, Any]] = []  # Track emitted citations for later DB save
        for word in response_text.split():
            await asyncio.sleep(0.03)  # simulate realistic streaming delay
            yield word + " "

            # Clean word of punctuation to detect citation placeholders
            if word.rstrip(".,!?") in sources:
                if __event_emitter__:
                    citation = sources[word.rstrip(".,!?")]
                    emitted_citations.append(citation)
                    await __event_emitter__({"type": "citation", "data": citation})
        # End of response streaming

        """
        # Alternatively, you can emit the entire response with citations at once at the end using chat:completion.
        if __event_emitter__:
            await __event_emitter__(
                {
                    "type": "chat:completion",
                    "data": {
                        "content": "",  # Leave as empty string to prevent overwriting previously streamed content. ALWAYS include this field when using chat:completion or the UI freezes if user closes page mid-stream.
                        "sources": [
                            {
                                "source": {"name": "Harvard Health"},
                                "document": ["Mediterranean diet linked to improved cardiovascular outcomes."],
                                "metadata": [{"source": "https://health.harvard.edu", "date_accessed": datetime.date.today().isoformat()}],
                            },
                            {
                                "source": {"name": "Mayo Clinic"},
                                "document": ["Mediterranean diet reduces inflammation markers according to recent studies."],
                                "metadata": [{"source": "https://mayoclinic.org", "date_accessed": datetime.date.today().isoformat()}],
                            },
                        ],
                    },
                }
            )
        """

        # Bonus Best practice:
        # Save the sources (citations) directly to the database at the bottom of the pipe.
        # Normally, the browser saves this automatically after streaming completes.
        # However, if the user closes the window or navigates away before streaming finishes,
        # this ensures the citations remain saved and available when the user returns later.
        # If the user stays, the browser’s final update will overwrite this temporary save
        chat_id = __metadata__.get("chat_id") if __metadata__ else None
        message_id = __metadata__.get("message_id") if __metadata__ else None
        if chat_id and message_id:
            Chats.upsert_message_to_chat_by_id_and_message_id(
                chat_id,
                message_id,
                {
                    "sources": emitted_citations,
                },
            )