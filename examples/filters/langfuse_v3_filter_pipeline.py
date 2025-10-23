"""
title: Langfuse Filter Pipeline for v3
author: open-webui
date: 2025-07-31
version: 0.2.0
license: MIT
description: A filter pipeline that uses Langfuse v3 SDK with manual span management.
requirements: langfuse>=3.0.0
"""

from typing import List, Optional, Dict
import os
import uuid

from utils.pipelines.main import get_last_assistant_message
from pydantic import BaseModel
from langfuse import Langfuse


def get_last_assistant_message_obj(messages: List[dict]) -> dict:
    """Retrieve the last assistant message from the message list."""
    for message in reversed(messages):
        if message["role"] == "assistant":
            return message
    return {}


class Pipeline:
    class Valves(BaseModel):
        pipelines: List[str] = []
        priority: int = 0
        secret_key: str
        public_key: str
        host: str
        insert_tags: bool = True
        use_model_name_instead_of_id_for_generation: bool = False
        debug: bool = False

    def __init__(self):
        self.type = "filter"
        self.name = "Langfuse Filter v3"

        self.valves = self.Valves(
            **{
                "pipelines": ["*"],
                "secret_key": os.getenv("LANGFUSE_SECRET_KEY", "your-secret-key-here"),
                "public_key": os.getenv("LANGFUSE_PUBLIC_KEY", "your-public-key-here"),
                "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
                "use_model_name_instead_of_id_for_generation": os.getenv("USE_MODEL_NAME", "false").lower() == "true",
                "debug": os.getenv("DEBUG_MODE", "false").lower() == "true",
            }
        )

        self.langfuse = None
        # Store root span objects (traces) - they should NOT be ended until chat is done
        self.chat_root_spans: Dict[str, any] = {}
        self.suppressed_logs = set()
        # Dictionary to store model names for each chat
        self.model_names = {}

    def log(self, message: str, suppress_repeats: bool = False):
        if self.valves.debug:
            if suppress_repeats:
                if message in self.suppressed_logs:
                    return
                self.suppressed_logs.add(message)
            print(f"[DEBUG] {message}")

    async def on_startup(self):
        self.log(f"on_startup triggered for {__name__}")
        self.set_langfuse()

    async def on_shutdown(self):
        self.log(f"on_shutdown triggered for {__name__}")
        if self.langfuse:
            try:
                # End all root spans (traces)
                for chat_id, root_span in list(self.chat_root_spans.items()):
                    try:
                        root_span.end()
                        self.log(f"Ended root span for chat_id: {chat_id}")
                    except Exception as e:
                        self.log(f"Failed to end root span for {chat_id}: {e}")
                
                self.chat_root_spans.clear()
                self.langfuse.flush()
                self.log("Langfuse data flushed on shutdown")
            except Exception as e:
                self.log(f"Failed to flush Langfuse data: {e}")

    async def on_valves_updated(self):
        self.log("Valves updated, resetting Langfuse client.")
        self.set_langfuse()

    def set_langfuse(self):
        try:
            self.log(f"Initializing Langfuse with host: {self.valves.host}")
            
            self.langfuse = Langfuse(
                secret_key=self.valves.secret_key,
                public_key=self.valves.public_key,
                host=self.valves.host,
                debug=self.valves.debug,
            )

            # Test authentication
            try:
                self.langfuse.auth_check()
                self.log(f"Langfuse client initialized and authenticated successfully")
            except Exception as e:
                self.log(f"Auth check failed: {e}")
                self.langfuse = None
                return

        except Exception as e:
            self.log(f"Langfuse initialization error: {e}")
            self.langfuse = None

    def _build_tags(self, task_name: str) -> list:
        """Build tags list based on valve settings."""
        tags_list = []
        if self.valves.insert_tags:
            tags_list.append("open-webui")
            if task_name not in ["user_response", "llm_response"]:
                tags_list.append(task_name)
        return tags_list

    async def inlet(self, body: dict, user: Optional[dict] = None) -> dict:
        self.log("Langfuse Filter INLET called")

        if not self.langfuse:
            self.log("[WARNING] Langfuse client not initialized - Skipped")
            return body

        metadata = body.get("metadata", {})
        chat_id = metadata.get("chat_id", str(uuid.uuid4()))

        # Handle temporary chats
        if chat_id == "local":
            session_id = metadata.get("session_id")
            chat_id = f"temporary-session-{session_id}"

        metadata["chat_id"] = chat_id
        body["metadata"] = metadata

        # Extract and store model information
        model_info = metadata.get("model", {})
        model_id = body.get("model")
        
        if chat_id not in self.model_names:
            self.model_names[chat_id] = {"id": model_id}
        else:
            self.model_names[chat_id]["id"] = model_id
            
        if isinstance(model_info, dict) and "name" in model_info:
            self.model_names[chat_id]["name"] = model_info["name"]
            self.log(f"Stored model info - name: '{model_info['name']}', id: '{model_id}'")

        required_keys = ["model", "messages"]
        missing_keys = [key for key in required_keys if key not in body]
        if missing_keys:
            error_message = f"Error: Missing keys in the request body: {', '.join(missing_keys)}"
            self.log(error_message)
            raise ValueError(error_message)

        user_email = user.get("email") if user else None
        task_name = metadata.get("task", "user_response")
        tags_list = self._build_tags(task_name)

        # Create root span (trace) ONCE per chat and keep it open
        if chat_id not in self.chat_root_spans:
            self.log(f"Creating new root span (trace) for chat_id: {chat_id}")

            try:
                # Create root span using start_span (manual - no context manager)
                # This span represents the entire chat and should NOT be ended until chat is done
                root_span = self.langfuse.start_span(
                    name=f"chat:{chat_id}",
                    input=body,
                )
                
                # Set trace-level attributes
                root_span.update_trace(
                    user_id=user_email,
                    session_id=chat_id,
                    tags=tags_list if tags_list else None,
                    metadata={
                        **metadata,
                        "interface": "open-webui",
                    },
                )
                
                self.chat_root_spans[chat_id] = root_span
                self.log(f"Successfully created root span: {root_span.id}")
            except Exception as e:
                self.log(f"Failed to create root span: {e}")
                return body
        else:
            # Update existing root span with new tags if needed
            root_span = self.chat_root_spans[chat_id]
            if tags_list:
                try:
                    root_span.update_trace(tags=tags_list)
                except Exception as e:
                    self.log(f"Failed to update root span tags: {e}")

        return body

    async def outlet(self, body: dict, user: Optional[dict] = None) -> dict:
        self.log("Langfuse Filter OUTLET called")

        if not self.langfuse:
            self.log("[WARNING] Langfuse client not initialized - Skipped")
            return body

        chat_id = body.get("chat_id")

        # Handle temporary chats
        if chat_id == "local":
            session_id = body.get("session_id")
            chat_id = f"temporary-session-{session_id}"

        if chat_id not in self.chat_root_spans:
            self.log(f"[WARNING] No matching root span found for chat_id: {chat_id}")
            return body

        root_span = self.chat_root_spans[chat_id]
        metadata = body.get("metadata", {})
        task_name = metadata.get("task", "llm_response")
        tags_list = self._build_tags(task_name)

        assistant_message = get_last_assistant_message(body["messages"])
        assistant_message_obj = get_last_assistant_message_obj(body["messages"])

        # Extract usage details in Langfuse v3 format
        usage_details = None
        if assistant_message_obj:
            info = assistant_message_obj.get("usage", {})
            if isinstance(info, dict):
                input_tokens = info.get("prompt_eval_count") or info.get("prompt_tokens")
                output_tokens = info.get("eval_count") or info.get("completion_tokens")
                if input_tokens is not None and output_tokens is not None:
                    usage_details = {
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                    }
                    self.log(f"Usage data extracted: {usage_details}")

        # Update root span (trace) output
        try:
            root_span.update_trace(output=assistant_message)
        except Exception as e:
            self.log(f"Failed to update root span output: {e}")

        # Create LLM generation as child of root span
        try:
            model_id = self.model_names.get(chat_id, {}).get("id", body.get("model"))
            model_name = self.model_names.get(chat_id, {}).get("name", "unknown")
            
            model_value = (
                model_name
                if self.valves.use_model_name_instead_of_id_for_generation
                else model_id
            )

            generation_metadata = {
                **metadata,
                "type": "llm_response",
                "interface": "open-webui",
                "model_id": model_id,
                "model_name": model_name,
            }
            
            # Create generation as child of root span
            # Use start_generation on root span, then end it immediately
            generation = root_span.start_generation(
                name=f"llm_response:{str(uuid.uuid4())}",
                model=model_value,
                input=body["messages"],
                output=assistant_message,
                metadata=generation_metadata,
            )
            
            # Update with usage details if available
            if usage_details:
                generation.update(usage_details=usage_details)
            
            # End the generation immediately
            generation.end()
            
            self.log(f"LLM generation created and ended for chat_id: {chat_id}")
        except Exception as e:
            self.log(f"Failed to create LLM generation: {e}")

        # Flush data
        try:
            self.langfuse.flush()
            self.log("Langfuse data flushed")
        except Exception as e:
            self.log(f"Failed to flush Langfuse data: {e}")

        return body
