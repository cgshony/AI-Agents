"""LLM communication layer: provider-agnostic requests/responses via LiteLLM."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Type, Union

from litellm import acompletion
from pydantic import BaseModel, Field

from agents_bundle.data_types import ContentItem, Message, ToolCall, ToolResult
from agents_bundle.tools.base_tool import BaseTool

DEFAULT_MODEL = "claude-haiku-5-5"


class LlmRequest(BaseModel):
    """Request object for LLM calls: what we select to send. Instructions, contents, tools"""
    model_config = {"arbitrary_types_allowed": True}

    instructions: List[str] = Field(default_factory=list)
    contents: List[ContentItem] = Field(default_factory=list)
    tools: List[BaseTool] = Field(default_factory=list)
    tool_choice: Optional[str] = None
    model_id: Optional[str] = None

    def append_instructions(self, text: str) -> None:
        """Append one instruction (system prompt) string, e.g. injected memories or a summary."""
        self.instructions.append(text)


class LlmResponse(BaseModel):
    """Response object from LLM calls: a standardized view of what we receive."""
    content: List[ContentItem] = Field(default_factory=list)
    error_message: Optional[str] = None
    usage_metadata: Dict[str, Any] = Field(default_factory=dict)


def build_messages(request: LlmRequest) -> List[dict]:
    """Convert LlmRequest to API message format."""
    messages = []

    for instruction in request.instructions:
        messages.append({"role": "system", "content": instruction})

    for item in request.contents:
        if isinstance(item, Message):
            messages.append({"role": item.role, "content": item.content})

        elif isinstance(item, ToolCall):
            tool_call_dict = {
                "id": item.tool_call_id,
                "type": "function",
                "function": {
                    "name": item.name,
                    "arguments": json.dumps(item.arguments),
                },
            }
            # Parallel tool calls belong to the same assistant message
            if messages and messages[-1]["role"] == "assistant":
                messages[-1].setdefault("tool_calls", []).append(tool_call_dict)
            else:
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [tool_call_dict],
                })

        elif isinstance(item, ToolResult):
            messages.append({
                "role": "tool",
                "tool_call_id": item.tool_call_id,
                "content": str(item.content[0]) if item.content else "",
            })

    return messages


class LlmClient:
    """Client for LLM API calls using LiteLLM (the provider adapter)."""

    def __init__(self, model: str = DEFAULT_MODEL, **config):
        self.model = model
        self.config = config

    async def generate(self, request: LlmRequest) -> LlmResponse:
        """Generate a response from the LLM."""
        try:
            messages = build_messages(request)
            tools = [t.tool_definition for t in request.tools] if request.tools else None

            response = await acompletion(
                model=self.model,
                messages=messages,
                tools=tools,
                **({"tool_choice": request.tool_choice} if request.tool_choice else {}),
                **self.config,
            )

            return self._parse_response(response)
        except Exception as e:
            return LlmResponse(error_message=str(e))

    async def ask(
        self,
        prompt: str,
        response_format: Optional[Type[BaseModel]] = None,
    ) -> Union[str, BaseModel]:
        """One-shot prompt with optional structured (Pydantic) output.

        With response_format, the JSON schema is put in the prompt and the reply is
        parsed into that Pydantic model. Used for memory extraction, duplicate
        checks and summarization, where the caller needs data rather than prose.
        """
        if response_format is not None:
            schema_text = json.dumps(response_format.model_json_schema())
            instruction = (
                f"{prompt}\n\nRespond ONLY with valid JSON matching this schema:\n"
                f"{schema_text}"
            )
        else:
            instruction = prompt

        request = LlmRequest(
            model_id=self.model,
            instructions=[instruction],
            contents=[Message(role="user", content="Please respond.")],
        )
        response = await self.generate(request)
        if response.error_message:
            raise RuntimeError(f"LLM request failed: {response.error_message}")

        text = next(
            (item.content for item in response.content if isinstance(item, Message)),
            "",
        )
        if response_format is None:
            return text

        # Models sometimes wrap JSON in ```json fences; strip them
        cleaned = text.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("```", 1)[0]
        return response_format.model_validate_json(cleaned.strip())

    def _parse_response(self, response) -> LlmResponse:
        """Convert API response to LlmResponse."""
        choice = response.choices[0]
        content_items = []

        if choice.message.content:
            content_items.append(Message(
                role="assistant",
                content=choice.message.content,
            ))

        if choice.message.tool_calls:
            for tc in choice.message.tool_calls:
                content_items.append(ToolCall(
                    tool_call_id=tc.id,
                    name=tc.function.name,
                    arguments=json.loads(tc.function.arguments),
                ))

        return LlmResponse(
            content=content_items,
            usage_metadata={
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
            },
        )
