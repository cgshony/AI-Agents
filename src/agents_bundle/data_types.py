"""Core types: the content items that flow between the agent, the LLM and tools."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Literal, Union

from pydantic import BaseModel, Field


class Message(BaseModel):
    """A text message from the user or the assistant."""
    type: Literal["message"] = "message"
    role: Literal["system", "user", "assistant"]
    content: str


class ToolCall(BaseModel):
    """LLM's request to execute a tool."""
    type: Literal["tool_call"] = "tool_call"
    tool_call_id: str
    name: str
    arguments: dict


class ToolResult(BaseModel):
    """Result from tool execution."""
    type: Literal["tool_result"] = "tool_result"
    tool_call_id: str
    name: str
    status: Literal["success", "error"]
    content: list


ContentItem = Union[Message, ToolCall, ToolResult]


class Event(BaseModel):
    """A recorded occurrence during agent execution."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    execution_id: str
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    author: str  # "user" or agent name
    content: List[ContentItem] = Field(default_factory=list)
