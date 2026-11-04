"""Execution context: the agent's central storage, and the result of a run.

The ExecutionContext holds everything a run accumulates (events, step count,
free-form state) plus handles to the session and memory managers, so that
tools and callbacks can reach them.

A run can end in two ways: complete, or paused because a tool needs human
approval. In the paused case AgentResult lists the PendingToolCalls; the
caller answers with ToolConfirmations and calls run() again to resume.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from agents_bundle.data_types import Event, ToolCall


@dataclass
class ExecutionContext:
    """Central storage for all execution state."""

    execution_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[Event] = field(default_factory=list)
    current_step: int = 0
    state: Dict[str, Any] = field(default_factory=dict)
    final_result: Optional[str | BaseModel] = None
    # Persistence across runs (session) and learning across tasks (memory)
    session: Optional[Any] = None
    session_manager: Optional[Any] = None
    memory_manager: Optional[Any] = None  # read by MemoryTool

    def add_event(self, event: Event):
        """Append an event to the execution history."""
        self.events.append(event)

    def increment_step(self):
        """Move to the next execution step."""
        self.current_step += 1


@dataclass
class AgentResult:
    """Result of an agent execution."""
    output: Any  # str | BaseModel
    context: ExecutionContext
    status: str = "complete"  # "complete" | "pending_confirmation" | "error"
    pending_tool_calls: list = field(default_factory=list)


class PendingToolCall(BaseModel):
    """A tool call awaiting user confirmation."""
    tool_call: ToolCall
    confirmation_message: str


class ToolConfirmation(BaseModel):
    """User's response to a pending tool call."""
    tool_call_id: str
    approved: bool
    modified_arguments: dict | None = None
    reason: str | None = None
