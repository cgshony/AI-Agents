"""Memory tool: automatically injects relevant past experiences into the LLM request."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agents_bundle.data_types import Message
from agents_bundle.execution_context import ExecutionContext
from agents_bundle.tools.base_tool import BaseTool

if TYPE_CHECKING:
    from agents_bundle.llm_client import LlmRequest
    from agents_bundle.memory.long_term import TaskMemory


class MemoryTool(BaseTool):
    """Searches task memories and appends them to the system instructions.

    tool_definition=None means the LLM never sees or calls this tool;
    it works only through the process_llm_request hook.
    """

    def __init__(self):
        super().__init__(
            name="recall_memory",
            description=(
                "Search for past problem-solving records. "
                "Use this to check if similar problems were solved before."
            ),
            tool_definition=None,
        )

    async def execute(self, context: ExecutionContext, query: str = "") -> str:
        """Search memories and return them formatted."""
        if context.memory_manager is None:
            return ""
        memories = await context.memory_manager.search(query, top_k=3)
        return self._format_memories(memories) if memories else ""

    def _format_memories(self, memories: list["TaskMemory"]) -> str:
        results = []
        for i, mem in enumerate(memories, 1):
            status = "Correct" if mem.is_correct else "Incorrect"
            text = (
                f"[Record {i}]\n"
                f"- Problem: {mem.task_summary}\n"
                f"- Approach: {mem.approach}\n"
                f"- Answer: {mem.final_answer}\n"
                f"- Result: {status}"
            )
            if not mem.is_correct and mem.error_analysis:
                text += f"\n- Error analysis: {mem.error_analysis}"
            results.append(text)
        return "\n\n".join(results)

    async def process_llm_request(
        self,
        context: ExecutionContext,
        request: "LlmRequest",
    ) -> None:
        """Before each LLM call: search with the latest user message, inject matches."""
        if context.memory_manager is None:
            return

        user_msgs = [
            c for c in request.contents
            if isinstance(c, Message) and c.role == "user"
        ]
        if not user_msgs:
            return

        result = await self.execute(context, user_msgs[-1].content)
        if not result:
            return

        request.append_instructions(
            "The following are records from similar problems solved in the past:\n"
            "<PAST_EXPERIENCES>\n"
            f"{result}\n"
            "</PAST_EXPERIENCES>\n"
            "Reference successful approaches and avoid approaches that led to failures."
        )
