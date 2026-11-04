"""Context optimization strategies: sliding window, compaction, summarization.

All strategies run as before_llm_callbacks: they edit the LlmRequest that is
about to be sent, never context.events, so the full history is still recorded.
"""

from __future__ import annotations

import inspect
import json
from typing import TYPE_CHECKING, Dict, List, Optional

from agents_bundle.data_types import ContentItem, Message, ToolCall, ToolResult
from agents_bundle.execution_context import ExecutionContext

if TYPE_CHECKING:
    from agents_bundle.llm_client import LlmClient, LlmRequest, LlmResponse


def create_optimizer_callback(apply_optimization, threshold: int = 50000):
    """Wrap a strategy in a before_llm_callback that fires above `threshold` tokens."""

    async def callback(
        context: ExecutionContext,
        request: "LlmRequest",
    ) -> Optional["LlmResponse"]:
        if count_tokens(request) < threshold:
            return None

        result = apply_optimization(context, request)
        if inspect.isawaitable(result):
            await result
        return None  # None = "still call the LLM", with the modified request

    return callback


def count_tokens(request: "LlmRequest") -> int:
    """Estimate the token count of an LlmRequest.

    tiktoken is OpenAI's tokenizer; Claude's differs, so this is an approximation.
    Good enough to decide *when* to optimize.
    """
    import tiktoken

    from agents_bundle.llm_client import build_messages

    encoding = tiktoken.get_encoding("o200k_base")
    total_tokens = 0

    for message in build_messages(request):
        total_tokens += 4  # per-message overhead
        if message.get("content"):
            total_tokens += len(encoding.encode(str(message["content"])))
        for tool_call in message.get("tool_calls") or []:
            func = tool_call.get("function", {})
            if func.get("name"):
                total_tokens += len(encoding.encode(func["name"]))
            if func.get("arguments"):
                total_tokens += len(encoding.encode(func["arguments"]))

    for tool in request.tools or []:
        if tool.tool_definition:
            total_tokens += len(encoding.encode(json.dumps(tool.tool_definition)))

    return total_tokens


def _first_user_index(contents: List[ContentItem]) -> int | None:
    for i, item in enumerate(contents):
        if isinstance(item, Message) and item.role == "user":
            return i
    return None


def apply_sliding_window(
    context: ExecutionContext,
    request: "LlmRequest",
    window_size: int = 20,
) -> None:
    """Keep everything up to the first user message, plus the last N items."""
    contents = request.contents
    user_idx = _first_user_index(contents)
    if user_idx is None:
        return

    preserved = contents[: user_idx + 1]
    remaining = contents[user_idx + 1 :]
    if len(remaining) > window_size:
        remaining = remaining[-window_size:]
        # Don't start the window on an orphan ToolResult: its ToolCall was cut,
        # and the API rejects a tool result without its matching call
        while remaining and isinstance(remaining[0], ToolResult):
            remaining = remaining[1:]

    request.contents = preserved + remaining


# Tools whose ToolCall arguments get compressed
TOOLCALL_COMPACTION_RULES = {
    "create_file": "[Content saved to file]",
}

# Tools whose ToolResult content gets replaced by a reference
TOOLRESULT_COMPACTION_RULES = {
    "read_file": "File content from {file_path}. Re-read if needed.",
    "search_web": "Search results processed. Query: {query}. Re-search if needed.",
    "tavily_search": "Search results processed. Query: {query}. Re-search if needed.",
}


def apply_compaction(context: ExecutionContext, request: "LlmRequest") -> None:
    """Replace bulky, re-fetchable tool data with short reference messages."""
    tool_call_args: Dict[str, Dict] = {}
    compacted = []

    for item in request.contents:
        if isinstance(item, ToolCall):
            tool_call_args[item.tool_call_id] = item.arguments
            if item.name in TOOLCALL_COMPACTION_RULES:
                compressed_args = {
                    k: TOOLCALL_COMPACTION_RULES[item.name] if k == "content" else v
                    for k, v in item.arguments.items()
                }
                compacted.append(item.model_copy(update={"arguments": compressed_args}))
            else:
                compacted.append(item)

        elif isinstance(item, ToolResult) and item.name in TOOLRESULT_COMPACTION_RULES:
            args = tool_call_args.get(item.tool_call_id, {})
            compressed = TOOLRESULT_COMPACTION_RULES[item.name].format(
                file_path=args.get("file_path", args.get("path", "unknown")),
                query=args.get("query", "unknown"),
            )
            compacted.append(item.model_copy(update={"content": [compressed]}))

        else:
            compacted.append(item)

    request.contents = compacted


SUMMARIZATION_PROMPT = """You are summarizing an AI agent's work progress.

Given the following execution history, extract:
1. Key findings: Important information discovered
2. Tools used: List of tools that were called
3. Current status: What has been accomplished and what remains

Be concise. Focus on information that will help the agent continue its work.

Execution History:
{history}

Provide a structured summary."""


async def apply_summarization(
    context: ExecutionContext,
    request: "LlmRequest",
    llm_client: "LlmClient",
    keep_recent: int = 5,
) -> None:
    """Replace the middle of the history with an LLM-written summary."""
    contents = request.contents
    user_idx = _first_user_index(contents)
    if user_idx is None:
        return

    # Where the previous summary stopped (in request.contents index space)
    last_summary_idx = context.state.get("last_summary_idx", user_idx)

    summary_start = last_summary_idx + 1
    summary_end = len(contents) - keep_recent
    # Don't split a ToolCall from its ToolResult at the boundary
    while summary_end < len(contents) and isinstance(contents[summary_end], ToolResult):
        summary_end += 1
    if summary_end <= summary_start:
        return

    preserved_start = contents[:summary_start]
    preserved_end = contents[summary_end:]
    to_summarize = contents[summary_start:summary_end]

    summary = await generate_summary(llm_client, format_history_for_summary(to_summarize))

    request.append_instructions(f"[Previous work summary]\n{summary}")
    request.contents = preserved_start + preserved_end
    context.state["last_summary_idx"] = len(preserved_start) - 1


def format_history_for_summary(items: List[ContentItem]) -> str:
    """Convert content items to text for summarization."""
    lines = []
    for item in items:
        if isinstance(item, Message):
            lines.append(f"[{item.role}]: {item.content[:500]}...")
        elif isinstance(item, ToolCall):
            lines.append(f"[Tool Call]: {item.name}({item.arguments})")
        elif isinstance(item, ToolResult):
            preview = str(item.content[0])[:200] if item.content else ""
            lines.append(f"[Tool Result]: {item.name} -> {preview}...")
    return "\n".join(lines)


async def generate_summary(llm_client: "LlmClient", history: str) -> str:
    """Ask the LLM for a progress summary."""
    return await llm_client.ask(SUMMARIZATION_PROMPT.format(history=history))


class ContextOptimizer:
    """Hierarchical strategy: cheap compaction first, LLM summarization only if still too big."""

    def __init__(
        self,
        llm_client: "LlmClient",
        token_threshold: int = 50000,
        enable_compaction: bool = True,
        enable_summarization: bool = True,
        keep_recent: int = 5,
    ):
        self.llm_client = llm_client
        self.token_threshold = token_threshold
        self.enable_compaction = enable_compaction
        self.enable_summarization = enable_summarization
        self.keep_recent = keep_recent

    async def __call__(
        self,
        context: ExecutionContext,
        request: "LlmRequest",
    ) -> Optional["LlmResponse"]:
        """Register an instance directly as a before_llm_callback."""
        if count_tokens(request) < self.token_threshold:
            return None

        if self.enable_compaction:
            apply_compaction(context, request)
            if count_tokens(request) < self.token_threshold:
                return None

        if self.enable_summarization:
            await apply_summarization(context, request, self.llm_client, self.keep_recent)

        return None
