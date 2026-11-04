"""Memory systems for the agent.

An LLM is stateless: it only "remembers" what is in the current request.
Memory is therefore a question of what to put into that request:

  - Session (short-term): the event history of one conversation, persisted
    across run() calls so the agent can continue where it left off.
  - Context optimization (working memory): shrinking the request when the
    history grows too large for the context window or the budget.
  - Task memory (long-term): lessons distilled from past runs, retrieved by
    semantic similarity and injected into new, related tasks.
"""

from agents_bundle.memory.session import BaseSessionManager, InMemorySessionManager, Session
from agents_bundle.memory.context_optimizer import (
    ContextOptimizer,
    apply_compaction,
    apply_sliding_window,
    apply_summarization,
    count_tokens,
    create_optimizer_callback,
)
from agents_bundle.memory.long_term import TaskMemory, TaskMemoryManager

__all__ = [
    "Session",
    "BaseSessionManager",
    "InMemorySessionManager",
    "ContextOptimizer",
    "create_optimizer_callback",
    "count_tokens",
    "apply_sliding_window",
    "apply_compaction",
    "apply_summarization",
    "TaskMemory",
    "TaskMemoryManager",
]
