"""Tool abstraction: a unified interface for every tool the agent can call.

A tool has two faces:
  - tool_definition: the JSON schema the LLM sees, so it knows the tool
    exists and how to call it (None hides the tool from the LLM).
  - execute(): the Python code that runs when the LLM calls it.

Tools can also:
  - require human confirmation before running (requires_confirmation), for
    actions that are destructive or irreversible;
  - modify the LLM request before each call (process_llm_request), which
    lets a tool contribute context without the LLM explicitly calling it.
"""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Callable, Dict

from agents_bundle.execution_context import ExecutionContext
from agents_bundle.tools.tool_schema import format_tool_definition, function_to_input_schema

if TYPE_CHECKING:
    from agents_bundle.llm_client import LlmRequest


class BaseTool(ABC):
    """Abstract base class for all tools."""

    DEFAULT_CONFIRMATION_TEMPLATE = (
        "The agent wants to execute '{name}' with arguments: {arguments}. "
        "Do you approve?"
    )

    def __init__(
        self,
        name: str = None,
        description: str = None,
        tool_definition: Dict[str, Any] = None,
        requires_confirmation: bool = False,
        confirmation_message_template: str | None = None,
    ):
        self.name = name or self.__class__.__name__
        self.description = description or self.__doc__ or ""
        self._tool_definition = tool_definition
        self.requires_confirmation = requires_confirmation
        self.confirmation_message_template = (
            confirmation_message_template or self.DEFAULT_CONFIRMATION_TEMPLATE
        )

    @property
    def tool_definition(self) -> Dict[str, Any] | None:
        """JSON schema sent to the LLM. None means the LLM never sees this tool."""
        return self._tool_definition

    def get_confirmation_message(self, arguments: dict) -> str:
        """Message shown to the human when this tool call needs approval."""
        return self.confirmation_message_template.format(name=self.name, arguments=arguments)

    async def process_llm_request(
        self,
        context: ExecutionContext,
        request: "LlmRequest",
    ) -> None:
        """Hook: modify the LlmRequest before it is sent. Default does nothing."""
        return None

    @abstractmethod
    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        pass

    async def __call__(self, context: ExecutionContext, **kwargs) -> Any:
        return await self.execute(context, **kwargs)


class FunctionTool(BaseTool):
    """Wraps a Python function (sync or async) as a BaseTool."""

    def __init__(
        self,
        func: Callable,
        name: str = None,
        description: str = None,
        tool_definition: Dict[str, Any] = None,
        requires_confirmation: bool = False,
        confirmation_message_template: str | None = None,
    ):
        self.func = func
        self.needs_context = "context" in inspect.signature(func).parameters

        # Must set name/description before _generate_definition uses them
        super().__init__(
            name=name or func.__name__,
            description=description or (func.__doc__ or "").strip(),
            tool_definition=tool_definition,
            requires_confirmation=requires_confirmation,
            confirmation_message_template=confirmation_message_template,
        )

        if self._tool_definition is None:
            self._tool_definition = self._generate_definition()

    async def execute(self, context: ExecutionContext, **kwargs) -> Any:
        """Execute the wrapped function."""
        if self.needs_context:
            result = self.func(context=context, **kwargs)
        else:
            result = self.func(**kwargs)

        # Handle both sync and async functions
        if inspect.iscoroutine(result):
            return await result
        return result

    def _generate_definition(self) -> Dict[str, Any]:
        """Generate tool definition from function signature."""
        parameters = function_to_input_schema(self.func)
        return format_tool_definition(self.name, self.description, parameters)


def tool(
    func: Callable = None,
    *,
    name: str = None,
    description: str = None,
    requires_confirmation: bool = False,
    confirmation_message: str | None = None,
):
    """Decorator to create a FunctionTool from a function.

    Can be used with or without arguments:
        @tool
        def my_func(...): ...

        @tool(requires_confirmation=True,
              confirmation_message="Delete {arguments}?")
        def delete_file(...): ...
    """
    def decorator(f: Callable) -> FunctionTool:
        return FunctionTool(
            f,
            name=name,
            description=description,
            requires_confirmation=requires_confirmation,
            confirmation_message_template=confirmation_message,
        )

    if func is not None:
        return decorator(func)
    return decorator
