"""Core Agent class: the ReAct (think-act) loop with structured output and tool callbacks."""

from __future__ import annotations

import logging
from typing import Any, Callable, List, Optional, Type

from pydantic import BaseModel

from agents_bundle.data_types import Event, Message, ToolCall, ToolResult
from agents_bundle.execution_context import AgentResult, ExecutionContext
from agents_bundle.llm_client import LlmClient, LlmRequest, LlmResponse
from agents_bundle.tools.base_tool import BaseTool, FunctionTool
from agents_bundle.tools.tool_schema import format_tool_definition

logger = logging.getLogger(__name__)


class Agent:
    """Tool-calling agent with a ReAct loop."""

    def __init__(
        self,
        model: LlmClient,
        tools: List[BaseTool] | None = None,
        instructions: str = "",
        max_steps: int = 10,
        name: str = "agent",
        description: str = "",
        output_type: Optional[Type[BaseModel]] = None,
        before_tool_callbacks: list[Callable] | None = None,
        after_tool_callbacks: list[Callable] | None = None,
    ):
        self.model = model
        self.instructions = instructions
        self.max_steps = max_steps
        self.name = name
        self.description = description
        self.output_type = output_type
        self.output_tool_name: str | None = None
        self.before_tool_callbacks = before_tool_callbacks or []
        self.after_tool_callbacks = after_tool_callbacks or []
        self.tools = self._setup_tools(tools or [])

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #

    async def run(
        self,
        user_input: str | None = None,
        context: ExecutionContext | None = None,
        verbose: bool = False,
    ) -> AgentResult:
        """Execute the agent until it produces a final result or hits max_steps.

        user_input may be None when continuing an existing context.
        """
        if context is None:
            context = ExecutionContext()

        if user_input:
            user_event = Event(
                execution_id=context.execution_id,
                author="user",
                content=[Message(role="user", content=user_input)],
            )
            context.add_event(user_event)

        # Execute steps until completion or max steps reached
        while not context.final_result and context.current_step < self.max_steps:
            await self.step(context, verbose=verbose)

            # Check if the last event is a final response
            if context.events:
                last_event = context.events[-1]
                if self._is_final_response(last_event):
                    context.final_result = self._extract_final_result(last_event)

        return AgentResult(output=context.final_result, context=context)

    async def step(self, context: ExecutionContext, verbose: bool = False) -> None:
        """Perform one think-act cycle."""
        # Prepare what to send to the LLM
        llm_request = self._prepare_llm_request(context)

        # Get LLM's decision
        llm_response = await self.think(llm_request)
        if llm_response.error_message:
            raise RuntimeError(f"LLM request failed: {llm_response.error_message}")

        if verbose:
            self._log_response(llm_response)

        # Record LLM response as an event
        response_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            content=llm_response.content,
        )
        context.add_event(response_event)

        # Execute tools if the LLM requested any (act records its own event)
        tool_calls = [c for c in llm_response.content if isinstance(c, ToolCall)]
        if tool_calls:
            await self.act(context, tool_calls)

        context.increment_step()

    async def think(self, llm_request: LlmRequest) -> LlmResponse:
        """Call the LLM to decide the next action."""
        return await self.model.generate(llm_request)

    async def act(
        self,
        context: ExecutionContext,
        tool_calls: List[ToolCall],
    ) -> None:
        """Execute the tools requested by the LLM, with callbacks, and record the results."""
        tools_dict = {t.name: t for t in self.tools}
        results = []

        for tool_call in tool_calls:
            if tool_call.name not in tools_dict:
                results.append(ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="error",
                    content=[f"Tool '{tool_call.name}' not found"],
                ))
                continue

            tool_obj = tools_dict[tool_call.name]

            # Stage 1: before_tool callbacks (a non-None return replaces the tool call)
            skip = False
            for cb in self.before_tool_callbacks:
                cb_result = cb(context, tool_call)
                if hasattr(cb_result, "__await__"):
                    cb_result = await cb_result
                if cb_result is not None:
                    results.append(ToolResult(
                        tool_call_id=tool_call.tool_call_id,
                        name=tool_call.name,
                        status="success",
                        content=[cb_result],
                    ))
                    skip = True
                    break

            if skip:
                continue

            # Stage 2: execute the tool
            try:
                output = await tool_obj(context, **tool_call.arguments)
                tool_result = ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="success",
                    content=[output],
                )
            except Exception as e:
                tool_result = ToolResult(
                    tool_call_id=tool_call.tool_call_id,
                    name=tool_call.name,
                    status="error",
                    content=[str(e)],
                )

            # Stage 3: after_tool callbacks (a non-None return replaces the ToolResult)
            for cb in self.after_tool_callbacks:
                cb_result = cb(context, tool_result)
                if hasattr(cb_result, "__await__"):
                    cb_result = await cb_result
                if cb_result is not None:
                    tool_result = cb_result

            results.append(tool_result)

        if results:
            tool_event = Event(
                execution_id=context.execution_id,
                author=self.name,
                content=results,
            )
            context.add_event(tool_event)

    # ------------------------------------------------------------------ #
    # Internal methods
    # ------------------------------------------------------------------ #

    def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
        """Build an LlmRequest from the current context."""
        # Flatten events into content items
        flat_contents = []
        for event in context.events:
            flat_contents.extend(event.content)

        # With an output tool the LLM must always call a tool (eventually final_answer)
        if self.output_tool_name:
            tool_choice = "required"
        elif self.tools:
            tool_choice = "auto"
        else:
            tool_choice = None

        return LlmRequest(
            model_id=self.model.model,
            instructions=[self.instructions] if self.instructions else [],
            contents=flat_contents,
            tools=self.tools,
            tool_choice=tool_choice,
        )

    def _is_final_response(self, event: Event) -> bool:
        """Check if this event contains a final response."""
        if self.output_tool_name:
            return any(
                isinstance(item, ToolResult)
                and item.name == self.output_tool_name
                and item.status == "success"
                for item in event.content
            )

        has_tool_calls = any(isinstance(c, ToolCall) for c in event.content)
        has_tool_results = any(isinstance(c, ToolResult) for c in event.content)
        return not has_tool_calls and not has_tool_results

    def _extract_final_result(self, event: Event) -> Any:
        """Extract the final result from an event."""
        if self.output_tool_name:
            for item in event.content:
                if (
                    isinstance(item, ToolResult)
                    and item.name == self.output_tool_name
                    and item.status == "success"
                    and item.content
                ):
                    return item.content[0]

        for item in event.content:
            if isinstance(item, Message) and item.role == "assistant":
                return item.content
        return None

    def _setup_tools(self, tools: List[BaseTool]) -> List[BaseTool]:
        """Wrap plain functions as FunctionTools and add the output tool if needed."""
        tools = [t if isinstance(t, BaseTool) else FunctionTool(t) for t in tools]

        if self.output_type is not None:
            tools.append(self._create_output_tool())
            self.output_tool_name = "final_answer"

        return tools

    def _create_output_tool(self) -> FunctionTool:
        """Create the final_answer tool whose parameters are the output_type schema."""
        output_schema = self.output_type.model_json_schema()
        output_schema.pop("title", None)
        output_schema.pop("$defs", None)

        description = "Return the final structured answer matching the required schema."
        tool_definition = format_tool_definition(
            "final_answer",
            description,
            {
                "type": "object",
                "properties": {"output": output_schema},
                "required": ["output"],
            },
        )

        output_type = self.output_type

        def final_answer(output: dict) -> BaseModel:
            return output_type.model_validate(output)

        return FunctionTool(
            func=final_answer,
            name="final_answer",
            description=description,
            tool_definition=tool_definition,
        )

    def _log_response(self, response: LlmResponse):
        """Log LLM response for verbose mode."""
        for item in response.content:
            if isinstance(item, Message):
                logger.info(f"[{self.name}] {item.content}")
            elif isinstance(item, ToolCall):
                logger.info(f"[{self.name}] Tool call: {item.name}({item.arguments})")
