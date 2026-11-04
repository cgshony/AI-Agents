"""Core Agent class: the ReAct (think-act) loop.

Each step the agent *thinks* (asks the LLM what to do next) and *acts*
(executes the tool calls the LLM requested), recording everything as events
in an ExecutionContext. The loop ends when the LLM answers without calling
a tool (or calls final_answer when a structured output_type is set).

Extension points around the loop:
  - before_llm_callbacks: inspect/modify the request, or answer without
    calling the LLM (used for context optimization and caching).
  - before/after_tool_callbacks: guard, replace, or post-process tool calls.
  - requires_confirmation tools: pause the run until a human approves
    (human-in-the-loop); run() can later resume from the saved state.
  - session_manager: persist the conversation across run() calls.
  - memory_manager: learn from finished runs and recall similar ones.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Type

from pydantic import BaseModel

from agents_bundle.data_types import Event, Message, ToolCall, ToolResult
from agents_bundle.execution_context import (
    AgentResult,
    ExecutionContext,
    PendingToolCall,
    ToolConfirmation,
)
from agents_bundle.llm_client import LlmClient, LlmRequest, LlmResponse
from agents_bundle.tools.base_tool import BaseTool, FunctionTool
from agents_bundle.tools.tool_schema import format_tool_definition

if TYPE_CHECKING:
    from agents_bundle.memory.long_term import TaskMemoryManager
    from agents_bundle.memory.session import BaseSessionManager

logger = logging.getLogger(__name__)


class Agent:
    """Tool-calling agent with ReAct loop, callbacks, sessions, and memory."""

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
        session_manager: Optional["BaseSessionManager"] = None,
        memory_manager: Optional["TaskMemoryManager"] = None,
        before_llm_callbacks: list[Callable] | None = None,
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
        self.before_llm_callbacks = before_llm_callbacks or []
        self.session_manager = session_manager
        self.memory_manager = memory_manager
        self.tools = self._setup_tools(tools or [])

    # ------------------------------------------------------------------ #
    # Core loop
    # ------------------------------------------------------------------ #

    async def run(
        self,
        user_input: str | None = None,
        context: ExecutionContext | None = None,
        session_id: str | None = None,
        tool_confirmations: list[ToolConfirmation] | None = None,
        verbose: bool = False,
    ) -> AgentResult:
        """Execute the agent until a final result, max_steps, or a pending confirmation.

        user_input may be None when resuming (e.g. with tool_confirmations).
        """
        # Load or create the session (short-term memory across run() calls)
        session = None
        if session_id and self.session_manager:
            session = await self.session_manager.get_or_create(session_id)

        if context is None:
            context = ExecutionContext(
                session=session,
                session_manager=self.session_manager,
                memory_manager=self.memory_manager,
            )
            # Restore previous events/state from the session
            if session:
                context.events = list(session.events)
                context.state = dict(session.state)
        elif context.memory_manager is None:
            context.memory_manager = self.memory_manager

        # Human-in-the-loop resume: run (or deny) the tool calls that were paused
        if tool_confirmations:
            await self._process_confirmations(context, tool_confirmations)

        if user_input:
            user_event = Event(
                execution_id=context.execution_id,
                author="user",
                content=[Message(role="user", content=user_input)],
            )
            context.add_event(user_event)

        while not context.final_result and context.current_step < self.max_steps:
            result = await self.step(context, verbose=verbose)

            # Paused for confirmation: persist and hand control back to the caller
            if result and result.status == "pending_confirmation":
                await self._save_session(session, context)
                return result

            if context.events:
                last_event = context.events[-1]
                if self._is_final_response(last_event):
                    context.final_result = self._extract_final_result(last_event)

        # Long-term memory: distill this run into a TaskMemory
        if self.memory_manager:
            try:
                await self.memory_manager.save(context)
            except Exception as e:
                logger.warning(f"Failed to save memory: {e}")

        await self._save_session(session, context)
        return AgentResult(output=context.final_result, context=context)

    async def step(
        self,
        context: ExecutionContext,
        verbose: bool = False,
    ) -> AgentResult | None:
        """Perform one think-act cycle. Returns an AgentResult only when paused."""
        llm_request = await self._prepare_llm_request(context)

        # before_llm callbacks may modify the request in place (context optimization)
        # or return an LlmResponse to skip the LLM call entirely
        for callback in self.before_llm_callbacks:
            cb_result = callback(context, llm_request)
            if hasattr(cb_result, "__await__"):
                cb_result = await cb_result
            if isinstance(cb_result, LlmResponse):
                llm_response = cb_result
                break
        else:
            llm_response = await self.think(llm_request)

        if llm_response.error_message:
            raise RuntimeError(f"LLM request failed: {llm_response.error_message}")

        if verbose:
            self._log_response(llm_response)

        response_event = Event(
            execution_id=context.execution_id,
            author=self.name,
            content=llm_response.content,
        )
        context.add_event(response_event)

        # Execute tools if the LLM requested any (act records its own event)
        tool_calls = [c for c in llm_response.content if isinstance(c, ToolCall)]
        if tool_calls:
            result = await self.act(context, tool_calls)
            if result and result.status == "pending_confirmation":
                return result

        context.increment_step()
        return None

    async def think(self, llm_request: LlmRequest) -> LlmResponse:
        """Call the LLM to decide the next action."""
        return await self.model.generate(llm_request)

    async def act(
        self,
        context: ExecutionContext,
        tool_calls: List[ToolCall],
    ) -> AgentResult | None:
        """Execute the requested tools with callbacks; pause if any need confirmation."""
        tools_dict = {t.name: t for t in self.tools}
        results = []
        pending = []

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

            # Risky tools are not executed now: they are deferred to a human
            if tool_obj.requires_confirmation:
                pending.append(PendingToolCall(
                    tool_call=tool_call,
                    confirmation_message=tool_obj.get_confirmation_message(tool_call.arguments),
                ))
                continue

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
            tool_result = await self._execute_tool(context, tool_obj, tool_call, tool_call.arguments)

            # Stage 3: after_tool callbacks (a non-None return replaces the ToolResult)
            for cb in self.after_tool_callbacks:
                cb_result = cb(context, tool_result)
                if hasattr(cb_result, "__await__"):
                    cb_result = await cb_result
                if cb_result is not None:
                    tool_result = cb_result

            results.append(tool_result)

        if results:
            context.add_event(Event(
                execution_id=context.execution_id,
                author=self.name,
                content=results,
            ))

        # Pause: store pending calls in state so a later run() can resume them
        if pending:
            context.state["pending_tool_calls"] = [p.model_dump() for p in pending]
            return AgentResult(
                output=None,
                context=context,
                status="pending_confirmation",
                pending_tool_calls=pending,
            )

        return None

    # ------------------------------------------------------------------ #
    # Internal methods
    # ------------------------------------------------------------------ #

    async def _execute_tool(
        self,
        context: ExecutionContext,
        tool_obj: BaseTool,
        tool_call: ToolCall,
        arguments: dict,
    ) -> ToolResult:
        """Run one tool and wrap the output (or exception) in a ToolResult."""
        try:
            output = await tool_obj(context, **arguments)
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                status="success",
                content=[output],
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.tool_call_id,
                name=tool_call.name,
                status="error",
                content=[str(e)],
            )

    async def _process_confirmations(
        self,
        context: ExecutionContext,
        confirmations: list[ToolConfirmation],
    ) -> None:
        """Resume after a pause: run approved calls, record denials as errors."""
        raw_pending = context.state.pop("pending_tool_calls", [])
        pending = [PendingToolCall.model_validate(d) for d in raw_pending]

        tools_dict = {t.name: t for t in self.tools}
        results = []

        for pending_call in pending:
            tc = pending_call.tool_call
            confirmation = next(
                (c for c in confirmations if c.tool_call_id == tc.tool_call_id),
                None,
            )

            if confirmation and confirmation.approved and tc.name in tools_dict:
                args = confirmation.modified_arguments or tc.arguments
                results.append(await self._execute_tool(context, tools_dict[tc.name], tc, args))
            else:
                reason = f" Reason: {confirmation.reason}" if confirmation and confirmation.reason else ""
                results.append(ToolResult(
                    tool_call_id=tc.tool_call_id,
                    name=tc.name,
                    status="error",
                    content=[f"User denied the tool execution.{reason}"],
                ))

        if results:
            context.add_event(Event(
                execution_id=context.execution_id,
                author=self.name,
                content=results,
            ))

    async def _save_session(self, session, context: ExecutionContext) -> None:
        """Sync events/state back into the session and persist it."""
        if session and self.session_manager:
            session.events = list(context.events)
            session.state = dict(context.state)
            await self.session_manager.save(session)

    async def _prepare_llm_request(self, context: ExecutionContext) -> LlmRequest:
        """Build an LlmRequest from the current context."""
        flat_contents = []
        for event in context.events:
            flat_contents.extend(event.content)

        # Hidden tools (tool_definition=None, e.g. MemoryTool) are not sent to the LLM
        llm_tools = [t for t in self.tools if t.tool_definition is not None]

        # With an output tool the LLM must always call a tool (eventually final_answer)
        if self.output_tool_name:
            tool_choice = "required"
        elif llm_tools:
            tool_choice = "auto"
        else:
            tool_choice = None

        request = LlmRequest(
            model_id=self.model.model,
            instructions=[self.instructions] if self.instructions else [],
            contents=flat_contents,
            tools=llm_tools,
            tool_choice=tool_choice,
        )

        # Let tools modify the request (e.g. MemoryTool injects past experiences)
        for tool_obj in self.tools:
            await tool_obj.process_llm_request(context, request)

        return request

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
        """Wrap plain functions as FunctionTools and add built-in tools."""
        tools = [t if isinstance(t, BaseTool) else FunctionTool(t) for t in tools]

        if self.output_type is not None:
            tools.append(self._create_output_tool())
            self.output_tool_name = "final_answer"

        # Hidden tool that injects relevant long-term memories into each request
        if self.memory_manager:
            from agents_bundle.tools.memory_tool import MemoryTool
            tools.append(MemoryTool())

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
