"""Run the agents_bundle ReAct agent from the command line.

    uv run python main.py "What is 1234 * 5678?"
    uv run python main.py --trace "Who won the 2024 Nobel Prize in Physics?"
    uv run python main.py            # interactive: ask questions until you type 'exit'
"""

import argparse
import asyncio

from dotenv import load_dotenv

from agents_bundle import DEFAULT_MODEL, Agent, AgentResult, LlmClient
from agents_bundle.tools import calculator, search_web

load_dotenv()


def build_agent(model: str, max_steps: int) -> Agent:
    return Agent(
        model=LlmClient(model=model),
        tools=[calculator, search_web],
        instructions="You are a helpful assistant. Use tools when they help you answer accurately.",
        max_steps=max_steps,
    )


def print_trace(result: AgentResult) -> None:
    """Print every event of the run: user input, tool calls, tool results, answers."""
    for event in result.context.events:
        for item in event.content:
            if item.type == "message":
                print(f"  [{event.author}] {item.role}: {item.content}")
            elif item.type == "tool_call":
                print(f"  [{event.author}] -> {item.name}({item.arguments})")
            elif item.type == "tool_result":
                print(f"  [{event.author}] <- {item.name} {item.status}: {str(item.content)[:200]}")


async def ask(agent: Agent, question: str, trace: bool) -> None:
    result = await agent.run(question)
    if trace:
        print_trace(result)
    if result.output is None:
        print(f"No final answer after {result.context.current_step} steps (max_steps reached).")
    else:
        print(f"\n{result.output}\n({result.context.current_step} steps)")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the ReAct agent.")
    parser.add_argument("question", nargs="?", help="Question to ask. Omit for interactive mode.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"LiteLLM model name (default: {DEFAULT_MODEL})")
    parser.add_argument("--max-steps", type=int, default=10, help="Maximum think-act cycles (default: 10)")
    parser.add_argument("--trace", action="store_true", help="Print every tool call and result")
    args = parser.parse_args()

    agent = build_agent(args.model, args.max_steps)

    if args.question:
        await ask(agent, args.question, args.trace)
        return

    print(f"Agent ready ({args.model}). Type 'exit' to quit.")
    while True:
        question = input("\nYou: ").strip()
        if question.lower() in ("exit", "quit"):
            break
        if question:
            await ask(agent, question, args.trace)


if __name__ == "__main__":
    asyncio.run(main())
