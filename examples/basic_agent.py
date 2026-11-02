"""Run the agents_bundle ReAct agent: tool use and structured output.

    uv run python examples/basic_agent.py
"""

import asyncio
from typing import List, Literal

from dotenv import load_dotenv
from pydantic import BaseModel

from agents_bundle import Agent, LlmClient
from agents_bundle.tools import calculator, search_web

load_dotenv()


class SentimentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    key_phrases: List[str]


async def main():
    # Tool use: the agent decides to call the calculator
    agent = Agent(
        model=LlmClient(model="claude-haiku-5-5"),
        tools=[calculator, search_web],
        instructions="You are a helpful assistant",
    )
    result = await agent.run("What is 1234 * 5678?")
    print(f"Answer: {result.output} ({result.context.current_step} steps)")

    # Structured output: the agent must finish by calling final_answer
    agent = Agent(
        model=LlmClient(model="claude-haiku-5-5"),
        instructions="Analyze the sentiment of the user's text.",
        output_type=SentimentAnalysis,
    )
    result = await agent.run("I love this product! It works perfectly and exceeded my expectations.")
    print(f"Structured: {result.output!r}")


if __name__ == "__main__":
    asyncio.run(main())
