"""Evaluate the agent on GAIA level 1 and compare it with the LLM alone (section 4.8).

    uv run python examples/run_gaia.py --limit 10            # agent with tools
    uv run python examples/run_gaia.py --limit 10 --no-tools # LLM alone (baseline)

GAIA is a gated dataset: request access at https://huggingface.co/datasets/gaia-benchmark/GAIA
and put HF_TOKEN in .env.
"""

import argparse
import asyncio

from datasets import load_dataset
from dotenv import load_dotenv

from agents_bundle import DEFAULT_MODEL
from agents_bundle.evaluation.gaia_benchmark import run_experiment
from agents_bundle.tools import calculator, search_web

load_dotenv()


def load_problems(limit: int, include_files: bool) -> list[dict]:
    problems = load_dataset("gaia-benchmark/GAIA", "2023_level1", split="validation")
    # Some questions come with an attached file (spreadsheet, image, audio...).
    # Our agent has no tool to read files yet, so skip them by default.
    if not include_files:
        problems = problems.filter(lambda p: not p["file_name"])
    return list(problems.select(range(min(limit, len(problems)))))


def print_report(results: dict[str, list]) -> None:
    for model, rows in results.items():
        correct = sum(r["correct"] for r in rows)
        errors = sum(1 for r in rows if r.get("error"))
        unsolvable = sum(1 for r in rows if r.get("is_solvable") is False)
        print(f"\n{model}: {correct}/{len(rows)} correct ({correct / len(rows):.0%})"
              f" | said unsolvable: {unsolvable} | errors: {errors}")
        for r in rows:
            mark = "OK " if r["correct"] else "ERR" if r.get("error") else "-- "
            detail = r.get("error") or r.get("unsolvable_reason") or ""
            print(f"  {mark} predicted={r['prediction']!r:30.30} expected={r['answer']!r:20.20}"
                  f" steps={r.get('steps', '-')} {detail[:60]}")


async def main() -> None:
    parser = argparse.ArgumentParser(description="Run the GAIA level 1 benchmark.")
    parser.add_argument("--limit", type=int, default=10, help="Number of problems (default: 10)")
    parser.add_argument("--models", nargs="+", default=[DEFAULT_MODEL], help="One or more LiteLLM model names")
    parser.add_argument("--no-tools", action="store_true", help="LLM alone, no agent loop (baseline)")
    parser.add_argument("--include-files", action="store_true", help="Include problems with attached files")
    args = parser.parse_args()

    problems = load_problems(args.limit, args.include_files)
    tools = None if args.no_tools else [calculator, search_web]
    print(f"Running {len(problems)} problems on {args.models} ({'LLM only' if tools is None else 'agent + tools'})")

    results = await run_experiment(problems, args.models, tools=tools)
    print_report(results)


if __name__ == "__main__":
    asyncio.run(main())
