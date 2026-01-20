#!/usr/bin/env python3
"""
Q&A CLI for Voice Agent Analytics

Ask analytical questions about call data without generating full reports.

Usage:
    python3 tools/ask.py "Why do calls fail?"
    python3 tools/ask.py "What causes name issues?" --limit 50 --verbose
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

# Load .env file if present
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types


QA_SYSTEM_PROMPT = """You are a call center analytics expert. You answer questions about voice agent
performance based on analyzed call data.

## Data Structure

Each call analysis contains:
- call_id: unique identifier
- outcome: "resolved" or "abandoned"
- call_disposition: success/failure classification
- failure_point: root cause category (none, policy_gap, tech_issue, etc.)
- failure_description: what went wrong
- customer_verbatim: direct customer quote
- summary: executive overview
- friction: conversation quality (turns, clarifications, corrections, loops)
- agent_effectiveness, conversation_quality, customer_effort: 1-5 ratings

## Your Task

Answer the user's question based ONLY on the provided call data.

Guidelines:
- Identify patterns and provide counts/percentages where relevant
- Cite only a FEW calls (2-4) that best ILLUSTRATE your findings
- Do NOT list every call - just the most representative examples
- If the data doesn't fully answer the question, say so
- Structure your answer clearly with sections/bullets if appropriate"""


def get_genai_client() -> genai.Client:
    """Get configured Google GenAI client."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


def load_analyses(analyses_dir: Path) -> list[dict]:
    """Load all analysis JSON files from directory."""
    analyses = []

    if not analyses_dir.exists():
        raise FileNotFoundError(f"Analyses directory not found: {analyses_dir}")

    json_files = list(analyses_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No analysis JSON files found in {analyses_dir}")

    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                analysis = json.load(f)
                analyses.append(analysis)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {json_file}: {e}", file=sys.stderr)
            continue

    return analyses


def sample_analyses(analyses: list[dict], limit: int) -> list[dict]:
    """Random sample up to limit analyses."""
    if len(analyses) <= limit:
        return analyses
    return random.sample(analyses, limit)


def format_call_for_prompt(analysis: dict) -> str:
    """Format a single call analysis for inclusion in the LLM prompt."""
    call_id = analysis.get("call_id", "unknown")[:8]
    outcome = analysis.get("outcome", "unknown")
    disposition = analysis.get("call_disposition", "unknown")
    summary = analysis.get("summary", "")
    failure_point = analysis.get("failure_point", "none")
    failure_desc = analysis.get("failure_description", "")
    verbatim = analysis.get("customer_verbatim", "")

    # Friction data
    friction = analysis.get("friction", {})
    turns = friction.get("turns", 0)
    loops = friction.get("loops", [])
    clarifications = friction.get("clarifications", [])
    corrections = friction.get("corrections", [])

    # Build the formatted entry
    lines = [f"### Call {call_id}"]
    lines.append(f"- Outcome: {outcome} ({disposition})")

    if summary:
        lines.append(f"- Summary: {summary}")

    if failure_point != "none" and failure_desc:
        lines.append(f"- Failure: {failure_point} - {failure_desc}")
    elif failure_point != "none":
        lines.append(f"- Failure: {failure_point}")

    if verbatim:
        lines.append(f'- Customer: "{verbatim}"')

    # Friction summary
    friction_parts = [f"{turns} turns"]
    if loops:
        friction_parts.append(f"{len(loops)} loops")
    if clarifications:
        friction_parts.append(f"{len(clarifications)} clarifications")
    if corrections:
        friction_parts.append(f"{len(corrections)} corrections")
    lines.append(f"- Friction: {', '.join(friction_parts)}")

    return "\n".join(lines)


def build_qa_prompt(question: str, sampled: list[dict]) -> str:
    """Assemble the full prompt for the LLM."""
    # Format each call
    call_data = "\n\n".join(format_call_for_prompt(a) for a in sampled)

    return f"""## Question
{question}

## Call Data ({len(sampled)} calls sampled)

{call_data}

---

Answer the question based on the data above. Cite specific evidence."""


def call_llm(prompt: str, model: str, verbose: bool = False) -> tuple[str, dict]:
    """Call Gemini and return the answer text with usage stats.

    Returns:
        tuple: (answer_text, usage_stats_dict)
    """
    client = get_genai_client()

    config = types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=8192,
        system_instruction=QA_SYSTEM_PROMPT,
    )

    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=config,
    )

    # Extract usage stats
    usage_stats = {}
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        meta = response.usage_metadata
        usage_stats = {
            'input_tokens': getattr(meta, 'prompt_token_count', None),
            'output_tokens': getattr(meta, 'candidates_token_count', None),
            'thinking_tokens': getattr(meta, 'thoughts_token_count', None),
            'total_tokens': getattr(meta, 'total_token_count', None),
        }
        # Filter out None values
        usage_stats = {k: v for k, v in usage_stats.items() if v is not None}

    return response.text, usage_stats


def save_ask_results(
    output_dir: Path,
    question: str,
    answer: str,
    usage_stats: dict,
    model: str,
    limit: int,
    total_analyses: int,
    sampled_count: int,
    sampled_call_ids: list[str]
) -> Path:
    """Save the full ask call to a timestamped subfolder.

    Creates:
        asks/Jan20-10h40/
            question.txt
            answer.md
            metadata.json

    Returns:
        Path to the created subfolder
    """
    # Create timestamp folder name: Jan20-10h40
    now = datetime.now()
    folder_name = now.strftime("%b%d-%Hh%M")

    # Create the output directory
    ask_dir = output_dir / folder_name
    ask_dir.mkdir(parents=True, exist_ok=True)

    # Save question.txt
    question_file = ask_dir / "question.txt"
    question_file.write_text(question, encoding='utf-8')

    # Save answer.md
    answer_file = ask_dir / "answer.md"
    answer_file.write_text(answer or "(No answer generated)", encoding='utf-8')

    # Save metadata.json (usage_stats before large call_ids list for readability)
    metadata = {
        "timestamp": now.isoformat(),
        "question": question,
        "model": model,
        "limit": limit,
        "total_analyses_available": total_analyses,
        "sampled_count": sampled_count,
        "usage_stats": usage_stats,
        "sampled_call_ids": sampled_call_ids,
    }
    metadata_file = ask_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return ask_dir


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about call data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tools/ask.py "What are the main failure patterns?"
  python3 tools/ask.py "Why do calls get escalated?" --verbose
  python3 tools/ask.py "What causes name clarification issues?" --limit 50
"""
    )
    parser.add_argument("question", help="The question to answer")
    parser.add_argument("--analyses-dir", type=Path, default=Path("analyses"),
                        help="Directory containing analysis JSONs (default: analyses/)")
    parser.add_argument("--limit", type=int, default=100,
                        help="Max calls to sample for context (default: 100)")
    parser.add_argument("--model", default="gemini-3-pro-preview",
                        help="LLM model to use (default: gemini-3-pro-preview)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show sampling info (count loaded, count sampled)")
    parser.add_argument("--stats", action="store_true",
                        help="Show token usage stats after the answer")
    parser.add_argument("--output-dir", type=Path, default=Path("asks"),
                        help="Directory to save ask results (default: asks/)")

    args = parser.parse_args()

    # Resolve analyses directory relative to script location if needed
    if not args.analyses_dir.is_absolute():
        # Try relative to current working directory first
        if not args.analyses_dir.exists():
            # Try relative to script directory
            script_dir = Path(__file__).parent.parent
            args.analyses_dir = script_dir / args.analyses_dir

    # 1. Load all analyses
    print(f"Loading analyses from {args.analyses_dir}...", file=sys.stderr)
    try:
        analyses = load_analyses(args.analyses_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # 2. Random sample up to limit
    sampled = sample_analyses(analyses, args.limit)
    print(f"Sampled {len(sampled)} of {len(analyses)} analyses", file=sys.stderr)

    # 3. Build prompt
    prompt = build_qa_prompt(args.question, sampled)
    prompt_tokens_est = len(prompt) // 4

    # 4. Call LLM
    print(f"Calling {args.model} (~{prompt_tokens_est:,} input tokens)...", file=sys.stderr)

    try:
        answer, usage_stats = call_llm(prompt, args.model, args.verbose)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"LLM error: {e}", file=sys.stderr)
        return 1

    print("", file=sys.stderr)  # Blank line before answer

    # 5. Print answer
    print(answer)

    # 6. Save results to asks/ subfolder
    sampled_call_ids = [a.get("call_id", "unknown") for a in sampled]
    ask_dir = save_ask_results(
        output_dir=args.output_dir,
        question=args.question,
        answer=answer,
        usage_stats=usage_stats,
        model=args.model,
        limit=args.limit,
        total_analyses=len(analyses),
        sampled_count=len(sampled),
        sampled_call_ids=sampled_call_ids
    )
    print(f"\nSaved to: {ask_dir}", file=sys.stderr)

    # 7. Print token stats if requested
    if args.stats or args.verbose:
        print("\n---", file=sys.stderr)
        print("Token Usage:", file=sys.stderr)
        if usage_stats:
            if 'input_tokens' in usage_stats:
                print(f"  Input tokens:    {usage_stats['input_tokens']:,}", file=sys.stderr)
            if 'thinking_tokens' in usage_stats:
                print(f"  Thinking tokens: {usage_stats['thinking_tokens']:,}", file=sys.stderr)
            if 'output_tokens' in usage_stats:
                print(f"  Output tokens:   {usage_stats['output_tokens']:,}", file=sys.stderr)
            if 'total_tokens' in usage_stats:
                print(f"  Total tokens:    {usage_stats['total_tokens']:,}", file=sys.stderr)
        else:
            print("  (No usage data available)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
