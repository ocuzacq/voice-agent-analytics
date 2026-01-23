#!/usr/bin/env python3
"""
Q&A CLI for Voice Agent Analytics (v4.1)

Ask analytical questions about call data without generating full reports.

v4.1 additions:
- Run-based isolation support via --run-dir argument
- Reads from run's analyses/ when --run-dir is specified
- Records run_id in metadata for traceability
- Backwards compatible with legacy flat directory mode

v4.0 additions:
- Intent display: Shows intent and intent_context when available
- Sentiment journey: Displays sentiment_start → sentiment_end transitions
- Updated field mappings: disposition (unified), failure_type, verbatim, coaching
- Backwards-compatible with v3.x analyses (uses fallback field names)

Usage:
    python3 tools/ask.py "Why do calls fail?"
    python3 tools/ask.py "What causes name issues?" --limit 50 --verbose
    python3 tools/ask.py "What are the top customer intents?" --stats
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

from run_utils import (
    add_run_arguments, resolve_run_from_args, get_run_paths,
    prompt_for_run, confirm_or_select_run, require_explicit_run_noninteractive
)


QA_SYSTEM_PROMPT = """You are a call center analytics expert. You answer questions about voice agent
performance based on analyzed call data.

## Data Structure

Each call analysis contains (v4.0 fields, with v3.x fallback names in parentheses):

**Core Fields:**
- call_id: unique identifier
- schema_version: "v4.0" or "v3.9.x"
- disposition (v3: outcome + call_disposition): unified call outcome
- summary: executive overview
- verbatim (v3: customer_verbatim): direct customer quote

**Intent (v4.0 only):**
- intent: primary customer request (normalized phrase)
- intent_context: underlying reason/situation
- secondary_intent: additional requests

**Sentiment (v4.0 only):**
- sentiment_start: customer mood at call start
- sentiment_end: customer mood at call end

**Quality Scores (1-5):**
- effectiveness (v3: agent_effectiveness)
- quality (v3: conversation_quality)
- effort (v3: customer_effort)

**Failure Analysis:**
- failure_type (v3: failure_point): root cause category
- failure_detail (v3: failure_description): what went wrong

**Friction (v4.0: top-level, v3.x: nested under friction):**
- turns: total conversation turns
- clarifications, corrections, loops: friction events

**Coaching:**
- coaching (v3: agent_miss_detail): agent improvement recommendations

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
    """Format a single call analysis for inclusion in the LLM prompt.

    v4.0: Supports both v4.0 and v3.x field names with backwards compatibility.
    """
    call_id = analysis.get("call_id", "unknown")[:8]

    # v4.0: unified disposition | v3.x: outcome + call_disposition
    disposition = analysis.get("disposition") or analysis.get("call_disposition", "unknown")
    outcome = analysis.get("outcome")  # Only in v3.x

    summary = analysis.get("summary", "")

    # v4.0: failure_type/failure_detail | v3.x: failure_point/failure_description
    failure_type = analysis.get("failure_type") or analysis.get("failure_point", "none")
    failure_detail = analysis.get("failure_detail") or analysis.get("failure_description", "")

    # v4.0: verbatim | v3.x: customer_verbatim
    verbatim = analysis.get("verbatim") or analysis.get("customer_verbatim", "")

    # v4.0 only: intent, sentiment, coaching
    intent = analysis.get("intent", "")
    intent_context = analysis.get("intent_context", "")
    sentiment_start = analysis.get("sentiment_start", "")
    sentiment_end = analysis.get("sentiment_end", "")
    coaching = analysis.get("coaching") or analysis.get("agent_miss_detail", "")

    # Friction: v4.0 top-level | v3.x nested under friction
    friction = analysis.get("friction", {})
    turns = analysis.get("turns") or friction.get("turns", 0)
    loops = analysis.get("loops") or friction.get("loops", [])
    clarifications = analysis.get("clarifications") or friction.get("clarifications", [])
    corrections = analysis.get("corrections") or friction.get("corrections", [])

    # Build the formatted entry
    lines = [f"### Call {call_id}"]

    # v4.0: single disposition | v3.x: outcome (disposition)
    if outcome:
        lines.append(f"- Outcome: {outcome} ({disposition})")
    else:
        lines.append(f"- Disposition: {disposition}")

    # Intent (v4.0 only) - critical for WHY customer called
    if intent:
        intent_line = f"- Intent: {intent}"
        if intent_context:
            intent_line += f" ({intent_context})"
        lines.append(intent_line)

    # Sentiment journey (v4.0 only)
    if sentiment_start and sentiment_end:
        lines.append(f"- Sentiment: {sentiment_start} → {sentiment_end}")

    if summary:
        lines.append(f"- Summary: {summary}")

    if failure_type and failure_type != "none" and failure_detail:
        lines.append(f"- Failure: {failure_type} - {failure_detail}")
    elif failure_type and failure_type != "none":
        lines.append(f"- Failure: {failure_type}")

    if coaching:
        lines.append(f"- Coaching: {coaching}")

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


def call_llm(prompt: str, model: str, thinking_level: str = None, verbose: bool = False) -> tuple[str, dict]:
    """Call Gemini and return the answer text with usage stats.

    Args:
        prompt: The prompt to send to the model.
        model: The model name to use.
        thinking_level: Optional thinking level for Gemini 3 models (minimal/low/medium/high).
        verbose: Whether to print verbose output.

    Returns:
        tuple: (answer_text, usage_stats_dict)
    """
    client = get_genai_client()

    config = types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=8192,
        system_instruction=QA_SYSTEM_PROMPT,
    )

    # Override with thinking config if specified
    if thinking_level:
        config = types.GenerateContentConfig(
            temperature=0.3,
            max_output_tokens=8192,
            system_instruction=QA_SYSTEM_PROMPT,
            thinking_config=types.ThinkingConfig(thinking_level=thinking_level.upper()),
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


def prepare_ask_dir(
    output_dir: Path,
    question: str,
    prompt: str,
    model: str,
    limit: int,
    total_analyses: int,
    sampled_count: int,
    sampled_call_ids: list[str],
    run_id: str | None = None
) -> tuple[Path, datetime]:
    """Create output directory and save inputs BEFORE LLM call.

    Saves immediately (before LLM call):
        - question.txt
        - prompt.txt
        - system_prompt.txt
        - metadata.json (partial, without answer/usage)

    Returns:
        tuple: (ask_dir path, timestamp for later use)
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

    # Save prompt.txt (full LLM prompt for replay)
    prompt_file = ask_dir / "prompt.txt"
    prompt_file.write_text(prompt, encoding='utf-8')

    # Save system_prompt.txt
    system_prompt_file = ask_dir / "system_prompt.txt"
    system_prompt_file.write_text(QA_SYSTEM_PROMPT, encoding='utf-8')

    # Save partial metadata.json (will be updated after LLM call)
    metadata = {
        "timestamp": now.isoformat(),
        "question": question,
        "model": model,
        "limit": limit,
        "total_analyses_available": total_analyses,
        "sampled_count": sampled_count,
        "status": "pending",  # Will be updated to "completed" or "failed"
        "sampled_call_ids": sampled_call_ids,
    }
    # v4.1: Include run_id for traceability
    if run_id:
        metadata["run_id"] = run_id
    metadata_file = ask_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)

    return ask_dir, now


def finalize_ask_results(
    ask_dir: Path,
    timestamp: datetime,
    question: str,
    answer: str,
    usage_stats: dict,
    model: str,
    limit: int,
    total_analyses: int,
    sampled_count: int,
    sampled_call_ids: list[str],
    status: str = "completed",
    run_id: str | None = None
) -> None:
    """Save answer and finalize metadata AFTER LLM call.

    Saves:
        - answer.md
        - metadata.json (complete, with usage stats)
    """
    # Save answer.md
    answer_file = ask_dir / "answer.md"
    answer_file.write_text(answer or "(No answer generated)", encoding='utf-8')

    # Save complete metadata.json
    metadata = {
        "timestamp": timestamp.isoformat(),
        "question": question,
        "model": model,
        "limit": limit,
        "total_analyses_available": total_analyses,
        "sampled_count": sampled_count,
        "status": status,
        "usage_stats": usage_stats,
        "sampled_call_ids": sampled_call_ids,
    }
    # v4.1: Include run_id for traceability
    if run_id:
        metadata["run_id"] = run_id
    metadata_file = ask_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


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
    parser.add_argument("--thinking", choices=["minimal", "low", "medium", "high"],
                        help="Set thinking level for Gemini 3 models (default: model default)")

    # v4.1: Run-based isolation
    add_run_arguments(parser)

    args = parser.parse_args()

    # v4.1: Resolve run directory from --run-dir or --run-id
    project_dir = Path(__file__).parent.parent
    run_dir, run_id, source = resolve_run_from_args(args, project_dir)

    # Non-interactive mode requires explicit --run-id or --run-dir
    require_explicit_run_noninteractive(source)

    # Interactive run selection/confirmation
    if source in (".last_run", "$LAST_RUN"):
        # Implicit source - ask for confirmation
        run_dir, run_id = confirm_or_select_run(project_dir, run_dir, run_id, source)
    elif run_dir is None:
        # No run specified - show selection menu
        run_dir, run_id = prompt_for_run(project_dir)

    if run_dir:
        paths = get_run_paths(run_dir, project_dir)
        args.analyses_dir = paths["analyses_dir"]
        print(f"Using run: {run_id} ({run_dir})", file=sys.stderr)
    else:
        # Resolve analyses directory relative to script location if needed
        if not args.analyses_dir.is_absolute():
            # Try relative to current working directory first
            if not args.analyses_dir.exists():
                # Try relative to script directory
                args.analyses_dir = project_dir / args.analyses_dir

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

    # 4. Save inputs BEFORE LLM call (so prompt is preserved if call fails)
    sampled_call_ids = [a.get("call_id", "unknown") for a in sampled]
    ask_dir, timestamp = prepare_ask_dir(
        output_dir=args.output_dir,
        question=args.question,
        prompt=prompt,
        model=args.model,
        limit=args.limit,
        total_analyses=len(analyses),
        sampled_count=len(sampled),
        sampled_call_ids=sampled_call_ids,
        run_id=run_id
    )
    print(f"Saved prompt to: {ask_dir}", file=sys.stderr)

    # 5. Call LLM
    print(f"Calling {args.model} (~{prompt_tokens_est:,} input tokens)...", file=sys.stderr)

    try:
        answer, usage_stats = call_llm(prompt, args.model, args.thinking, args.verbose)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        finalize_ask_results(ask_dir, timestamp, args.question, f"Error: {e}", {},
                            args.model, args.limit, len(analyses), len(sampled),
                            sampled_call_ids, status="failed", run_id=run_id)
        return 1
    except Exception as e:
        print(f"LLM error: {e}", file=sys.stderr)
        finalize_ask_results(ask_dir, timestamp, args.question, f"LLM error: {e}", {},
                            args.model, args.limit, len(analyses), len(sampled),
                            sampled_call_ids, status="failed", run_id=run_id)
        return 1

    print("", file=sys.stderr)  # Blank line before answer

    # 6. Print answer
    print(answer)

    # 7. Finalize results (save answer and complete metadata)
    finalize_ask_results(
        ask_dir=ask_dir,
        timestamp=timestamp,
        question=args.question,
        answer=answer,
        usage_stats=usage_stats,
        model=args.model,
        limit=args.limit,
        total_analyses=len(analyses),
        sampled_count=len(sampled),
        sampled_call_ids=sampled_call_ids,
        run_id=run_id
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
