#!/usr/bin/env python3
"""
Q&A CLI for Raw Voice Agent Transcripts

Ask analytical questions about transcript data without LLM analysis preprocessing.
Uses sampled/*.json and preprocesses on-the-fly (coalesces fragmented messages).

Why use this instead of ask.py?
- Skip expensive per-call LLM analysis step entirely
- Great for quick exploration and hypothesis testing
- Works directly after sampling (no batch_analyze needed)

Workflow:
    # Option A: Using run_analysis.py --sample-only
    python3 tools/run_analysis.py -n 30 --sample-only
    python3 tools/ask_raw.py "What are customers calling about?"

    # Option B: Using sample_transcripts.py directly
    python3 tools/sample_transcripts.py -n 30 --run-id my_run
    python3 tools/ask_raw.py "What are customers calling about?" --run-id my_run

Usage:
    python3 tools/ask_raw.py "What are customers calling about?"
    python3 tools/ask_raw.py "Why do calls get escalated?" --limit 20 --verbose
    python3 tools/ask_raw.py "What patterns?" --run-dir runs/my_run
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

# Import preprocessing function for message coalescing
from preprocess_transcript import preprocess_json_transcript

from run_utils import (
    add_run_arguments, resolve_run_from_args, get_run_paths,
    prompt_for_run, confirm_or_select_run, require_explicit_run_noninteractive
)


QA_SYSTEM_PROMPT = """You are a call center analytics expert. You analyze voice agent call transcripts
to answer questions about call patterns, customer needs, and agent performance.

## Data Structure

Each call shows:
- Call ID (Unique, 8 chars, good for distinct call counts), duration, and how it ended
- Conversation turns with role (A=Agent, C=Customer) and text

## Your Task

Analyze the conversation transcripts to answer the user's question.

Guidelines:
- Read through conversations to identify patterns
- Look for: outcomes, issues, friction points, customer sentiment, common requests
- Provide counts/percentages where relevant
- Cite only a FEW calls (2-4) that best ILLUSTRATE your findings
- Quote brief relevant excerpts from conversations as evidence
- Do NOT list every call - just the most representative examples
- If the data doesn't fully answer the question, say so
- Structure your answer clearly with sections/bullets if appropriate.

CRITICAL: When asked for call counts (nmatching a given trait/criteria etc..) make sure to count the distinct call IDs
CRITICAL: When asked for breakdowns/partitions etc.. make sure the sum of all your items counts do match the total of unique calls analyzed.
"""


def get_genai_client() -> genai.Client:
    """Get configured Google GenAI client."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


def load_transcripts(transcripts_dir: Path) -> list[tuple[str, dict]]:
    """Load and preprocess all transcript JSON files from directory.

    Preprocessing coalesces fragmented ASR messages into proper turns.

    Returns:
        List of (call_id, preprocessed_dict) tuples
    """
    transcripts = []

    if not transcripts_dir.exists():
        raise FileNotFoundError(f"Transcripts directory not found: {transcripts_dir}")

    json_files = list(transcripts_dir.glob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No transcript JSON files found in {transcripts_dir}")

    for json_file in json_files:
        try:
            # Preprocess on-the-fly: coalesces messages into turns
            preprocessed = preprocess_json_transcript(json_file)
            call_id = preprocessed["call_id"]
            transcripts.append((call_id, preprocessed))
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {json_file}: {e}", file=sys.stderr)
            continue

    return transcripts


def sample_transcripts(transcripts: list[tuple[str, dict]], limit: int) -> list[tuple[str, dict]]:
    """Random sample up to limit transcripts."""
    if len(transcripts) <= limit:
        return transcripts
    return random.sample(transcripts, limit)


def format_call_for_prompt(call_id: str, preprocessed: dict) -> str:
    """Format a preprocessed transcript for inclusion in the LLM prompt.

    Uses compact format: A/C for roles, coalesced turns.
    """
    short_id = call_id[:8]
    metadata = preprocessed.get("metadata", {})
    duration = metadata.get("duration", 0) or 0
    ended_reason = metadata.get("ended_reason", "unknown")
    turns = preprocessed.get("turns", [])

    lines = [f"### Call {short_id} ({duration:.0f}s, {ended_reason})"]

    for turn in turns:
        role = "A" if turn.get("role") == "assistant" else "C"
        text = turn.get("text", "")
        if text:
            lines.append(f"{role}: {text}")

    return "\n".join(lines)


def build_qa_prompt(question: str, sampled: list[tuple[str, dict]]) -> str:
    """Assemble the full prompt for the LLM."""
    # Format each call
    call_data = "\n\n".join(format_call_for_prompt(call_id, t) for call_id, t in sampled)

    return f"""## Question
{question}

## Call Transcripts ({len(sampled)} calls sampled)

{call_data}

---

Answer the question based on the transcripts above. Cite specific evidence and quote relevant excerpts.

CRITICAL: When asked for Call counts (nmatching a given trait/criteria etc..) make sure to thoroughly count ditinct call IDs
CRITICAL: When asked for breakdowns/partitions etc.. make sure the sum of all your items counts do match the total of unique calls analyzed.
"""


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

    # Note: thinking tokens count against max_output_tokens for Gemini 3 models.
    # With large contexts, the model may use 8K+ thinking tokens, so we need
    # headroom for both thinking AND the actual answer.
    config = types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=131072,
        system_instruction=QA_SYSTEM_PROMPT,
    )

    # Override with thinking config if specified
    if thinking_level:
        config = types.GenerateContentConfig(
            temperature=0,
            max_output_tokens=131072,
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
    total_transcripts: int,
    sampled_count: int,
    sampled_call_ids: list[str]
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
        "total_transcripts_available": total_transcripts,
        "sampled_count": sampled_count,
        "source": "preprocessed_transcripts",
        "status": "pending",  # Will be updated to "completed" or "failed"
        "sampled_call_ids": sampled_call_ids,
    }
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
    total_transcripts: int,
    sampled_count: int,
    sampled_call_ids: list[str],
    status: str = "completed"
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
        "total_transcripts_available": total_transcripts,
        "sampled_count": sampled_count,
        "source": "preprocessed_transcripts",
        "status": status,
        "usage_stats": usage_stats,
        "sampled_call_ids": sampled_call_ids,
    }
    metadata_file = ask_dir / "metadata.json"
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about raw call transcripts (without LLM preprocessing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tools/ask_raw.py "What are customers calling about?"
  python3 tools/ask_raw.py "Why do calls get escalated?" --verbose
  python3 tools/ask_raw.py "What friction points appear in conversations?" --limit 20
  python3 tools/ask_raw.py "Main patterns?" --run-dir runs/my_run

Workflow (skips expensive analysis step):
  python3 tools/run_analysis.py -n 30 --sample-only
  python3 tools/ask_raw.py "What patterns do you see?"
"""
    )
    parser.add_argument("question", help="The question to answer")
    parser.add_argument("--sampled-dir", type=Path, default=Path("sampled"),
                        help="Directory containing sampled transcript JSONs (default: sampled/)")
    parser.add_argument("--limit", type=int, default=30,
                        help="Max calls to sample for context (default: 30, lower than ask.py due to larger transcripts)")
    parser.add_argument("--model", default="gemini-3-pro-preview",
                        help="LLM model to use (default: gemini-3-pro-preview)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show sampling info (count loaded, count sampled)")
    parser.add_argument("--stats", action="store_true",
                        help="Show token usage stats after the answer")
    parser.add_argument("--output-dir", type=Path, default=Path("asks_raw"),
                        help="Directory to save ask results (default: asks_raw/)")
    parser.add_argument("--thinking", choices=["minimal", "low", "medium", "high"],
                        help="Set thinking level for Gemini 3 models (default: model default)")

    # v4.3: Run-based isolation
    add_run_arguments(parser)

    args = parser.parse_args()

    # v4.3: Resolve run directory from --run-dir or --run-id
    project_dir = Path(__file__).parent.parent
    run_dir, run_id, source = resolve_run_from_args(args, project_dir)

    # Non-interactive mode requires explicit --run-id or --run-dir
    require_explicit_run_noninteractive(source)

    # Interactive run selection/confirmation
    if source in (".last_run", "$LAST_RUN"):
        run_dir, run_id = confirm_or_select_run(project_dir, run_dir, run_id, source)
    elif run_dir is None:
        run_dir, run_id = prompt_for_run(project_dir)

    if run_dir:
        paths = get_run_paths(run_dir, project_dir)
        args.sampled_dir = paths["sampled_dir"]
        print(f"Using run: {run_id} ({run_dir})", file=sys.stderr)
    else:
        # Resolve sampled directory relative to script location if needed
        if not args.sampled_dir.is_absolute():
            if not args.sampled_dir.exists():
                args.sampled_dir = project_dir / args.sampled_dir

    # 1. Load all transcripts
    print(f"Loading transcripts from {args.sampled_dir}...", file=sys.stderr)
    try:
        transcripts = load_transcripts(args.sampled_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # 2. Random sample up to limit
    sampled = sample_transcripts(transcripts, args.limit)
    print(f"Sampled {len(sampled)} of {len(transcripts)} transcripts", file=sys.stderr)

    # 3. Build prompt
    prompt = build_qa_prompt(args.question, sampled)
    prompt_tokens_est = len(prompt) // 4

    # 4. Save inputs BEFORE LLM call (so prompt is preserved if call fails)
    sampled_call_ids = [call_id for call_id, _ in sampled]
    ask_dir, timestamp = prepare_ask_dir(
        output_dir=args.output_dir,
        question=args.question,
        prompt=prompt,
        model=args.model,
        limit=args.limit,
        total_transcripts=len(transcripts),
        sampled_count=len(sampled),
        sampled_call_ids=sampled_call_ids
    )
    print(f"Saved prompt to: {ask_dir}", file=sys.stderr)

    # 5. Call LLM
    print(f"Calling {args.model} (~{prompt_tokens_est:,} input tokens)...", file=sys.stderr)

    try:
        answer, usage_stats = call_llm(prompt, args.model, args.thinking, args.verbose)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        finalize_ask_results(ask_dir, timestamp, args.question, f"Error: {e}", {},
                            args.model, args.limit, len(transcripts), len(sampled),
                            sampled_call_ids, status="failed")
        return 1
    except Exception as e:
        print(f"LLM error: {e}", file=sys.stderr)
        finalize_ask_results(ask_dir, timestamp, args.question, f"LLM error: {e}", {},
                            args.model, args.limit, len(transcripts), len(sampled),
                            sampled_call_ids, status="failed")
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
        total_transcripts=len(transcripts),
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
