#!/usr/bin/env python3
"""
Q&A CLI for v7 Structured Analyses

Query v7.0 per-intent analysis JSONs with full schema awareness.
Generates a schema description from the Pydantic models automatically,
so the LLM understands every field and enum value.

Usage:
    python3 tools/ask_struct_v7.py "What are the main customer requests?" \\
      --analyses-dir tests/golden/analyses_v6_review/

    python3 tools/ask_struct_v7.py "Recap all RCI-related calls" \\
      --analyses-dir tests/golden/analyses_v6_review/ --filter RCI

    python3 tools/ask_struct_v7.py "Why do transfers fail?" \\
      --analyses-dir path/ --limit 50 --stats
"""

import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, get_args, get_origin

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

# Allow importing schema.py from same directory
sys.path.insert(0, str(Path(__file__).parent))
from schema import CallAnalysis


# ---------------------------------------------------------------------------
# 1. Schema Description Generator
# ---------------------------------------------------------------------------

SKIP_FIELDS = {"scores"}


def _type_label(annotation) -> str:
    """Return a human-readable type label for a field annotation."""
    origin = get_origin(annotation)

    # Optional[X] → unwrap
    if origin is type(None):
        return "None"

    # Handle Optional (Union with None)
    if origin is type(None) or str(origin) == "typing.Union":
        args = get_args(annotation)
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            return f"optional {_type_label(non_none[0])}"
        return " | ".join(_type_label(a) for a in non_none)

    # Literal["a", "b", "c"]
    if origin is type(None):
        pass
    try:
        from typing import Literal
        if origin is Literal:
            vals = get_args(annotation)
            return " | ".join(f'"{v}"' for v in vals)
    except ImportError:
        pass

    # list[X]
    if origin is list:
        inner = get_args(annotation)
        if inner:
            return f"list[{_type_label(inner[0])}]"
        return "list"

    # BaseModel subclass
    from pydantic import BaseModel
    if isinstance(annotation, type) and issubclass(annotation, BaseModel):
        return annotation.__name__

    # Primitives
    if annotation is str:
        return "string"
    if annotation is int:
        return "integer"
    if annotation is float:
        return "number"
    if annotation is bool:
        return "boolean"

    return str(annotation)


def generate_schema_description(
    model: type,
    indent: int = 0,
    max_depth: int = 3,
    visited: set | None = None,
) -> str:
    """Walk a Pydantic BaseModel and render a human-readable schema description."""
    from pydantic import BaseModel

    if visited is None:
        visited = set()
    if indent > max_depth or model in visited:
        return ""

    visited.add(model)
    lines: list[str] = []
    prefix = "  " * indent

    # Class docstring as section header
    if model.__doc__:
        doc = model.__doc__.strip().split("\n")[0]  # first line only
        lines.append(f"{prefix}# {doc}")

    for name, field_info in model.model_fields.items():
        if name in SKIP_FIELDS:
            continue

        annotation = field_info.annotation
        label = _type_label(annotation)
        desc = field_info.description or ""

        lines.append(f"{prefix}{name}: {label}")
        if desc:
            lines.append(f"{prefix}  → {desc}")

        # Recurse into nested BaseModel
        inner_type = annotation
        origin = get_origin(annotation)

        # Unwrap Optional
        if origin is type(None) or str(origin) == "typing.Union":
            args = get_args(annotation)
            non_none = [a for a in args if a is not type(None)]
            if len(non_none) == 1:
                inner_type = non_none[0]
                origin = get_origin(inner_type)

        # Unwrap list[Model]
        if origin is list:
            args = get_args(inner_type)
            if args:
                inner_type = args[0]

        if isinstance(inner_type, type) and issubclass(inner_type, BaseModel) and inner_type not in visited:
            sub = generate_schema_description(inner_type, indent + 1, max_depth, visited)
            if sub:
                lines.append(sub)

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 2. System Prompt
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    schema_desc = generate_schema_description(CallAnalysis)
    return f"""You are a call center analytics expert analyzing Vacatia's AI voice agent performance.

## Background
Vacatia is a timeshare hospitality company. Their AI voice agent handles inbound owner
calls — payments, account lookups, sending portal/payment/RCI links, and answering
property questions. Calls the AI cannot handle are transferred to human concierge agents.

## Data You Are Analyzing
Each record below is a structured LLM analysis of one call transcript. The analysis
was produced by a separate evaluation model — it is NOT the raw conversation, but an
extracted summary with per-intent resolution, quality scores, impediment tracking, and
friction events.

<schema_definition>
{schema_desc}
</schema_definition>

## Important Field Semantics
- impediment and agent_issue are ORTHOGONAL: a call can have both, either, or neither
- human_requested: null = no human request; "initial" = before AI service; "after_service" = after AI began helping
- scope reflects the UNDERLYING customer need, not the transfer action itself
- Each call has a primary intent and an optional secondary intent, independently resolved
- transfer.queue_result: what happened in the queue (connected/unavailable/caller_abandoned)
- friction contains clarifications, corrections, and loops — each with turn numbers and notes

## Your Task
Answer the user's question based ONLY on the provided call data.
- Identify patterns and provide counts/percentages where relevant
- Cite 2-4 calls that best ILLUSTRATE your findings (by call ID prefix)
- Do NOT list every call — synthesize
- If using --filter, note that you're seeing a filtered subset, not all calls
- If the data doesn't fully answer the question, say so
- Structure your answer clearly with sections/bullets if appropriate"""


# ---------------------------------------------------------------------------
# 3. Analysis Loading & Filtering
# ---------------------------------------------------------------------------

SKIP_PREFIXES = ("batch_", "metrics_", "report_", "insights_", "nl_data_")


def load_v7_analyses(dir_path: Path) -> list[dict]:
    """Load v7-compatible analysis JSONs from a directory.

    Tries flat glob first; falls back to recursive for subdirectory layouts
    (e.g. analyses_v6_review/ with scope:outcome subdirs).
    """
    if not dir_path.exists():
        raise FileNotFoundError(f"Directory not found: {dir_path}")

    json_files = list(dir_path.glob("*.json"))

    # If no top-level JSONs, try recursive
    if not json_files:
        json_files = list(dir_path.glob("**/*.json"))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {dir_path}")

    analyses = []
    skipped = 0
    for f in json_files:
        if any(f.name.startswith(p) for p in SKIP_PREFIXES):
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError):
            skipped += 1
            continue

        if not isinstance(data, dict):
            continue

        # Must have nested resolution.primary to be a v7-style analysis
        res = data.get("resolution")
        if not isinstance(res, dict) or "primary" not in res:
            continue

        analyses.append(data)

    if skipped:
        print(f"Warning: skipped {skipped} unreadable files", file=sys.stderr)

    return analyses


def filter_analyses(analyses: list[dict], keyword: str) -> list[dict]:
    """Case-insensitive keyword search across all text fields of each analysis."""
    kw = keyword.lower()
    matches = []

    for a in analyses:
        if _text_match(a, kw):
            matches.append(a)

    return matches


def _text_match(data: Any, kw: str) -> bool:
    """Recursively check if keyword appears in any string value."""
    if isinstance(data, str):
        return kw in data.lower()
    if isinstance(data, dict):
        return any(_text_match(v, kw) for v in data.values())
    if isinstance(data, list):
        return any(_text_match(item, kw) for item in data)
    return False


# ---------------------------------------------------------------------------
# 4. Call Formatter
# ---------------------------------------------------------------------------

def format_call_for_prompt(a: dict) -> str:
    """Render one v7 analysis as structured markdown for the LLM prompt."""
    call_id = a.get("call_id", "unknown")[:8]
    duration = a.get("duration_seconds")
    turns = a.get("turns", "?")
    repeat = "yes" if a.get("repeat_caller") else "no"

    sent = a.get("sentiment", {})
    sent_str = f"{sent.get('start', '?')} → {sent.get('end', '?')}" if sent else "unknown"

    lines = [f"### Call {call_id}"]
    meta_parts = [f"Turns: {turns}", f"Repeat caller: {repeat}"]
    if duration is not None:
        meta_parts.insert(0, f"Duration: {int(duration)}s")
    lines.append(f"- {' | '.join(meta_parts)}")
    lines.append(f"- Sentiment: {sent_str}")

    # Primary intent
    res = a.get("resolution", {})
    primary = res.get("primary", {})
    lines.append("")
    lines.append(_format_intent(primary, "Primary Intent"))

    # Secondary intent
    secondary = res.get("secondary")
    if secondary:
        lines.append("")
        lines.append(_format_intent(secondary, "Secondary Intent"))

    # Steps
    steps = res.get("steps", [])
    if steps:
        lines.append(f"- Steps: {' → '.join(steps)}")

    # Actions
    actions = a.get("actions", [])
    if actions:
        action_strs = [f"{act.get('type', '?')}→{act.get('outcome', '?')}" for act in actions]
        lines.append(f"- Actions: {', '.join(action_strs)}")

    # Impediment
    imp = a.get("impediment")
    if imp:
        lines.append(f"- Impediment: {imp.get('type', '?')} — {imp.get('detail', '')}")
        pg = imp.get("policy_gap")
        if pg:
            lines.append(f"  Policy gap: {pg.get('category', '?')} — {pg.get('specific_gap', '')}")
    else:
        lines.append("- Impediment: none")

    # Agent issue
    ai = a.get("agent_issue")
    if ai:
        lines.append(f"- Agent Issue: {ai.get('detail', '')}")
    else:
        lines.append("- Agent Issue: none")

    # Friction summary
    friction = a.get("friction", {})
    clars = friction.get("clarifications", [])
    corrs = friction.get("corrections", [])
    loops = friction.get("loops", [])
    lines.append(f"- Friction: {len(clars)} clarifications, {len(corrs)} corrections, {len(loops)} loops")

    # Friction details (type + cause/note only, skip turn numbers)
    for c in clars:
        lines.append(f"  - Clarification: {c.get('type', '?')}/{c.get('cause', '?')} — {c.get('note', '')}")
    for c in corrs:
        lines.append(f"  - Correction: {c.get('severity', '?')} — {c.get('note', '')}")
    for lo in loops:
        lines.append(f"  - Loop: {lo.get('type', '?')} on {lo.get('subject', '?')} — {lo.get('note', '')}")

    # Insights
    ins = a.get("insights", {})
    if ins.get("summary"):
        lines.append(f"- Summary: {ins['summary']}")
    if ins.get("verbatim"):
        lines.append(f'- Verbatim: "{ins["verbatim"]}"')
    if ins.get("coaching"):
        lines.append(f"- Coaching: {ins['coaching']}")

    return "\n".join(lines)


def _format_intent(intent: dict, label: str) -> str:
    """Format a single IntentResolution block."""
    request = intent.get("request", "unknown")
    context = intent.get("context", "")
    scope = intent.get("scope", "?")
    outcome = intent.get("outcome", "?")
    detail = intent.get("detail", "")

    parts = [f"**{label}**: {request}"]
    if context:
        parts.append(f"  Context: {context}")

    # Scope + outcome line
    outcome_line = f"  Scope: {scope} | Outcome: {outcome}"
    if outcome == "fulfilled" and intent.get("resolution_confirmed") is not None:
        outcome_line += f" | Confirmed: {'yes' if intent['resolution_confirmed'] else 'no'}"
    parts.append(outcome_line)

    # Human requested
    hr = intent.get("human_requested")
    dept = intent.get("department_requested")
    if hr:
        hr_line = f"  Human requested: {hr}"
        if dept:
            hr_line += f" | Dept: {dept}"
        parts.append(hr_line)

    # Transfer
    transfer = intent.get("transfer")
    if transfer:
        t_line = f"  Transfer: {transfer.get('reason', '?')} → {transfer.get('destination', '?')}"
        if transfer.get("queue_detected"):
            qr = transfer.get("queue_result")
            t_line += f" (queue: {qr or 'detected'})"
        parts.append(t_line)

    # Abandon stage
    abn = intent.get("abandon_stage")
    if abn:
        parts.append(f"  Abandon stage: {abn}")

    if detail:
        parts.append(f"  Detail: {detail}")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
# 5. Prompt Builder
# ---------------------------------------------------------------------------

def build_prompt(question: str, sampled: list[dict], filter_kw: str | None) -> str:
    call_data = "\n\n".join(format_call_for_prompt(a) for a in sampled)

    header = f"## Question\n{question}\n"
    if filter_kw:
        header += f"\n*Note: Data pre-filtered by keyword \"{filter_kw}\"*\n"

    return f"""{header}
## Call Data ({len(sampled)} calls)

{call_data}

---

Answer the question based on the data above. Cite specific evidence."""


# ---------------------------------------------------------------------------
# 6. LLM Call
# ---------------------------------------------------------------------------

def get_genai_client() -> genai.Client:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


def call_llm(
    prompt: str,
    system_prompt: str,
    model: str,
    thinking_level: str | None = None,
    max_retries: int = 3,
) -> tuple[str, dict]:
    """Call Gemini and return (answer_text, usage_stats).

    Retries on 429 RESOURCE_EXHAUSTED with the delay suggested by the API.
    """
    client = get_genai_client()

    config_kwargs = dict(
        temperature=0.3,
        max_output_tokens=65536,
        system_instruction=system_prompt,
    )
    if thinking_level:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level=thinking_level.upper()
        )

    config = types.GenerateContentConfig(**config_kwargs)

    last_exc = None
    for attempt in range(1, max_retries + 1):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=config,
            )

            usage_stats = {}
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                meta = response.usage_metadata
                usage_stats = {
                    k: v for k, v in {
                        "input_tokens": getattr(meta, "prompt_token_count", None),
                        "output_tokens": getattr(meta, "candidates_token_count", None),
                        "thinking_tokens": getattr(meta, "thoughts_token_count", None),
                        "total_tokens": getattr(meta, "total_token_count", None),
                    }.items() if v is not None
                }

            return response.text, usage_stats

        except Exception as e:
            last_exc = e
            err_str = str(e)
            if "429" in err_str and "RESOURCE_EXHAUSTED" in err_str:
                # Extract retry delay from error message
                delay = 15.0  # default
                delay_match = re.search(r"retry in ([\d.]+)s", err_str, re.IGNORECASE)
                if delay_match:
                    delay = float(delay_match.group(1)) + 1.0  # add 1s buffer

                if attempt < max_retries:
                    print(
                        f"  Rate limited (attempt {attempt}/{max_retries}). "
                        f"Retrying in {delay:.0f}s...",
                        file=sys.stderr,
                    )
                    time.sleep(delay)
                    continue
            # Non-retryable error or max retries exceeded
            raise

    raise last_exc


# ---------------------------------------------------------------------------
# 7. Output Persistence
# ---------------------------------------------------------------------------

def prepare_output_dir(
    output_dir: Path,
    question: str,
    prompt: str,
    system_prompt: str,
    model: str,
    limit: int,
    filter_kw: str | None,
    total_available: int,
    filtered_count: int | None,
    sampled_count: int,
    sampled_ids: list[str],
) -> tuple[Path, datetime]:
    """Create output directory and save inputs BEFORE the LLM call."""
    now = datetime.now()
    folder = now.strftime("%b%d-%Hh%M")
    ask_dir = output_dir / folder
    ask_dir.mkdir(parents=True, exist_ok=True)

    (ask_dir / "question.txt").write_text(question, encoding="utf-8")
    (ask_dir / "prompt.txt").write_text(prompt, encoding="utf-8")
    (ask_dir / "system_prompt.txt").write_text(system_prompt, encoding="utf-8")

    metadata = {
        "timestamp": now.isoformat(),
        "question": question,
        "model": model,
        "limit": limit,
        "filter_keyword": filter_kw,
        "total_analyses_available": total_available,
        "filtered_count": filtered_count,
        "sampled_count": sampled_count,
        "status": "pending",
        "sampled_call_ids": sampled_ids,
    }
    (ask_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    return ask_dir, now


def finalize_output(
    ask_dir: Path,
    timestamp: datetime,
    answer: str,
    usage_stats: dict,
    question: str,
    model: str,
    limit: int,
    filter_kw: str | None,
    total_available: int,
    filtered_count: int | None,
    sampled_count: int,
    sampled_ids: list[str],
    status: str = "completed",
) -> None:
    """Save answer and finalize metadata AFTER the LLM call."""
    (ask_dir / "answer.md").write_text(answer or "(No answer generated)", encoding="utf-8")

    metadata = {
        "timestamp": timestamp.isoformat(),
        "question": question,
        "model": model,
        "limit": limit,
        "filter_keyword": filter_kw,
        "total_analyses_available": total_available,
        "filtered_count": filtered_count,
        "sampled_count": sampled_count,
        "status": status,
        "usage_stats": usage_stats,
        "sampled_call_ids": sampled_ids,
    }
    (ask_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


# ---------------------------------------------------------------------------
# 8. CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Q&A on v7 structured analyses",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 tools/ask_struct_v7.py "What are the main customer requests?" \\
    --analyses-dir tests/golden/analyses_v6_review/

  python3 tools/ask_struct_v7.py "Recap all RCI-related calls" \\
    --analyses-dir tests/golden/analyses_v6_review/ --filter RCI

  python3 tools/ask_struct_v7.py "Why do transfers fail?" \\
    --analyses-dir path/ --limit 50 --stats
""",
    )
    parser.add_argument("question", help="The question to answer")
    parser.add_argument(
        "--analyses-dir", type=Path, required=True,
        help="Directory containing v7 analysis JSONs",
    )
    parser.add_argument(
        "--filter", dest="filter_kw", default=None,
        help="Pre-filter analyses by keyword before sampling",
    )
    parser.add_argument(
        "--limit", type=int, default=100,
        help="Max calls to sample for context (default: 100)",
    )
    parser.add_argument(
        "--model", default="gemini-3-pro-preview",
        help="LLM model (default: gemini-3-pro-preview)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("asks_v7"),
        help="Output directory (default: asks_v7/)",
    )
    parser.add_argument(
        "--thinking", choices=["minimal", "low", "medium", "high"],
        help="Thinking level for Gemini 3 models",
    )
    parser.add_argument("--verbose", action="store_true", help="Show sampling info")
    parser.add_argument("--stats", action="store_true", help="Show token usage")

    args = parser.parse_args()

    # 1. Load analyses
    print(f"Loading analyses from {args.analyses_dir}...", file=sys.stderr)
    try:
        analyses = load_v7_analyses(args.analyses_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    print(f"Loaded {len(analyses)} analyses", file=sys.stderr)

    # 2. Filter (optional)
    filtered_count = None
    pool = analyses
    if args.filter_kw:
        pool = filter_analyses(analyses, args.filter_kw)
        filtered_count = len(pool)
        print(
            f"Filter \"{args.filter_kw}\": {filtered_count} of {len(analyses)} match",
            file=sys.stderr,
        )
        if not pool:
            print("No analyses match the filter. Try a different keyword.", file=sys.stderr)
            return 1

    # 3. Sample
    if len(pool) > args.limit:
        sampled = random.sample(pool, args.limit)
    else:
        sampled = pool
    sampled_ids = [a.get("call_id", "unknown") for a in sampled]
    print(f"Sampled {len(sampled)} calls for context", file=sys.stderr)

    # 4. Build prompts
    system_prompt = build_system_prompt()
    prompt = build_prompt(args.question, sampled, args.filter_kw)
    prompt_tokens_est = (len(prompt) + len(system_prompt)) // 4

    if args.verbose:
        print(f"\n--- Schema description ({len(system_prompt):,} chars) ---", file=sys.stderr)
        print(system_prompt[:500] + "...", file=sys.stderr)
        print("---\n", file=sys.stderr)

    # Warn if estimated tokens approach rate limit (1M/min for gemini-3-pro)
    RATE_LIMIT_TOKENS = 1_000_000
    if prompt_tokens_est > RATE_LIMIT_TOKENS:
        suggested_limit = int(args.limit * (RATE_LIMIT_TOKENS * 0.9) / prompt_tokens_est)
        print(
            f"\n⚠ Estimated ~{prompt_tokens_est:,} tokens exceeds Gemini Pro "
            f"rate limit ({RATE_LIMIT_TOKENS:,}/min).",
            file=sys.stderr,
        )
        print(
            f"  Suggestion: use --limit {suggested_limit} to stay under the limit.",
            file=sys.stderr,
        )
        print(
            f"  Proceeding anyway — will retry if rate-limited.\n",
            file=sys.stderr,
        )

    # 5. Save inputs before LLM call
    ask_dir, timestamp = prepare_output_dir(
        output_dir=args.output_dir,
        question=args.question,
        prompt=prompt,
        system_prompt=system_prompt,
        model=args.model,
        limit=args.limit,
        filter_kw=args.filter_kw,
        total_available=len(analyses),
        filtered_count=filtered_count,
        sampled_count=len(sampled),
        sampled_ids=sampled_ids,
    )
    print(f"Saved prompt to: {ask_dir}", file=sys.stderr)

    # 6. Call LLM
    print(
        f"Calling {args.model} (~{prompt_tokens_est:,} input tokens)...",
        file=sys.stderr,
    )
    try:
        answer, usage_stats = call_llm(
            prompt, system_prompt, args.model, args.thinking
        )
    except Exception as e:
        print(f"LLM error: {e}", file=sys.stderr)
        finalize_output(
            ask_dir, timestamp, f"Error: {e}", {},
            args.question, args.model, args.limit, args.filter_kw,
            len(analyses), filtered_count, len(sampled), sampled_ids,
            status="failed",
        )
        return 1

    # 7. Print answer
    print("", file=sys.stderr)
    print(answer)

    # 8. Finalize
    finalize_output(
        ask_dir, timestamp, answer, usage_stats,
        args.question, args.model, args.limit, args.filter_kw,
        len(analyses), filtered_count, len(sampled), sampled_ids,
    )
    print(f"\nSaved to: {ask_dir}", file=sys.stderr)

    # 9. Token stats
    if args.stats or args.verbose:
        print("\n---", file=sys.stderr)
        print("Token Usage:", file=sys.stderr)
        if usage_stats:
            for key in ("input_tokens", "thinking_tokens", "output_tokens", "total_tokens"):
                if key in usage_stats:
                    label = key.replace("_", " ").title()
                    print(f"  {label:20s} {usage_stats[key]:,}", file=sys.stderr)
        else:
            print("  (No usage data available)", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
