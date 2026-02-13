#!/usr/bin/env python3
"""
Stability test: run volatile transcripts N times to measure prompt consistency.

For each transcript, runs poc_structured_full analysis N times and collects
key classification fields. Outputs a table showing flip frequency.

Usage:
    source .env
    python3 tools/stability_test.py                    # 3 volatile transcripts, 5 reps
    python3 tools/stability_test.py -n 3               # 3 reps (faster)
    python3 tools/stability_test.py --transcripts path1 path2  # custom transcripts
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_transcript import preprocess_transcript, format_for_llm
from schema import CallAnalysis, CallRecord, SCHEMA_VERSION
from poc_structured_full import SYSTEM_PROMPT, _map_ended_reason

# Default volatile transcripts identified in v2→v3 comparison
VOLATILE_TRANSCRIPTS = [
    "tests/golden/transcripts/042dfaf7-8265-4bdd-a957-1cbec199b173.json",
    "tests/golden/transcripts/0a5afc8b-170e-4d53-8cb9-414afa914328.json",
    "tests/golden/transcripts/0acca17b-d9c5-476c-965a-ac8697b4a21d.json",
]


def extract_key_fields(analysis: CallAnalysis) -> dict:
    """Extract the fields most likely to flip between runs."""
    p = analysis.resolution.primary
    s = analysis.resolution.secondary
    return {
        "scope": p.scope,
        "outcome": p.outcome,
        "request_category": p.request_category,
        "human_requested": p.human_requested,
        "dept_requested": p.department_requested,
        "abandon_stage": p.abandon_stage,
        "transfer_reason": p.transfer.reason if p.transfer else None,
        "transfer_dest": p.transfer.destination if p.transfer else None,
        "has_secondary": s is not None,
        "secondary_request": s.request if s else None,
        "impediment_type": analysis.impediment.type if analysis.impediment else None,
        "has_agent_issue": analysis.agent_issue is not None,
    }


def run_one(client: genai.Client, transcript_path: Path, config: types.GenerateContentConfig, model: str) -> dict:
    """Run a single analysis and return key fields."""
    preprocessed = preprocess_transcript(transcript_path)
    formatted = format_for_llm(preprocessed)

    t0 = time.time()
    response = client.models.generate_content(
        model=model,
        contents=f"Analyze this call transcript:\n\n{formatted}",
        config=config,
    )
    elapsed = time.time() - t0

    if hasattr(response, 'parsed') and response.parsed is not None:
        fields = extract_key_fields(response.parsed)
        fields["elapsed"] = round(elapsed, 1)
        fields["status"] = "ok"
        return fields

    # Fallback: manual parse
    raw = response.text if hasattr(response, 'text') else ""
    try:
        data = json.loads(raw)
        analysis = CallAnalysis.model_validate(data)
        fields = extract_key_fields(analysis)
        fields["elapsed"] = round(elapsed, 1)
        fields["status"] = "ok (manual)"
        return fields
    except Exception as e:
        return {"status": f"error: {e}", "elapsed": round(elapsed, 1)}


def print_results(call_id: str, runs: list[dict]) -> dict:
    """Print results for one transcript and return stability stats."""
    print(f"\n{'=' * 80}")
    print(f"  {call_id}")
    print(f"{'=' * 80}")

    # Header
    cols = ["#", "scope", "outcome", "req_category", "human_req", "dept_req", "abandon_stg", "xfer_reason", "secondary", "impediment", "agent_iss", "time"]
    print(f"  {'  '.join(f'{c:<14}' for c in cols)}")
    print(f"  {'-' * 162}")

    for i, r in enumerate(runs):
        if r.get("status", "").startswith("error"):
            print(f"  {i+1:<14}  ERROR: {r['status']}")
            continue
        print(f"  {i+1:<14}  {r.get('scope', '?'):<14}  {r.get('outcome', '?'):<14}"
              f"  {str(r.get('request_category') or '-'):<14}"
              f"  {str(r.get('human_requested') or '-'):<14}  {str(r.get('dept_requested') or '-'):<14}"
              f"  {str(r.get('abandon_stage') or '-'):<14}"
              f"  {str(r.get('transfer_reason') or '-'):<14}  "
              f"{'yes: ' + (r.get('secondary_request') or '?') if r.get('has_secondary') else 'no':<14}"
              f"  {str(r.get('impediment_type') or '-'):<14}  {'yes' if r.get('has_agent_issue') else '-':<14}"
              f"  {r.get('elapsed', '?')}s")

    # Compute flip counts
    ok_runs = [r for r in runs if r.get("status", "").startswith("ok")]
    if len(ok_runs) < 2:
        print(f"\n  Too few successful runs ({len(ok_runs)}) to assess stability")
        return {"stable": None, "flips": {}}

    fields_to_check = ["scope", "outcome", "request_category", "human_requested", "dept_requested", "abandon_stage", "transfer_reason", "has_secondary", "impediment_type", "has_agent_issue"]
    flips = {}
    for field in fields_to_check:
        values = [r.get(field) for r in ok_runs]
        unique = set(str(v) for v in values)
        if len(unique) > 1:
            flips[field] = list(unique)

    if flips:
        print(f"\n  UNSTABLE — flips detected on {len(flips)} field(s):")
        for field, vals in flips.items():
            print(f"    {field}: {' vs '.join(vals)}")
    else:
        print(f"\n  STABLE — all {len(ok_runs)} runs agree")

    return {"stable": len(flips) == 0, "flips": flips}


def main():
    parser = argparse.ArgumentParser(description="Stability test for volatile transcripts")
    parser.add_argument("-n", "--reps", type=int, default=5, help="Number of repetitions per transcript")
    parser.add_argument("--transcripts", nargs="+", help="Override default volatile transcript paths")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--delay", type=float, default=1.5, help="Delay between API calls (seconds)")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GOOGLE_API_KEY or GEMINI_API_KEY", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)
    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=16384,
        system_instruction=SYSTEM_PROMPT,
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        response_mime_type="application/json",
        response_schema=CallAnalysis,
    )

    transcript_paths = [Path(p) for p in (args.transcripts or VOLATILE_TRANSCRIPTS)]
    for tp in transcript_paths:
        if not tp.exists():
            print(f"Error: {tp} not found", file=sys.stderr)
            sys.exit(1)

    total_calls = len(transcript_paths) * args.reps
    print(f"Stability test: {len(transcript_paths)} transcripts × {args.reps} reps = {total_calls} API calls")
    print(f"Model: {args.model}  |  Temp: 0.2  |  Delay: {args.delay}s")

    all_stats = {}
    total_t0 = time.time()

    for tp in transcript_paths:
        call_id = tp.stem
        runs = []
        for i in range(args.reps):
            print(f"  [{call_id[:8]}] rep {i+1}/{args.reps}...", end=" ", flush=True)
            result = run_one(client, tp, config, args.model)
            runs.append(result)
            print(f"{result.get('outcome', 'ERR')} ({result.get('elapsed', '?')}s)")
            if i < args.reps - 1:
                time.sleep(args.delay)

        stats = print_results(call_id, runs)
        all_stats[call_id] = stats

        # Pause between transcripts
        time.sleep(args.delay)

    # Final summary
    total_elapsed = time.time() - total_t0
    print(f"\n{'=' * 80}")
    print(f"  FINAL SUMMARY  ({total_elapsed:.0f}s total)")
    print(f"{'=' * 80}")

    stable_count = sum(1 for s in all_stats.values() if s.get("stable") is True)
    unstable_count = sum(1 for s in all_stats.values() if s.get("stable") is False)
    unknown_count = sum(1 for s in all_stats.values() if s.get("stable") is None)

    for call_id, stats in all_stats.items():
        status = "STABLE" if stats.get("stable") else "UNSTABLE" if stats.get("stable") is False else "UNKNOWN"
        flip_info = ""
        if stats.get("flips"):
            flip_info = f" — flips: {', '.join(stats['flips'].keys())}"
        print(f"  {call_id[:8]}  {status}{flip_info}")

    print(f"\n  {stable_count} stable  |  {unstable_count} unstable  |  {unknown_count} unknown")


if __name__ == "__main__":
    main()
