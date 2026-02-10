#!/usr/bin/env python3
"""
Batch-analyze all golden transcripts with the v6.0 per-intent schema.

Saves each CallRecord JSON to tests/golden/analyses_v6_review/.
Prints a summary table at the end.

Usage:
    source .env
    python3 tools/batch_golden_v6.py
    python3 tools/batch_golden_v6.py --model gemini-3-flash-preview
"""

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

# Reuse the system prompt from poc_structured_full.py
from poc_structured_full import SYSTEM_PROMPT, _map_ended_reason


def analyze_one(client: genai.Client, transcript_path: Path, config: types.GenerateContentConfig, model_name: str) -> dict:
    """Analyze one transcript and return the result dict."""
    preprocessed = preprocess_transcript(transcript_path)
    call_id = preprocessed["call_id"]
    formatted = format_for_llm(preprocessed)

    t0 = time.time()
    response = client.models.generate_content(
        model=model_name,
        contents=f"Analyze this call transcript:\n\n{formatted}",
        config=config,
    )
    elapsed = time.time() - t0

    if hasattr(response, 'parsed') and response.parsed is not None:
        analysis: CallAnalysis = response.parsed
        record = CallRecord(
            **analysis.model_dump(),
            call_id=call_id,
            schema_version=SCHEMA_VERSION,
            duration_seconds=round(preprocessed.get("metadata", {}).get("duration", 0), 1) or None,
            ended_by=_map_ended_reason(preprocessed.get("metadata", {}).get("ended_reason")),
        )
        return {
            "success": True,
            "call_id": call_id,
            "record": record.model_dump(),
            "elapsed": elapsed,
            "primary_scope": analysis.resolution.primary.scope,
            "primary_outcome": analysis.resolution.primary.outcome,
            "primary_request": analysis.resolution.primary.request,
            "has_secondary": analysis.resolution.secondary is not None,
            "failure": analysis.failure.type if analysis.failure else None,
        }
    else:
        # Try manual parse
        raw = response.text if hasattr(response, 'text') else ""
        try:
            data = json.loads(raw)
            analysis = CallAnalysis.model_validate(data)
            record = CallRecord(
                **analysis.model_dump(),
                call_id=call_id,
                schema_version=SCHEMA_VERSION,
                duration_seconds=round(preprocessed.get("metadata", {}).get("duration", 0), 1) or None,
                ended_by=_map_ended_reason(preprocessed.get("metadata", {}).get("ended_reason")),
            )
            return {
                "success": True,
                "call_id": call_id,
                "record": record.model_dump(),
                "elapsed": elapsed,
                "primary_scope": analysis.resolution.primary.scope,
                "primary_outcome": analysis.resolution.primary.outcome,
                "primary_request": analysis.resolution.primary.request,
                "has_secondary": analysis.resolution.secondary is not None,
                "failure": analysis.failure.type if analysis.failure else None,
                "note": "manual_parse",
            }
        except Exception as e:
            return {
                "success": False,
                "call_id": call_id,
                "elapsed": elapsed,
                "error": str(e),
            }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Batch-analyze golden transcripts with v6.0 schema")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between API calls (seconds)")
    parser.add_argument("--output-dir", default="tests/golden/analyses_v6_review", help="Output directory for analyses")
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

    # Find all golden transcripts
    golden_dir = Path("tests/golden/transcripts")
    transcripts = sorted(golden_dir.glob("*.json"))
    print(f"Found {len(transcripts)} golden transcripts")
    print(f"Model: {args.model}")
    print(f"Schema: {SCHEMA_VERSION}")
    print("=" * 70)

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total_t0 = time.time()

    for i, tp in enumerate(transcripts):
        call_id = tp.stem
        print(f"\n[{i+1}/{len(transcripts)}] {call_id}...", end=" ", flush=True)

        try:
            result = analyze_one(client, tp, config, args.model)
        except Exception as e:
            result = {"success": False, "call_id": call_id, "elapsed": 0, "error": str(e)}

        results.append(result)

        if result["success"]:
            # Save the analysis JSON
            out_path = output_dir / f"{call_id}.json"
            with open(out_path, "w") as f:
                json.dump(result["record"], f, indent=2)

            scope = result["primary_scope"]
            outcome = result["primary_outcome"]
            secondary = " +secondary" if result["has_secondary"] else ""
            failure = f" FAIL:{result['failure']}" if result["failure"] else ""
            print(f"OK ({result['elapsed']:.1f}s) [{scope}:{outcome}]{secondary}{failure}")
        else:
            print(f"FAILED ({result.get('elapsed', 0):.1f}s): {result.get('error', 'unknown')}")

        # Rate limiting
        if i < len(transcripts) - 1:
            time.sleep(args.delay)

    total_elapsed = time.time() - total_t0

    # Summary
    print("\n" + "=" * 70)
    print(f"SUMMARY: {sum(1 for r in results if r['success'])}/{len(results)} succeeded in {total_elapsed:.0f}s")
    print(f"Output: {output_dir}/")
    print()

    # Breakdown table
    successes = [r for r in results if r["success"]]
    if successes:
        print(f"{'Call ID':<40} {'Scope':<14} {'Outcome':<13} {'2nd?':<5} {'Failure':<12} {'Time':>5}")
        print("-" * 95)
        for r in successes:
            print(f"{r['call_id']:<40} {r['primary_scope']:<14} {r['primary_outcome']:<13} "
                  f"{'yes' if r['has_secondary'] else 'no':<5} "
                  f"{r.get('failure') or '-':<12} {r['elapsed']:>4.1f}s")

    failures = [r for r in results if not r["success"]]
    if failures:
        print(f"\nFAILED ({len(failures)}):")
        for r in failures:
            print(f"  {r['call_id']}: {r.get('error', 'unknown')}")


if __name__ == "__main__":
    main()
