#!/usr/bin/env python3
"""
Compare output quality between thinking levels (MEDIUM vs LOW).

Runs same transcript through both thinking levels and compares:
1. Duration
2. Output completeness
3. Output quality (field presence, detail level)
"""

import json
import os
import sys
import time
from pathlib import Path

from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_transcript import preprocess_transcript, format_for_llm
from analyze_transcript import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, ANALYSIS_SCHEMA, extract_json_from_response


def analyze_with_thinking_level(
    transcript_path: Path,
    thinking_level: str,
    client: genai.Client,
    model_name: str = "gemini-3-flash-preview"
) -> dict:
    """Analyze a transcript with specific thinking level."""

    preprocessed = preprocess_transcript(transcript_path)
    call_id = preprocessed["call_id"]
    formatted_transcript = format_for_llm(preprocessed)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        schema=ANALYSIS_SCHEMA,
        transcript=formatted_transcript
    )

    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=16384,
        thinking_config=types.ThinkingConfig(
            thinking_level=thinking_level
        ),
        system_instruction=SYSTEM_PROMPT,
    )

    start_time = time.time()

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=config,
    )

    duration = time.time() - start_time

    # Check for truncation
    truncated = False
    if response.candidates:
        finish_reason = response.candidates[0].finish_reason
        if finish_reason and "MAX_TOKENS" in str(finish_reason):
            truncated = True

    analysis = extract_json_from_response(response.text)
    analysis["call_id"] = call_id

    return {
        "thinking_level": thinking_level,
        "duration_seconds": round(duration, 2),
        "truncated": truncated,
        "output_bytes": len(json.dumps(analysis)),
        "analysis": analysis,
    }


def compare_outputs(medium_result: dict, low_result: dict) -> dict:
    """Compare outputs from two thinking levels."""

    medium = medium_result["analysis"]
    low = low_result["analysis"]

    # Check field presence
    key_fields = ["friction", "summary", "failure_description", "agent_miss_detail",
                  "customer_verbatim", "resolution_steps"]

    field_comparison = {}
    for field in key_fields:
        m_val = medium.get(field)
        l_val = low.get(field)

        m_present = m_val is not None and m_val != []
        l_present = l_val is not None and l_val != []

        if field == "friction":
            m_complete = m_present and all(k in m_val for k in ["turns", "clarifications", "corrections", "loops"])
            l_complete = l_present and all(k in l_val for k in ["turns", "clarifications", "corrections", "loops"])
            field_comparison[field] = {
                "medium": "complete" if m_complete else ("partial" if m_present else "missing"),
                "low": "complete" if l_complete else ("partial" if l_present else "missing"),
            }
        elif field in ["summary", "failure_description", "agent_miss_detail", "customer_verbatim"]:
            m_len = len(m_val) if m_val else 0
            l_len = len(l_val) if l_val else 0
            field_comparison[field] = {
                "medium": f"{m_len} chars" if m_present else "missing",
                "low": f"{l_len} chars" if l_present else "missing",
            }
        elif field == "resolution_steps":
            m_len = len(m_val) if m_val else 0
            l_len = len(l_val) if l_val else 0
            field_comparison[field] = {
                "medium": f"{m_len} steps" if m_present else "missing",
                "low": f"{l_len} steps" if l_present else "missing",
            }

    # Check outcome consistency
    outcome_match = medium.get("outcome") == low.get("outcome")

    return {
        "duration_diff": round(medium_result["duration_seconds"] - low_result["duration_seconds"], 2),
        "size_diff": medium_result["output_bytes"] - low_result["output_bytes"],
        "outcome_match": outcome_match,
        "field_comparison": field_comparison,
    }


def run_comparison(transcript_path: Path, model_name: str = "gemini-3-flash-preview"):
    """Run comparison on a single transcript."""

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")

    client = genai.Client(api_key=api_key)

    print(f"Analyzing: {transcript_path.name}")
    print("=" * 60)

    # Run with MEDIUM
    print("\n[MEDIUM thinking]...", end=" ", flush=True)
    medium_result = analyze_with_thinking_level(transcript_path, "MEDIUM", client, model_name)
    print(f"{medium_result['duration_seconds']}s, {medium_result['output_bytes']} bytes")

    # Run with LOW
    print("[LOW thinking]...", end=" ", flush=True)
    low_result = analyze_with_thinking_level(transcript_path, "LOW", client, model_name)
    print(f"{low_result['duration_seconds']}s, {low_result['output_bytes']} bytes")

    # Compare
    comparison = compare_outputs(medium_result, low_result)

    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    print(f"Duration diff: {comparison['duration_diff']}s (MEDIUM - LOW)")
    print(f"Size diff: {comparison['size_diff']} bytes (MEDIUM - LOW)")
    print(f"Outcome match: {'✓' if comparison['outcome_match'] else '✗'}")
    print(f"  MEDIUM: {medium_result['analysis'].get('outcome')}")
    print(f"  LOW: {low_result['analysis'].get('outcome')}")

    print("\nField comparison:")
    for field, values in comparison["field_comparison"].items():
        print(f"  {field}:")
        print(f"    MEDIUM: {values['medium']}")
        print(f"    LOW: {values['low']}")

    # Print summaries for comparison
    print("\n" + "=" * 60)
    print("SUMMARY COMPARISON")
    print("=" * 60)
    print(f"\n[MEDIUM] {medium_result['analysis'].get('summary', 'N/A')}")
    print(f"\n[LOW] {low_result['analysis'].get('summary', 'N/A')}")

    return {
        "transcript": transcript_path.name,
        "medium": medium_result,
        "low": low_result,
        "comparison": comparison,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare thinking levels")
    parser.add_argument("transcript", type=Path, help="Path to transcript file")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview")

    args = parser.parse_args()

    if not args.transcript.exists():
        print(f"Error: File not found: {args.transcript}", file=sys.stderr)
        sys.exit(1)

    run_comparison(args.transcript, args.model)
