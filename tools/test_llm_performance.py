#!/usr/bin/env python3
"""
LLM Performance Test: Duration and Truncation Behavior

Tests analyze_transcript.py on N transcripts to measure:
1. Request duration (seconds)
2. Truncation rate (MAX_TOKENS finish reason)
3. Output completeness (friction field present)
4. Token usage if available

Usage:
    python3 tools/test_llm_performance.py -n 10
    python3 tools/test_llm_performance.py -n 10 --model gemini-3-flash-preview
"""

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))

from analyze_transcript import configure_genai, analyze_transcript


def get_sample_transcripts(transcripts_dir: Path, n: int, seed: int = None) -> list[Path]:
    """Get N random transcript files."""
    all_transcripts = list(transcripts_dir.glob("*.txt"))
    if seed:
        random.seed(seed)
    return random.sample(all_transcripts, min(n, len(all_transcripts)))


def test_single_transcript(transcript_path: Path, model_name: str) -> dict:
    """Analyze a single transcript and capture performance metrics."""
    result = {
        "call_id": transcript_path.stem,
        "transcript_bytes": transcript_path.stat().st_size,
        "duration_seconds": None,
        "truncated": None,
        "has_friction": None,
        "output_bytes": None,
        "finish_reason": None,
        "error": None,
    }

    start_time = time.time()

    try:
        # Run analysis
        analysis = analyze_transcript(transcript_path, model_name)

        result["duration_seconds"] = round(time.time() - start_time, 2)
        result["has_friction"] = "friction" in analysis
        result["output_bytes"] = len(json.dumps(analysis))

        # Check if friction is complete (has all expected fields)
        if "friction" in analysis:
            friction = analysis["friction"]
            result["friction_complete"] = all(
                k in friction for k in ["turns", "derailed_at", "clarifications", "corrections", "loops"]
            )
        else:
            result["friction_complete"] = False
            result["truncated"] = True

    except Exception as e:
        result["duration_seconds"] = round(time.time() - start_time, 2)
        result["error"] = str(e)

    return result


def run_performance_test(
    transcripts_dir: Path,
    n: int,
    model_name: str,
    seed: int = None
) -> dict:
    """Run performance test on N transcripts."""

    print(f"=" * 60)
    print(f"LLM Performance Test: {model_name}")
    print(f"=" * 60)
    print(f"Transcripts: {n}")
    print(f"Seed: {seed or 'random'}")
    print()

    # Get sample transcripts
    transcripts = get_sample_transcripts(transcripts_dir, n, seed)

    results = []
    truncated_count = 0
    error_count = 0
    total_duration = 0

    for i, transcript in enumerate(transcripts, 1):
        print(f"[{i}/{n}] {transcript.name}...", end=" ", flush=True)

        result = test_single_transcript(transcript, model_name)
        results.append(result)

        # Report result
        status_parts = []
        if result["error"]:
            status_parts.append(f"ERROR: {result['error'][:50]}")
            error_count += 1
        else:
            status_parts.append(f"{result['duration_seconds']}s")
            status_parts.append(f"{result['output_bytes']} bytes")

            if result["truncated"]:
                status_parts.append("⚠️ TRUNCATED")
                truncated_count += 1
            elif result["has_friction"]:
                status_parts.append("✓ complete")

            total_duration += result["duration_seconds"]

        print(" | ".join(status_parts))

    # Summary
    successful = n - error_count
    avg_duration = total_duration / successful if successful > 0 else 0

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total transcripts: {n}")
    print(f"Successful: {successful}")
    print(f"Errors: {error_count}")
    print(f"Truncated: {truncated_count} ({truncated_count/n*100:.1f}%)")
    print(f"Complete: {successful - truncated_count} ({(successful - truncated_count)/n*100:.1f}%)")
    print()
    print(f"Avg duration: {avg_duration:.2f}s")
    print(f"Total duration: {total_duration:.2f}s")

    if results:
        durations = [r["duration_seconds"] for r in results if r["duration_seconds"] and not r["error"]]
        if durations:
            print(f"Min duration: {min(durations):.2f}s")
            print(f"Max duration: {max(durations):.2f}s")

    # Truncation details
    if truncated_count > 0:
        print()
        print("TRUNCATED CALLS:")
        for r in results:
            if r.get("truncated"):
                print(f"  - {r['call_id']} ({r['transcript_bytes']} bytes input)")

    return {
        "model": model_name,
        "n": n,
        "successful": successful,
        "errors": error_count,
        "truncated": truncated_count,
        "truncation_rate": truncated_count / n if n > 0 else 0,
        "avg_duration_seconds": avg_duration,
        "total_duration_seconds": total_duration,
        "results": results,
    }


def main():
    parser = argparse.ArgumentParser(description="Test LLM performance on transcript analysis")
    parser.add_argument("-n", type=int, default=10, help="Number of transcripts to test (default: 10)")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview",
                        help="Model to use (default: gemini-3-flash-preview)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    parser.add_argument("--transcripts-dir", type=Path,
                        default=Path(__file__).parent.parent / "transcripts",
                        help="Directory containing transcripts")
    parser.add_argument("--output", type=Path, help="Save results to JSON file")

    args = parser.parse_args()

    if not args.transcripts_dir.exists():
        print(f"Error: Transcripts directory not found: {args.transcripts_dir}", file=sys.stderr)
        return 1

    try:
        configure_genai()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    results = run_performance_test(
        args.transcripts_dir,
        args.n,
        args.model,
        args.seed
    )

    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {args.output}")

    # Exit with error if truncation occurred
    return 1 if results["truncated"] > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
