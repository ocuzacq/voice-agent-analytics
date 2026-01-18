#!/usr/bin/env python3
"""
Batch Transcript Analyzer for Vacatia AI Voice Agent Analytics (v2)

Analyzes multiple transcripts with rate limiting and progress tracking.
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from analyze_transcript import configure_genai, analyze_transcript, save_analysis


def get_transcripts_to_analyze(input_dir: Path, output_dir: Path, skip_existing: bool = True) -> list[Path]:
    """Get list of transcripts needing analysis."""
    transcripts = [f for f in input_dir.iterdir() if f.suffix == '.txt']

    if skip_existing and output_dir.exists():
        analyzed_ids = {f.stem for f in output_dir.iterdir() if f.suffix == '.json'}
        transcripts = [t for t in transcripts if t.stem not in analyzed_ids]

    return sorted(transcripts)


def batch_analyze(
    transcripts: list[Path],
    output_dir: Path,
    model_name: str,
    rate_limit: float = 1.0,
    max_retries: int = 3,
    stop_on_error: bool = False
) -> dict:
    """Analyze multiple transcripts."""
    results = {"total": len(transcripts), "successful": 0, "failed": 0, "errors": []}

    for i, transcript in enumerate(transcripts, 1):
        print(f"\n[{i}/{len(transcripts)}] {transcript.name}")

        for retry in range(max_retries):
            try:
                analysis = analyze_transcript(transcript, model_name)
                save_analysis(analysis, output_dir)
                results["successful"] += 1
                outcome = analysis.get("outcome", "?")
                effort = analysis.get("customer_effort", "?")
                print(f"  ✓ {outcome} | effort={effort}")
                break
            except Exception as e:
                if retry < max_retries - 1:
                    print(f"  ⚠ Retry {retry+1}: {e}")
                    time.sleep(rate_limit * 2)
                else:
                    print(f"  ✗ Failed: {e}")
                    results["failed"] += 1
                    results["errors"].append(f"{transcript.name}: {e}")
                    if stop_on_error:
                        return results

        if i < len(transcripts):
            time.sleep(rate_limit)

    return results


def main():
    parser = argparse.ArgumentParser(description="Batch analyze transcripts (v2)")
    parser.add_argument("-i", "--input-dir", type=Path,
                        default=Path(__file__).parent.parent.parent / "sampled",
                        help="Input directory")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path(__file__).parent.parent.parent / "analyses",
                        help="Output directory")
    parser.add_argument("--model", default="gemini-2.5-flash", help="Model name")
    parser.add_argument("--rate-limit", type=float, default=1.0, help="Delay between calls")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries")
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-analyze existing")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first error")
    parser.add_argument("--limit", type=int, help="Limit number to analyze")

    args = parser.parse_args()

    print("Configuring API...")
    try:
        configure_genai()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Scanning: {args.input_dir}")
    transcripts = get_transcripts_to_analyze(
        args.input_dir, args.output_dir, not args.no_skip_existing
    )

    if args.limit:
        transcripts = transcripts[:args.limit]

    if not transcripts:
        print("No transcripts to analyze")
        return 0

    print(f"Found {len(transcripts)} transcripts")

    start = datetime.now()
    results = batch_analyze(
        transcripts, args.output_dir, args.model,
        args.rate_limit, args.max_retries, args.stop_on_error
    )
    duration = (datetime.now() - start).total_seconds()

    print("\n" + "=" * 40)
    print("BATCH COMPLETE (v2)")
    print("=" * 40)
    print(f"Total: {results['total']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Duration: {duration:.1f}s")

    if results['errors']:
        print("\nErrors:")
        for err in results['errors']:
            print(f"  - {err}")

    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())
