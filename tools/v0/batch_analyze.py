#!/usr/bin/env python3
"""
Batch Transcript Analyzer for Vacatia AI Voice Agent Analytics (v0)

Analyzes multiple transcripts from a directory, with rate limiting
and progress tracking.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

# Import the single-file analyzer
from analyze_transcript import (
    configure_genai,
    analyze_transcript,
    save_analysis
)


def get_transcripts_to_analyze(
    input_dir: Path,
    output_dir: Path,
    skip_existing: bool = True
) -> list[Path]:
    """Get list of transcripts that need analysis."""
    # Get all transcript files
    transcripts = [f for f in input_dir.iterdir() if f.suffix == '.txt']

    if skip_existing:
        # Get already analyzed call IDs
        analyzed_ids = set()
        if output_dir.exists():
            for f in output_dir.iterdir():
                if f.suffix == '.json':
                    analyzed_ids.add(f.stem)

        # Filter out already analyzed
        transcripts = [t for t in transcripts if t.stem not in analyzed_ids]

    return sorted(transcripts)


def batch_analyze(
    transcripts: list[Path],
    output_dir: Path,
    model_name: str,
    rate_limit_delay: float = 1.0,
    max_retries: int = 3,
    stop_on_error: bool = False
) -> dict:
    """Analyze multiple transcripts with progress tracking."""
    results = {
        "total": len(transcripts),
        "successful": 0,
        "failed": 0,
        "errors": []
    }

    for i, transcript in enumerate(transcripts, 1):
        print(f"\n[{i}/{len(transcripts)}] Analyzing: {transcript.name}")

        retries = 0
        success = False

        while retries < max_retries and not success:
            try:
                analysis = analyze_transcript(transcript, model_name)
                save_analysis(analysis, output_dir)
                results["successful"] += 1
                success = True
                print(f"  ✓ Success: {analysis.get('coverage', 'N/A')} / {analysis.get('outcome', 'N/A')}")

            except Exception as e:
                retries += 1
                error_msg = f"{transcript.name}: {str(e)}"

                if retries < max_retries:
                    print(f"  ⚠ Retry {retries}/{max_retries}: {e}")
                    time.sleep(rate_limit_delay * 2)  # Extra delay on retry
                else:
                    print(f"  ✗ Failed: {e}")
                    results["failed"] += 1
                    results["errors"].append(error_msg)

                    if stop_on_error:
                        print("\nStopping due to error (--stop-on-error)")
                        return results

        # Rate limiting
        if i < len(transcripts):
            time.sleep(rate_limit_delay)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Batch analyze multiple call transcripts (v0 - simple schema)"
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "sampled",
        help="Directory containing transcript files to analyze"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "analyses",
        help="Output directory for analysis JSON files"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--rate-limit",
        type=float,
        default=1.0,
        help="Delay between API calls in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="Maximum retries per transcript (default: 3)"
    )
    parser.add_argument(
        "--no-skip-existing",
        action="store_true",
        help="Re-analyze transcripts that already have analysis files"
    )
    parser.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Stop batch processing on first error"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of transcripts to analyze"
    )

    args = parser.parse_args()

    # Configure API
    print("Configuring Google AI API...")
    try:
        configure_genai()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Get transcripts to analyze
    print(f"Scanning: {args.input_dir}")
    transcripts = get_transcripts_to_analyze(
        args.input_dir,
        args.output_dir,
        skip_existing=not args.no_skip_existing
    )

    if args.limit:
        transcripts = transcripts[:args.limit]

    if not transcripts:
        print("No transcripts to analyze (all may already be processed)")
        return 0

    print(f"Found {len(transcripts)} transcripts to analyze")
    print(f"Output directory: {args.output_dir}")
    print(f"Model: {args.model}")
    print(f"Rate limit: {args.rate_limit}s between calls")

    # Run batch analysis
    start_time = datetime.now()
    results = batch_analyze(
        transcripts,
        args.output_dir,
        args.model,
        args.rate_limit,
        args.max_retries,
        args.stop_on_error
    )
    end_time = datetime.now()

    # Summary
    duration = (end_time - start_time).total_seconds()
    print("\n" + "=" * 50)
    print("BATCH ANALYSIS COMPLETE (v0)")
    print("=" * 50)
    print(f"Total: {results['total']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Duration: {duration:.1f}s")

    if results['errors']:
        print("\nErrors:")
        for error in results['errors']:
            print(f"  - {error}")

    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())
