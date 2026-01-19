#!/usr/bin/env python3
"""
Batch Transcript Analyzer for Vacatia AI Voice Agent Analytics (v3.2)

Analyzes multiple transcripts with configurable parallelization and rate limiting.
Produces v3 schema analyses with 18 fields including policy_gap_detail,
customer_verbatim, agent_miss_detail, and resolution_steps.

v3.2: Added parallel processing (default 3 workers) for faster batch analysis.
"""

import argparse
import csv
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent))

from analyze_transcript import configure_genai, analyze_transcript, save_analysis


# Thread-safe counter for progress tracking
class ProgressTracker:
    """Thread-safe progress tracker."""

    def __init__(self, total: int):
        self.total = total
        self.completed = 0
        self.successful = 0
        self.failed = 0
        self.errors: list[str] = []
        self.lock = threading.Lock()
        self.start_time = datetime.now()

    def record_success(self, transcript_name: str, outcome: str, effort: str):
        with self.lock:
            self.completed += 1
            self.successful += 1
            elapsed = (datetime.now() - self.start_time).total_seconds()
            rate = self.completed / elapsed if elapsed > 0 else 0
            eta = (self.total - self.completed) / rate if rate > 0 else 0
            print(f"[{self.completed}/{self.total}] ✓ {transcript_name}: {outcome} | effort={effort} ({rate:.1f}/s, ETA {eta:.0f}s)")

    def record_failure(self, transcript_name: str, error: str):
        with self.lock:
            self.completed += 1
            self.failed += 1
            self.errors.append(f"{transcript_name}: {error}")
            print(f"[{self.completed}/{self.total}] ✗ {transcript_name}: {error}")

    def get_results(self) -> dict:
        with self.lock:
            return {
                "total": self.total,
                "successful": self.successful,
                "failed": self.failed,
                "errors": self.errors.copy()
            }


def get_transcripts_from_manifest(input_dir: Path) -> list[Path] | None:
    """Read manifest.csv to get the scoped file list.

    Returns list of transcript paths if manifest exists, None otherwise.
    This enables v3.2 run isolation - only process files from current scope.
    """
    manifest_path = input_dir / "manifest.csv"
    if not manifest_path.exists():
        return None

    transcripts = []
    with open(manifest_path, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_path = input_dir / row['filename']
            if file_path.exists():
                transcripts.append(file_path)

    return transcripts if transcripts else None


def get_transcripts_to_analyze(input_dir: Path, output_dir: Path, skip_existing: bool = True) -> list[Path]:
    """Get list of transcripts needing analysis.

    v3.2: If manifest.csv exists, only process files listed in it (scope enforcement).
    Falls back to directory scan if no manifest found (backward compatible).
    """
    # Try manifest first (v3.2 scope enforcement)
    transcripts = get_transcripts_from_manifest(input_dir)
    manifest_used = transcripts is not None

    if transcripts is None:
        # Fall back to directory scan (backward compatible)
        transcripts = [f for f in input_dir.iterdir() if f.suffix == '.txt']

    if skip_existing and output_dir.exists():
        analyzed_ids = {f.stem for f in output_dir.iterdir() if f.suffix == '.json'}
        transcripts = [t for t in transcripts if t.stem not in analyzed_ids]

    return sorted(transcripts), manifest_used


def analyze_single(
    transcript: Path,
    output_dir: Path,
    model_name: str,
    tracker: ProgressTracker,
    rate_limit: float,
    max_retries: int
) -> bool:
    """Analyze a single transcript with retries. Thread-safe."""
    # v3.4.1: Extra retries for JSON parse errors (usually transient API issues)
    json_parse_retries = max_retries + 2

    for retry in range(json_parse_retries):
        try:
            analysis = analyze_transcript(transcript, model_name)
            save_analysis(analysis, output_dir)
            outcome = analysis.get("outcome", "?")
            effort = analysis.get("customer_effort", "?")
            tracker.record_success(transcript.name, outcome, str(effort))

            # Rate limiting per call
            if rate_limit > 0:
                time.sleep(rate_limit)

            return True
        except ValueError as e:
            # JSON parse errors are usually transient - use more aggressive backoff
            if "Could not parse JSON" in str(e) and retry < json_parse_retries - 1:
                wait_time = max(rate_limit, 2.0) * (2 ** (retry + 1))
                time.sleep(wait_time)
            elif retry < max_retries - 1:
                wait_time = rate_limit * (2 ** (retry + 1))
                time.sleep(wait_time)
            else:
                tracker.record_failure(transcript.name, str(e))
                return False
        except Exception as e:
            if retry < max_retries - 1:
                # Exponential backoff on retry
                wait_time = rate_limit * (2 ** (retry + 1))
                time.sleep(wait_time)
            else:
                tracker.record_failure(transcript.name, str(e))
                return False

    return False


def batch_analyze_parallel(
    transcripts: list[Path],
    output_dir: Path,
    model_name: str,
    workers: int = 3,
    rate_limit: float = 1.0,
    max_retries: int = 3,
    stop_on_error: bool = False
) -> dict:
    """Analyze multiple transcripts in parallel."""
    tracker = ProgressTracker(len(transcripts))

    print(f"\nStarting parallel analysis with {workers} workers...")
    print(f"Rate limit: {rate_limit}s per call, Max retries: {max_retries}")
    print()

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                analyze_single,
                transcript, output_dir, model_name, tracker, rate_limit, max_retries
            ): transcript
            for transcript in transcripts
        }

        for future in as_completed(futures):
            if stop_on_error:
                results = tracker.get_results()
                if results["failed"] > 0:
                    # Cancel remaining futures
                    for f in futures:
                        f.cancel()
                    break

    return tracker.get_results()


def batch_analyze_sequential(
    transcripts: list[Path],
    output_dir: Path,
    model_name: str,
    rate_limit: float = 1.0,
    max_retries: int = 3,
    stop_on_error: bool = False
) -> dict:
    """Analyze multiple transcripts sequentially (legacy mode)."""
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
    parser = argparse.ArgumentParser(
        description="Batch analyze transcripts (v3.2 - parallel processing)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Default: 3 parallel workers
    python3 tools/batch_analyze.py

    # Custom parallelization
    python3 tools/batch_analyze.py --workers 5

    # Sequential mode (v3.1 behavior)
    python3 tools/batch_analyze.py --workers 1

    # Aggressive parallelization (use with caution)
    python3 tools/batch_analyze.py --workers 10 --rate-limit 0.5
        """
    )
    parser.add_argument("-i", "--input-dir", type=Path,
                        default=Path(__file__).parent.parent / "sampled",
                        help="Input directory")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "analyses",
                        help="Output directory")
    parser.add_argument("--model", default="gemini-3-flash-preview", help="Model name (default: gemini-3-flash-preview)")
    parser.add_argument("-w", "--workers", type=int, default=3,
                        help="Number of parallel workers (default: 3, use 1 for sequential)")
    parser.add_argument("--rate-limit", type=float, default=1.0,
                        help="Delay between API calls per worker (seconds)")
    parser.add_argument("--max-retries", type=int, default=3, help="Max retries per transcript")
    parser.add_argument("--no-skip-existing", action="store_true", help="Re-analyze existing")
    parser.add_argument("--stop-on-error", action="store_true", help="Stop on first error")
    parser.add_argument("--limit", type=int, help="Limit number to analyze")

    args = parser.parse_args()

    print("=" * 50)
    print("BATCH ANALYZER v3.2 (Parallel Processing)")
    print("=" * 50)

    print("\nConfiguring API...")
    try:
        configure_genai()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Scanning: {args.input_dir}")
    transcripts, manifest_used = get_transcripts_to_analyze(
        args.input_dir, args.output_dir, not args.no_skip_existing
    )

    if manifest_used:
        print("✓ Using manifest.csv for scope (v3.2 run isolation)")
    else:
        print("⚠ No manifest.csv found - using all .txt files in directory")

    if args.limit:
        transcripts = transcripts[:args.limit]

    if not transcripts:
        print("No transcripts to analyze")
        return 0

    print(f"Found {len(transcripts)} transcripts to analyze")
    print(f"Workers: {args.workers}, Rate limit: {args.rate_limit}s")

    start = datetime.now()

    if args.workers == 1:
        print("\nRunning in sequential mode...")
        results = batch_analyze_sequential(
            transcripts, args.output_dir, args.model,
            args.rate_limit, args.max_retries, args.stop_on_error
        )
    else:
        results = batch_analyze_parallel(
            transcripts, args.output_dir, args.model,
            args.workers, args.rate_limit, args.max_retries, args.stop_on_error
        )

    duration = (datetime.now() - start).total_seconds()
    rate = results['successful'] / duration if duration > 0 else 0

    print("\n" + "=" * 50)
    print("BATCH COMPLETE (v3.2)")
    print("=" * 50)
    print(f"Total: {results['total']}")
    print(f"Successful: {results['successful']}")
    print(f"Failed: {results['failed']}")
    print(f"Duration: {duration:.1f}s")
    print(f"Throughput: {rate:.2f} transcripts/sec")

    if results['errors']:
        print("\nErrors:")
        for err in results['errors'][:10]:  # Limit error output
            print(f"  - {err}")
        if len(results['errors']) > 10:
            print(f"  ... and {len(results['errors']) - 10} more errors")

    return 0 if results['failed'] == 0 else 1


if __name__ == "__main__":
    exit(main())
