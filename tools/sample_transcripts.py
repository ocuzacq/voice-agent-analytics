#!/usr/bin/env python3
"""
Transcript Sampling Script for Vacatia AI Voice Agent Analytics (v4.3)

Randomly selects N transcripts stratified by file size (proxy for call complexity).
Larger files typically represent longer, more complex calls.

v4.3: Target-based augmentation when run has existing samples.
- If run already has samples, -n becomes a TARGET (not count to add)
- System calculates delta: target - existing
- Example: run has 100, -n 500 → adds 400 more
- Use --force-count to override and sample exactly N (clearing existing)

v4.2: Smart append mode for run augmentation.
- When --no-clear is used, automatically excludes already-sampled files
- Enables iterative run growth: start with 50, add 30 more, add 20 more...
- Manifest is updated (not replaced) in append mode

v4.1: Run-based isolation support.
- --run-dir argument for isolated execution
- Manifest written to run directory when specified
- Backwards compatible with legacy flat directory mode

v3.9.2: Supports both .txt and .json transcript formats.
- Prioritizes .json if both exist for the same call_id
- Clears both .txt and .json files when clearing existing samples
"""

import argparse
import csv
import os
import random
import shutil
import statistics
from pathlib import Path
from datetime import datetime

from run_utils import (
    add_run_arguments, resolve_run_from_args, get_run_paths,
    prompt_for_run, confirm_or_select_run, require_explicit_run_noninteractive
)


def get_transcript_files(transcript_dir: Path) -> list[tuple[Path, int]]:
    """Get all transcript files with their sizes.

    Supports both .txt and .json formats.
    Prioritizes .json if both exist for the same call_id (dedupes by stem).
    """
    files = []
    seen_stems = set()

    # First pass: prefer .json files
    for ext in ['.json', '.txt']:
        for f in transcript_dir.iterdir():
            if f.is_file() and f.suffix == ext and f.stem not in seen_stems:
                files.append((f, f.stat().st_size))
                seen_stems.add(f.stem)

    return files


def get_already_sampled_stems(output_dir: Path) -> set[str]:
    """Get stems of files already in the output directory.

    v4.2: Used in append mode to avoid re-sampling the same transcripts.
    Reads from manifest.csv if available, otherwise scans directory.
    """
    already_sampled = set()

    manifest_path = output_dir / "manifest.csv"
    if manifest_path.exists():
        # Read from manifest (more reliable)
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename", "")
                if filename:
                    stem = Path(filename).stem
                    already_sampled.add(stem)
    else:
        # Fall back to directory scan
        for ext in ['.json', '.txt']:
            for f in output_dir.glob(f"*{ext}"):
                if f.stem not in ('manifest', 'classification_checkpoint'):
                    already_sampled.add(f.stem)

    return already_sampled


def stratified_sample(
    files: list[tuple[Path, int]],
    n_total: int,
    n_large: int,
    n_small: int
) -> tuple[list[tuple[Path, int]], int]:
    """
    Stratified sampling by file size.

    Returns sampled files and the median size used for stratification.
    """
    if not files:
        raise ValueError("No transcript files found")

    sizes = [size for _, size in files]
    median_size = statistics.median(sizes)

    # Split into pools
    large_pool = [(f, s) for f, s in files if s >= median_size]
    small_pool = [(f, s) for f, s in files if s < median_size]

    # Adjust sample sizes if pools are smaller than requested
    actual_n_large = min(n_large, len(large_pool))
    actual_n_small = min(n_small, len(small_pool))

    # If we can't meet quotas, redistribute
    if actual_n_large < n_large and len(small_pool) > n_small:
        extra_small = min(n_large - actual_n_large, len(small_pool) - n_small)
        actual_n_small += extra_small
    elif actual_n_small < n_small and len(large_pool) > n_large:
        extra_large = min(n_small - actual_n_small, len(large_pool) - n_large)
        actual_n_large += extra_large

    # Random sampling
    sampled_large = random.sample(large_pool, actual_n_large) if large_pool else []
    sampled_small = random.sample(small_pool, actual_n_small) if small_pool else []

    return sampled_large + sampled_small, int(median_size)


def copy_samples(
    samples: list[tuple[Path, int]],
    output_dir: Path,
    median_size: int,
    clear_existing: bool = True
) -> list[dict]:
    """Copy sampled files to output directory and return manifest data.

    Args:
        samples: List of (file_path, size) tuples to copy
        output_dir: Destination directory
        median_size: Median file size for categorization
        clear_existing: If True, clear existing transcript files before copying (default: True)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing transcript files if requested (v3.2 run isolation)
    # v3.9.2: Clear both .txt and .json files
    if clear_existing:
        existing_files = list(output_dir.glob("*.txt")) + list(output_dir.glob("*.json"))
        # Exclude special files (manifest, checkpoint, etc.)
        existing_files = [f for f in existing_files if f.stem not in ('manifest', 'classification_checkpoint')]
        if existing_files:
            print(f"Clearing {len(existing_files)} existing transcript files...")
            for f in existing_files:
                f.unlink()

    manifest = []
    for file_path, size in samples:
        dest = output_dir / file_path.name
        shutil.copy2(file_path, dest)

        category = "large" if size >= median_size else "small"
        manifest.append({
            "filename": file_path.name,
            "size_bytes": size,
            "category": category,
            "original_path": str(file_path)
        })

    return manifest


def write_manifest(manifest: list[dict], output_dir: Path, append: bool = False) -> Path:
    """Write manifest CSV file.

    Args:
        manifest: New manifest entries to write
        output_dir: Output directory
        append: If True, append to existing manifest (v4.2)
    """
    manifest_path = output_dir / "manifest.csv"

    # v4.2: In append mode, merge with existing manifest
    if append and manifest_path.exists():
        existing = []
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            existing = list(reader)
        # Convert size_bytes back to int for sorting
        for row in existing:
            row["size_bytes"] = int(row["size_bytes"])
        manifest = existing + manifest

    # Sort by size descending for readability
    manifest.sort(key=lambda x: x["size_bytes"], reverse=True)

    with open(manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "size_bytes", "category", "original_path"])
        writer.writeheader()
        writer.writerows(manifest)

    return manifest_path


def main():
    parser = argparse.ArgumentParser(
        description="Sample transcripts stratified by file size"
    )
    parser.add_argument(
        "-n", "--total",
        type=int,
        default=20,
        help="Total number of transcripts to sample (default: 20)"
    )
    parser.add_argument(
        "--large",
        type=int,
        default=None,
        help="Number of large transcripts (default: 60%% of total)"
    )
    parser.add_argument(
        "--small",
        type=int,
        default=None,
        help="Number of small transcripts (default: 40%% of total)"
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=Path(__file__).parent.parent / "transcripts",
        help="Directory containing transcript files"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "sampled",
        help="Output directory for sampled files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="Don't clear existing sampled files (append mode)"
    )
    parser.add_argument(
        "--force-count",
        action="store_true",
        help="Force sampling exactly N files (ignore target-based augmentation)"
    )

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
        args.output_dir = paths["sampled_dir"]
        print(f"Using run: {run_id} ({run_dir})")

    # v4.3: Target-based augmentation - check for existing samples
    target = args.total
    existing_count = 0
    is_augmenting = False
    manifest_path = args.output_dir / "manifest.csv"

    if manifest_path.exists() and not args.force_count:
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            existing_count = sum(1 for _ in reader)

        if existing_count > 0:
            delta = target - existing_count
            if delta <= 0:
                print(f"\nRun already has {existing_count} transcripts (target: {target}). Nothing to add.")
                print("Use --force-count to clear and sample exactly N files.")
                return 0
            else:
                print(f"\nTarget: {target} transcripts")
                print(f"  Existing: {existing_count}")
                print(f"  To add: {delta}")
                args.total = delta  # Sample only the delta
                args.no_clear = True  # Append mode
                is_augmenting = True

    # Set defaults for large/small split
    n_large = args.large if args.large is not None else int(args.total * 0.6)
    n_small = args.small if args.small is not None else args.total - n_large

    # Validate
    if n_large + n_small != args.total:
        print(f"Warning: large ({n_large}) + small ({n_small}) != total ({args.total})")
        print(f"Adjusting total to {n_large + n_small}")

    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)

    # Get all transcript files
    print(f"Scanning transcript directory: {args.input_dir}")
    files = get_transcript_files(args.input_dir)
    print(f"Found {len(files)} transcript files")

    if not files:
        print("Error: No transcript files found")
        return 1

    # v4.2: In append mode, exclude already-sampled files
    append_mode = args.no_clear
    already_sampled = set()
    if append_mode:
        already_sampled = get_already_sampled_stems(args.output_dir)
        if already_sampled:
            original_count = len(files)
            files = [(f, s) for f, s in files if f.stem not in already_sampled]
            excluded = original_count - len(files)
            print(f"\nAppend mode: excluding {excluded} already-sampled transcripts")
            print(f"  Available for sampling: {len(files)}")

            if not files:
                print("Error: No new transcripts available to sample")
                print(f"  (all {original_count} transcripts already in run)")
                return 1

    # Compute statistics
    sizes = [s for _, s in files]
    print(f"\nFile size statistics:")
    print(f"  Min: {min(sizes):,} bytes")
    print(f"  Max: {max(sizes):,} bytes")
    print(f"  Mean: {statistics.mean(sizes):,.0f} bytes")
    print(f"  Median: {statistics.median(sizes):,.0f} bytes")

    # Sample
    print(f"\nSampling {n_large} large + {n_small} small = {n_large + n_small} total")
    samples, median_size = stratified_sample(files, args.total, n_large, n_small)

    # Copy to output
    clear_existing = not append_mode
    print(f"\nCopying {len(samples)} files to: {args.output_dir}")
    if clear_existing:
        print("(clearing existing files first - use --no-clear to append)")
    else:
        print("(append mode - keeping existing files)")
    manifest = copy_samples(samples, args.output_dir, median_size, clear_existing)

    # Write manifest (append mode merges with existing)
    manifest_path = write_manifest(manifest, args.output_dir, append=append_mode)
    print(f"Manifest written to: {manifest_path}")

    # Summary
    new_large = sum(1 for m in manifest if m["category"] == "large")
    new_small = sum(1 for m in manifest if m["category"] == "small")

    if append_mode and already_sampled:
        # Read updated manifest to get total counts
        with open(manifest_path, 'r') as f:
            reader = csv.DictReader(f)
            all_entries = list(reader)
        total_large = sum(1 for m in all_entries if m["category"] == "large")
        total_small = sum(1 for m in all_entries if m["category"] == "small")
        print(f"\nAugmentation summary:")
        print(f"  Added: {len(manifest)} new transcripts ({new_large} large, {new_small} small)")
        print(f"  Total in run: {len(all_entries)} transcripts ({total_large} large, {total_small} small)")
    else:
        print(f"\nSample summary:")
        print(f"  Large (>= {median_size:,} bytes): {new_large}")
        print(f"  Small (< {median_size:,} bytes): {new_small}")
        print(f"  Total: {len(manifest)}")

    return 0


if __name__ == "__main__":
    exit(main())
