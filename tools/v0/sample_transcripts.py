#!/usr/bin/env python3
"""
Transcript Sampling Script for Vacatia AI Voice Agent Analytics (v0)

Randomly selects N transcripts stratified by file size (proxy for call complexity).
Larger files typically represent longer, more complex calls.
"""

import argparse
import csv
import os
import random
import shutil
import statistics
from pathlib import Path
from datetime import datetime


def get_transcript_files(transcript_dir: Path) -> list[tuple[Path, int]]:
    """Get all transcript files with their sizes."""
    files = []
    for f in transcript_dir.iterdir():
        if f.is_file() and f.suffix == '.txt':
            files.append((f, f.stat().st_size))
    return files


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
    median_size: int
) -> list[dict]:
    """Copy sampled files to output directory and return manifest data."""
    output_dir.mkdir(parents=True, exist_ok=True)

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


def write_manifest(manifest: list[dict], output_dir: Path) -> Path:
    """Write manifest CSV file."""
    manifest_path = output_dir / "manifest.csv"

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
        default=Path(__file__).parent.parent.parent / "transcripts",
        help="Directory containing transcript files"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "sampled",
        help="Output directory for sampled files"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility"
    )

    args = parser.parse_args()

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
    print(f"\nCopying {len(samples)} files to: {args.output_dir}")
    manifest = copy_samples(samples, args.output_dir, median_size)

    # Write manifest
    manifest_path = write_manifest(manifest, args.output_dir)
    print(f"Manifest written to: {manifest_path}")

    # Summary
    large_count = sum(1 for m in manifest if m["category"] == "large")
    small_count = sum(1 for m in manifest if m["category"] == "small")
    print(f"\nSample summary:")
    print(f"  Large (>= {median_size:,} bytes): {large_count}")
    print(f"  Small (< {median_size:,} bytes): {small_count}")
    print(f"  Total: {len(manifest)}")

    return 0


if __name__ == "__main__":
    exit(main())
