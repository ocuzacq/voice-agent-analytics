#!/usr/bin/env python3
"""
Launch two parallel batch_analyze_v7.py processes, each with its own
partition of remaining transcripts and 15 workers.

Usage:
    source .env
    python3 tools/run_parallel_batch.py \
        --input-dir transcripts/transcripts-feb10 \
        --output-dir runs/v7_batch
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Parallel dual-process batch analyzer")
    parser.add_argument("--input-dir", required=True, help="Directory of transcript JSONs")
    parser.add_argument("--output-dir", required=True, help="Output directory (shared by both processes)")
    parser.add_argument("--workers-per-process", type=int, default=15, help="Workers per process (default: 15)")
    parser.add_argument("--processes", type=int, default=2, help="Number of parallel processes (default: 2)")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--delay", type=float, default=0.2, help="Delay between API calls (seconds)")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per transcript on errors (default: 5)")
    parser.add_argument("--limit", type=int, default=0, help="Max total transcripts to process (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Show plan without executing")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Discover all transcripts
    all_transcripts = sorted(input_dir.glob("*.json"))
    print(f"Total transcripts in {input_dir}: {len(all_transcripts)}")

    # Filter out already-analyzed
    existing = {p.stem for p in output_dir.glob("*.json")} if output_dir.exists() else set()
    remaining = [t for t in all_transcripts if t.stem not in existing]
    print(f"Already analyzed: {len(existing)}")
    print(f"Remaining: {len(remaining)}")

    if not remaining:
        print("Nothing to do — all transcripts already analyzed.")
        return

    # Apply limit
    if args.limit > 0 and len(remaining) > args.limit:
        remaining = remaining[:args.limit]
        print(f"Limited to: {args.limit} transcripts")

    # Partition into N roughly equal lists
    n_procs = args.processes
    partitions = [[] for _ in range(n_procs)]
    for i, t in enumerate(remaining):
        partitions[i % n_procs].append(t)

    # Write partition files
    list_files = []
    for i, partition in enumerate(partitions):
        list_file = Path(f"/tmp/batch_partition_{i}.txt")
        with open(list_file, "w") as f:
            for t in partition:
                f.write(f"{t}\n")
        list_files.append(list_file)
        print(f"  Partition {i}: {len(partition)} transcripts → {list_file}")

    print(f"\nPlan: {n_procs} processes × {args.workers_per_process} workers = {n_procs * args.workers_per_process} concurrent API calls")
    print(f"Model: {args.model}  Delay: {args.delay}s")
    print(f"Output: {output_dir}/")

    if args.dry_run:
        print("\n[DRY RUN] Would launch:")
        for i, lf in enumerate(list_files):
            print(f"  Process {i}: python3 tools/batch_analyze_v7.py --transcript-list {lf} --output-dir {output_dir} --workers {args.workers_per_process}")
        return

    print(f"\nLaunching {n_procs} processes...")
    print("=" * 70)

    t0 = time.time()
    processes = []
    for i, lf in enumerate(list_files):
        cmd = [
            sys.executable, "tools/batch_analyze_v7.py",
            "--transcript-list", str(lf),
            "--output-dir", str(output_dir),
            "--workers", str(args.workers_per_process),
            "--delay", str(args.delay),
            "--max-retries", str(args.max_retries),
            "--model", args.model,
        ]
        log_file = open(f"/tmp/batch_process_{i}.log", "w")
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        processes.append((proc, log_file, lf, len(partitions[i])))
        print(f"  Process {i} started (PID {proc.pid}): {len(partitions[i])} transcripts")

    # Monitor progress
    print(f"\nMonitoring progress (check /tmp/batch_process_*.log for details)...")
    print(f"Output files accumulate in: {output_dir}/\n")

    last_count = len(existing)
    while any(p.poll() is None for p, _, _, _ in processes):
        time.sleep(10)
        current = len(list(output_dir.glob("*.json")))
        new = current - len(existing)
        total_remaining = len(remaining)
        elapsed = time.time() - t0
        rate = new / elapsed * 60 if elapsed > 0 else 0
        pct = new / total_remaining * 100 if total_remaining > 0 else 100
        eta_min = (total_remaining - new) / (rate) if rate > 0 else 0

        alive = sum(1 for p, _, _, _ in processes if p.poll() is None)
        print(f"  [{new}/{total_remaining}] {pct:.0f}%  |  {rate:.0f}/min  |  ETA {eta_min:.0f}min  |  {alive} procs alive  |  {elapsed:.0f}s elapsed")

    elapsed = time.time() - t0
    final_count = len(list(output_dir.glob("*.json")))
    new_total = final_count - len(existing)

    print("\n" + "=" * 70)
    print(f"ALL DONE: {new_total} new analyses in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Total in {output_dir}/: {final_count}")

    # Check exit codes
    for i, (proc, log_file, lf, n) in enumerate(processes):
        log_file.close()
        status = "OK" if proc.returncode == 0 else f"FAILED (exit {proc.returncode})"
        print(f"  Process {i}: {status} — log at /tmp/batch_process_{i}.log")


if __name__ == "__main__":
    main()
