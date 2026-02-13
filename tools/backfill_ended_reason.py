#!/usr/bin/env python3
"""
Backfill ended_reason, fix ended_by, and override queue_result on existing analysis files.

All changes are deterministic — derived from the raw transcript metadata and
existing analysis fields. No LLM calls.

What it does per analysis file:
  1. Looks up the matching transcript by call_id (filename stem)
  2. Reads ended_reason from the transcript
  3. Recomputes ended_by via _map_ended_reason() (fixes hangup/silence bug)
  4. Overrides queue_result using _override_queue_result() rules
  5. Writes ended_reason to the analysis

Usage:
    # Dry run (default) — shows what would change
    python3 tools/backfill_ended_reason.py \
        --analyses-dir runs/v7_batch_feb11/ \
        --transcripts-dir transcripts/transcripts-feb10/

    # Apply changes
    python3 tools/backfill_ended_reason.py \
        --analyses-dir runs/v7_batch_feb11/ \
        --transcripts-dir transcripts/transcripts-feb10/ \
        --apply
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from poc_structured_full import _map_ended_reason


def _override_queue_result_dict(analysis: dict, ended_reason: str | None) -> list[str]:
    """Apply queue_result override on raw dict. Returns list of change descriptions."""
    if not ended_reason:
        return []
    r = ended_reason.lower()
    changes = []

    for label in ("primary", "secondary"):
        intent = analysis.get("resolution", {}).get(label)
        if intent is None:
            continue
        transfer = intent.get("transfer")
        if transfer is None:
            continue

        old_qr = transfer.get("queue_result")

        if "forwarded" in r:
            if old_qr != "connected":
                transfer["queue_result"] = "connected"
                changes.append(f"{label}.queue_result: {old_qr} -> connected")
        elif "hangup" in r or "customer-ended" in r or "silence" in r:
            if transfer.get("queue_detected"):
                if old_qr != "caller_abandoned":
                    transfer["queue_result"] = "caller_abandoned"
                    changes.append(f"{label}.queue_result: {old_qr} -> caller_abandoned")

    return changes


def backfill(analyses_dir: Path, transcripts_dir: Path, apply: bool = False) -> dict:
    """Backfill ended_reason, fix ended_by, override queue_result.

    Returns summary dict with counts.
    """
    analysis_files = sorted(analyses_dir.glob("*.json"))
    transcript_index = {p.stem: p for p in transcripts_dir.glob("*.json")}

    stats = {
        "total": len(analysis_files),
        "transcript_found": 0,
        "transcript_missing": 0,
        "ended_reason_added": 0,
        "ended_by_fixed": 0,
        "queue_result_fixed": 0,
        "files_modified": 0,
        "unchanged": 0,
    }
    missing_transcripts = []
    all_changes = []

    for af in analysis_files:
        call_id = af.stem
        tp = transcript_index.get(call_id)
        if tp is None:
            stats["transcript_missing"] += 1
            missing_transcripts.append(call_id)
            continue
        stats["transcript_found"] += 1

        # Read transcript ended_reason
        with open(tp) as f:
            transcript_data = json.load(f)
        ended_reason_raw = transcript_data.get("ended_reason")

        # Read analysis
        with open(af) as f:
            analysis = json.load(f)

        file_changes = []

        # 1. Add/update ended_reason
        old_er = analysis.get("ended_reason")
        if old_er != ended_reason_raw:
            analysis["ended_reason"] = ended_reason_raw
            file_changes.append(f"ended_reason: {old_er} -> {ended_reason_raw}")
            stats["ended_reason_added"] += 1

        # 2. Fix ended_by
        new_ended_by = _map_ended_reason(ended_reason_raw)
        old_ended_by = analysis.get("ended_by")
        if old_ended_by != new_ended_by:
            analysis["ended_by"] = new_ended_by
            file_changes.append(f"ended_by: {old_ended_by} -> {new_ended_by}")
            stats["ended_by_fixed"] += 1

        # 3. Override queue_result
        qr_changes = _override_queue_result_dict(analysis, ended_reason_raw)
        if qr_changes:
            file_changes.extend(qr_changes)
            stats["queue_result_fixed"] += len(qr_changes)

        if file_changes:
            stats["files_modified"] += 1
            all_changes.append((call_id, file_changes))
            if apply:
                with open(af, "w") as f:
                    json.dump(analysis, f, indent=2)
        else:
            stats["unchanged"] += 1

    return stats, all_changes, missing_transcripts


def main():
    parser = argparse.ArgumentParser(description="Backfill ended_reason on existing analysis files")
    parser.add_argument("--analyses-dir", required=True, type=Path)
    parser.add_argument("--transcripts-dir", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Write changes (default: dry run)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show per-file changes")
    args = parser.parse_args()

    if not args.analyses_dir.is_dir():
        print(f"Error: {args.analyses_dir} is not a directory", file=sys.stderr)
        sys.exit(1)
    if not args.transcripts_dir.is_dir():
        print(f"Error: {args.transcripts_dir} is not a directory", file=sys.stderr)
        sys.exit(1)

    mode = "APPLYING" if args.apply else "DRY RUN"
    print(f"=== Backfill ended_reason ({mode}) ===")
    print(f"Analyses:    {args.analyses_dir}")
    print(f"Transcripts: {args.transcripts_dir}")
    print()

    stats, all_changes, missing = backfill(args.analyses_dir, args.transcripts_dir, apply=args.apply)

    # Per-file changes
    if args.verbose and all_changes:
        print(f"--- Changes ({len(all_changes)} files) ---")
        for call_id, changes in all_changes[:50]:
            print(f"  {call_id}:")
            for c in changes:
                print(f"    {c}")
        if len(all_changes) > 50:
            print(f"  ... and {len(all_changes) - 50} more")
        print()

    # Summary
    print("=== Summary ===")
    print(f"  Total analysis files:   {stats['total']}")
    print(f"  Transcript found:       {stats['transcript_found']}")
    print(f"  Transcript missing:     {stats['transcript_missing']}")
    print()
    print(f"  ended_reason added:     {stats['ended_reason_added']}")
    print(f"  ended_by fixed:         {stats['ended_by_fixed']}")
    print(f"  queue_result fixed:     {stats['queue_result_fixed']}")
    print(f"  Files modified:         {stats['files_modified']}")
    print(f"  Unchanged:              {stats['unchanged']}")

    if missing:
        print(f"\n  Missing transcripts ({len(missing)}):")
        for cid in missing[:10]:
            print(f"    {cid}")
        if len(missing) > 10:
            print(f"    ... and {len(missing) - 10} more")

    if not args.apply and stats["files_modified"] > 0:
        print(f"\nRe-run with --apply to write changes.")


if __name__ == "__main__":
    main()
