#!/usr/bin/env python3
"""
Compare two sets of golden transcript analyses field-by-field.

Loads analyses from two directories (supports both flat and nested layouts),
keys them by call_id, and reports differences across three tiers:
  CRITICAL - scope, outcome, failure type, transfer reason
  IMPORTANT - secondary present, abandon_stage, sentiment start/end
  INFO      - scores, derailed_at, turns

Usage:
    python3 tools/compare_golden.py <baseline_dir> <new_dir>
    python3 tools/compare_golden.py <baseline_dir> <new_dir> --no-info --diffs-only

Exit code: 0 if all CRITICAL fields match, 1 otherwise.
"""

import argparse
import json
import sys
from pathlib import Path

# ── Field definitions ────────────────────────────────────────────────────

CRITICAL_FIELDS = [
    ("scope",           lambda d: _get(d, "resolution", "primary", "scope")),
    ("outcome",         lambda d: _get(d, "resolution", "primary", "outcome")),
    ("failure",         lambda d: _get(d, "failure", "type") if isinstance(d.get("failure"), dict) else (None if d.get("failure") is None else d.get("failure"))),
    ("transfer_reason", lambda d: _get(d, "resolution", "primary", "transfer", "reason")),
]

IMPORTANT_FIELDS = [
    ("secondary",       lambda d: _get(d, "resolution", "secondary") is not None),
    ("abandon_stage",   lambda d: _get(d, "resolution", "primary", "abandon_stage")),
    ("sentiment_start", lambda d: _get(d, "sentiment", "start")),
    ("sentiment_end",   lambda d: _get(d, "sentiment", "end")),
]

INFO_FIELDS = [
    ("effectiveness",   lambda d: _get(d, "scores", "effectiveness")),
    ("quality",         lambda d: _get(d, "scores", "quality")),
    ("effort",          lambda d: _get(d, "scores", "effort")),
    ("derailed_at",     lambda d: _get(d, "derailed_at")),
    ("turns",           lambda d: _get(d, "turns")),
]

NUMERIC_INFO = {"effectiveness", "quality", "effort", "turns"}

# ── Helpers ──────────────────────────────────────────────────────────────

def _get(d: dict, *keys):
    """Safely traverse nested dict keys, returning None on any miss."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(k)
    return cur


def load_analyses(directory: Path) -> dict[str, dict]:
    """Load all JSON analyses from a directory (flat or nested subdirs).
    Returns {call_id: raw_dict}.
    """
    analyses = {}
    for p in directory.rglob("*.json"):
        try:
            data = json.loads(p.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  WARNING: skipping {p}: {e}", file=sys.stderr)
            continue
        call_id = data.get("call_id", p.stem)
        analyses[call_id] = data
    return analyses


def compare_field(baseline: dict, new: dict, field_name: str, extractor) -> dict:
    """Compare a single field between baseline and new analysis.
    Returns {field, baseline_val, new_val, status} where status is
    '==' | 'DIFF' | 'N/A' (both null).
    """
    b_val = extractor(baseline)
    n_val = extractor(new)

    if b_val is None and n_val is None:
        status = "N/A"
    elif b_val == n_val:
        status = "=="
    else:
        status = "DIFF"

    return {"field": field_name, "baseline": b_val, "new": n_val, "status": status}


# ── Core comparison ──────────────────────────────────────────────────────

def compare_all(baseline: dict[str, dict], new: dict[str, dict], include_info: bool = True):
    """Compare all matching call_ids across tiers.
    Returns (per_call_results, missing_in_new, extra_in_new).
    """
    b_ids = set(baseline)
    n_ids = set(new)
    common = sorted(b_ids & n_ids)
    missing_in_new = sorted(b_ids - n_ids)
    extra_in_new = sorted(n_ids - b_ids)

    fields = (
        [("CRITICAL", f) for f in CRITICAL_FIELDS]
        + [("IMPORTANT", f) for f in IMPORTANT_FIELDS]
    )
    if include_info:
        fields += [("INFO", f) for f in INFO_FIELDS]

    per_call = []
    for cid in common:
        comparisons = []
        for tier, (fname, extractor) in fields:
            result = compare_field(baseline[cid], new[cid], fname, extractor)
            result["tier"] = tier
            comparisons.append(result)
        per_call.append({"call_id": cid, "comparisons": comparisons})

    return per_call, missing_in_new, extra_in_new


# ── Output formatting ────────────────────────────────────────────────────

def short_id(call_id: str) -> str:
    """First 8 chars of UUID for compact display."""
    return call_id[:8]


def print_per_call_table(per_call: list[dict], diffs_only: bool):
    """Section 1: compact row per call."""
    # Gather all field names from first call
    if not per_call:
        print("No common calls to compare.")
        return

    field_names = [c["field"] for c in per_call[0]["comparisons"]]
    col_w = max(len(f) for f in field_names)
    col_w = max(col_w, 6)

    # Header
    header = f"{'call_id':<10} " + " ".join(f"{f:^{col_w}}" for f in field_names)
    print(header)
    print("-" * len(header))

    for row in per_call:
        statuses = [c["status"] for c in row["comparisons"]]
        has_diff = any(s == "DIFF" for s in statuses)
        if diffs_only and not has_diff:
            continue

        cells = []
        for s in statuses:
            if s == "DIFF":
                cells.append(f"{'DIFF':^{col_w}}")
            elif s == "N/A":
                cells.append(f"{'--':^{col_w}}")
            else:
                cells.append(f"{'==':^{col_w}}")

        line = f"{short_id(row['call_id']):<10} " + " ".join(cells)
        print(line)


def print_field_accuracy(per_call: list[dict]):
    """Section 2: match count / total / percentage per field."""
    if not per_call:
        return

    field_names = [c["field"] for c in per_call[0]["comparisons"]]
    field_tiers = {c["field"]: c["tier"] for c in per_call[0]["comparisons"]}

    print(f"\n{'Field':<18} {'Tier':<10} {'Match':>5} {'Total':>5} {'Pct':>7}  {'Note'}")
    print("-" * 65)

    has_critical_diff = False

    for fname in field_names:
        matches = 0
        total = 0
        deltas = []

        for row in per_call:
            comp = next(c for c in row["comparisons"] if c["field"] == fname)
            if comp["status"] == "N/A":
                continue
            total += 1
            if comp["status"] == "==":
                matches += 1
            elif fname in NUMERIC_INFO and comp["baseline"] is not None and comp["new"] is not None:
                try:
                    deltas.append(abs(float(comp["new"]) - float(comp["baseline"])))
                except (TypeError, ValueError):
                    pass

        tier = field_tiers[fname]
        if total == 0:
            pct_str = "  N/A  "
            note = "all null"
        else:
            pct = matches / total * 100
            pct_str = f"{pct:6.1f}%"
            note = ""
            if pct < 100 and tier == "CRITICAL":
                has_critical_diff = True
            if deltas:
                avg_delta = sum(deltas) / len(deltas)
                note = f"avg Δ={avg_delta:.1f}"

        print(f"{fname:<18} {tier:<10} {matches:>5} {total:>5} {pct_str}  {note}")

    return has_critical_diff


def print_diff_details(per_call: list[dict]):
    """Section 3: every single difference."""
    diffs = []
    for row in per_call:
        for comp in row["comparisons"]:
            if comp["status"] == "DIFF":
                diffs.append({
                    "call_id": row["call_id"],
                    "field": comp["field"],
                    "tier": comp["tier"],
                    "baseline": comp["baseline"],
                    "new": comp["new"],
                })

    if not diffs:
        print("\nNo differences found.")
        return

    print(f"\n{'call_id':<10} {'field':<18} {'tier':<10} {'baseline':<25} {'new':<25}")
    print("-" * 90)
    for d in diffs:
        b_str = str(d["baseline"]) if d["baseline"] is not None else "null"
        n_str = str(d["new"]) if d["new"] is not None else "null"
        print(f"{short_id(d['call_id']):<10} {d['field']:<18} {d['tier']:<10} {b_str:<25} {n_str:<25}")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare two sets of golden transcript analyses"
    )
    parser.add_argument("baseline_dir", type=Path, help="Baseline analyses directory")
    parser.add_argument("new_dir", type=Path, help="New analyses directory")
    parser.add_argument("--no-info", action="store_true", help="Hide INFO-tier fields")
    parser.add_argument("--diffs-only", action="store_true", help="Only show calls with differences")
    args = parser.parse_args()

    if not args.baseline_dir.is_dir():
        print(f"Error: baseline directory not found: {args.baseline_dir}", file=sys.stderr)
        sys.exit(2)
    if not args.new_dir.is_dir():
        print(f"Error: new directory not found: {args.new_dir}", file=sys.stderr)
        sys.exit(2)

    print(f"Baseline: {args.baseline_dir}")
    print(f"New:      {args.new_dir}")

    baseline = load_analyses(args.baseline_dir)
    new = load_analyses(args.new_dir)
    print(f"Loaded {len(baseline)} baseline, {len(new)} new analyses\n")

    per_call, missing, extra = compare_all(baseline, new, include_info=not args.no_info)

    # Report missing/extra
    if missing:
        print(f"⚠ MISSING in new ({len(missing)}): {', '.join(short_id(c) for c in missing)}")
    if extra:
        print(f"⚠ EXTRA in new ({len(extra)}): {', '.join(short_id(c) for c in extra)}")
    if missing or extra:
        print()

    # Section 1: Per-call table
    print("═══ Per-Call Comparison ═══\n")
    print_per_call_table(per_call, diffs_only=args.diffs_only)

    # Section 2: Field accuracy
    print("\n═══ Field Accuracy ═══")
    has_critical_diff = print_field_accuracy(per_call)

    # Section 3: Diff details
    print("\n═══ Diff Details ═══")
    print_diff_details(per_call)

    # Exit code
    if has_critical_diff:
        print("\n✘ CRITICAL field differences detected")
        sys.exit(1)
    else:
        print("\n✔ All CRITICAL fields match")
        sys.exit(0)


if __name__ == "__main__":
    main()
