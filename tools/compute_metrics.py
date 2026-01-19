#!/usr/bin/env python3
"""
Aggregate Metrics Calculator for Vacatia AI Voice Agent Analytics (v3.6)

Computes Section A: Deterministic Metrics - all Python-calculated, reproducible and auditable.

v3.6 additions:
- conversation_quality_metrics: Turn stats, clarification stats, correction stats, loop stats
- Aggregation of new v3.6 fields: conversation_turns, clarification_requests, user_corrections, repeated_prompts

v3.3 additions:
- validate_failure_consistency: Flags failure_point=none for non-resolved calls
- validation_warnings: Included in report output for data quality checks

Note: NL field extraction is now handled by the dedicated extract_nl_fields.py script (v3.1).
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


def load_manifest_ids(sampled_dir: Path) -> set[str] | None:
    """Load call IDs from manifest.csv if it exists."""
    manifest_path = sampled_dir / "manifest.csv"
    if not manifest_path.exists():
        return None

    call_ids = set()
    with open(manifest_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            # Extract call_id from filename (e.g., "abc123.txt" -> "abc123")
            filename = row.get("filename", "")
            if filename:
                call_ids.add(Path(filename).stem)
    return call_ids


def load_analyses(analyses_dir: Path, schema_version: str = "v3", manifest_ids: set[str] | None = None) -> list[dict]:
    """Load analysis JSON files from directory matching the specified schema version.

    Args:
        analyses_dir: Directory containing analysis JSON files
        schema_version: Schema version to filter by ("v3" or "v2")
        manifest_ids: Optional set of call_ids to filter by (scope coherence)
    """
    analyses = []
    skipped_not_in_manifest = 0

    for f in analyses_dir.iterdir():
        if f.is_file() and f.suffix == '.json':
            # v3.3: Scope coherence - filter by manifest if provided
            if manifest_ids is not None and f.stem not in manifest_ids:
                skipped_not_in_manifest += 1
                continue

            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    version = data.get("schema_version", "v2")
                    # Accept v3.x or v2 (v2 is forward-compatible)
                    # v3.6, v3.5, v3 all start with "v3"
                    if schema_version == "v3" and (version.startswith("v3") or version == "v2"):
                        analyses.append(data)
                    elif schema_version == "v2" and version == "v2":
                        analyses.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {f}: {e}", file=sys.stderr)

    if skipped_not_in_manifest > 0:
        print(f"Scope filter: Skipped {skipped_not_in_manifest} analyses not in manifest", file=sys.stderr)

    return analyses


def safe_rate(num: int, den: int, decimals: int = 3) -> float | None:
    """Calculate rate safely with configurable precision."""
    return round(num / den, decimals) if den > 0 else None


def safe_stats(values: list) -> dict:
    """Compute statistics for numeric values."""
    values = [v for v in values if v is not None and isinstance(v, (int, float))]
    if not values:
        return {"n": 0, "mean": None, "median": None, "std": None, "min": None, "max": None}

    result = {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": min(values),
        "max": max(values)
    }

    if len(values) > 1:
        result["std"] = round(statistics.stdev(values), 2)
    else:
        result["std"] = 0.0

    return result


def validate_failure_consistency(analyses: list[dict]) -> dict:
    """
    Flag failure_point inconsistencies (v3.3).

    A non-resolved call (abandoned, escalated, unclear) should NOT have
    failure_point='none' - this indicates a data quality issue.
    """
    warnings = []
    for a in analyses:
        outcome = a.get("outcome")
        failure_point = a.get("failure_point")

        # Non-resolved outcomes should have a failure_point other than "none"
        if outcome in ("abandoned", "escalated", "unclear") and failure_point == "none":
            warnings.append({
                "call_id": a.get("call_id"),
                "outcome": outcome,
                "issue": "failure_point='none' invalid for non-resolved call"
            })

    return {"failure_point_inconsistencies": warnings}


def extract_policy_gap_breakdown(analyses: list[dict]) -> dict:
    """Extract policy gap analysis from v3 policy_gap_detail fields."""
    policy_gaps = [a for a in analyses if a.get("failure_point") == "policy_gap"]

    if not policy_gaps:
        return {
            "by_category": {},
            "top_specific_gaps": [],
            "top_customer_asks": []
        }

    # Count by category
    categories = Counter()
    specific_gaps = Counter()
    customer_asks = Counter()

    for a in policy_gaps:
        detail = a.get("policy_gap_detail")
        if detail and isinstance(detail, dict):
            cat = detail.get("category")
            if cat:
                categories[cat] += 1

            gap = detail.get("specific_gap")
            if gap:
                # Normalize the gap description for grouping
                specific_gaps[gap.lower().strip()] += 1

            ask = detail.get("customer_ask")
            if ask:
                customer_asks[ask.lower().strip()] += 1

    total_with_category = sum(categories.values())

    return {
        "by_category": {
            k: {"count": v, "rate": safe_rate(v, total_with_category)}
            for k, v in categories.most_common()
        },
        "top_specific_gaps": [
            {"gap": gap, "count": count}
            for gap, count in specific_gaps.most_common(10)
        ],
        "top_customer_asks": [
            {"ask": ask, "count": count}
            for ask, count in customer_asks.most_common(10)
        ]
    }


def compute_conversation_quality_metrics(analyses: list[dict]) -> dict:
    """
    Compute conversation quality metrics from v3.6 fields.

    Aggregates:
    - Turn statistics (avg, median, by outcome, turns-to-failure)
    - Clarification request statistics (by type, resolution rate)
    - User correction statistics (frustration rate)
    - Loop detection statistics (calls with loops, max consecutive)
    """
    n = len(analyses)
    if n == 0:
        return {}

    # === Turn Statistics ===
    all_turns = []
    resolved_turns = []
    failed_turns = []
    turns_to_failure = []

    for a in analyses:
        turns = a.get("conversation_turns")
        if turns is not None and isinstance(turns, (int, float)):
            all_turns.append(turns)
            if a.get("outcome") == "resolved":
                resolved_turns.append(turns)
            else:
                failed_turns.append(turns)

        ttf = a.get("turns_to_failure")
        if ttf is not None and isinstance(ttf, (int, float)):
            turns_to_failure.append(ttf)

    turn_stats = {
        "avg_turns": round(statistics.mean(all_turns), 1) if all_turns else None,
        "median_turns": round(statistics.median(all_turns), 1) if all_turns else None,
        "avg_turns_resolved": round(statistics.mean(resolved_turns), 1) if resolved_turns else None,
        "avg_turns_failed": round(statistics.mean(failed_turns), 1) if failed_turns else None,
        "avg_turns_to_failure": round(statistics.mean(turns_to_failure), 1) if turns_to_failure else None,
        "calls_with_turn_data": len(all_turns)
    }

    # === Clarification Statistics ===
    clarification_counts = []
    clarification_types = Counter()
    clarification_resolved = 0
    clarification_total_details = 0
    calls_with_clarifications = 0

    for a in analyses:
        clar = a.get("clarification_requests")
        if clar and isinstance(clar, dict):
            count = clar.get("count") or 0
            if count > 0:
                calls_with_clarifications += 1
                clarification_counts.append(count)

            details = clar.get("details", [])
            if isinstance(details, list):
                for d in details:
                    clarification_total_details += 1
                    ctype = d.get("type")
                    if ctype:
                        clarification_types[ctype] += 1
                    if d.get("resolved"):
                        clarification_resolved += 1

    clarification_stats = {
        "calls_with_clarifications": calls_with_clarifications,
        "pct_calls_with_clarifications": safe_rate(calls_with_clarifications, n),
        "avg_clarifications_per_call": round(statistics.mean(clarification_counts), 2) if clarification_counts else 0,
        "by_type": {
            ctype: {"count": count, "rate": safe_rate(count, n)}
            for ctype, count in clarification_types.most_common()
        },
        "resolution_rate": safe_rate(clarification_resolved, clarification_total_details) if clarification_total_details > 0 else None
    }

    # === Correction Statistics ===
    correction_counts = []
    corrections_with_frustration = 0
    total_corrections = 0
    calls_with_corrections = 0

    for a in analyses:
        corr = a.get("user_corrections")
        if corr and isinstance(corr, dict):
            count = corr.get("count") or 0
            if count > 0:
                calls_with_corrections += 1
                correction_counts.append(count)

            details = corr.get("details", [])
            if isinstance(details, list):
                for d in details:
                    total_corrections += 1
                    if d.get("frustration_signal"):
                        corrections_with_frustration += 1

    correction_stats = {
        "calls_with_corrections": calls_with_corrections,
        "pct_calls_with_corrections": safe_rate(calls_with_corrections, n),
        "avg_corrections_per_call": round(statistics.mean(correction_counts), 2) if correction_counts else 0,
        "with_frustration_signal": corrections_with_frustration,
        "frustration_rate": safe_rate(corrections_with_frustration, total_corrections) if total_corrections > 0 else None
    }

    # === Loop Statistics ===
    calls_with_loops = 0
    all_repeat_counts = []
    all_max_consecutive = []

    for a in analyses:
        rp = a.get("repeated_prompts")
        if rp and isinstance(rp, dict):
            count = rp.get("count") or 0
            if count > 0:
                calls_with_loops += 1
                all_repeat_counts.append(count)

            max_cons = rp.get("max_consecutive") or 0
            if max_cons > 0:
                all_max_consecutive.append(max_cons)

    loop_stats = {
        "calls_with_loops": calls_with_loops,
        "pct_calls_with_loops": safe_rate(calls_with_loops, n),
        "avg_repeats": round(statistics.mean(all_repeat_counts), 2) if all_repeat_counts else 0,
        "max_consecutive_overall": max(all_max_consecutive) if all_max_consecutive else 0
    }

    return {
        "turn_stats": turn_stats,
        "clarification_stats": clarification_stats,
        "correction_stats": correction_stats,
        "loop_stats": loop_stats
    }


def aggregate_natural_language_fields(analyses: list[dict]) -> dict:
    """Aggregate natural language fields for LLM insight generation."""

    # Group failure descriptions by failure_point
    failure_descriptions_by_type = {}
    for a in analyses:
        fp = a.get("failure_point")
        desc = a.get("failure_description")
        if fp and fp != "none" and desc:
            if fp not in failure_descriptions_by_type:
                failure_descriptions_by_type[fp] = []
            failure_descriptions_by_type[fp].append(desc)

    # Collect customer verbatims
    customer_verbatims = [
        a.get("customer_verbatim")
        for a in analyses
        if a.get("customer_verbatim")
    ]

    # Collect agent miss details
    agent_miss_details = [
        a.get("agent_miss_detail")
        for a in analyses
        if a.get("agent_miss_detail")
    ]

    # Collect policy gap specific_gaps
    policy_gap_specifics = []
    for a in analyses:
        detail = a.get("policy_gap_detail")
        if detail and isinstance(detail, dict):
            gap = detail.get("specific_gap")
            if gap:
                policy_gap_specifics.append(gap)

    # Sample resolution_steps for failed calls
    failed_resolution_steps = []
    for a in analyses:
        if a.get("outcome") != "resolved":
            steps = a.get("resolution_steps")
            if steps and isinstance(steps, list):
                failed_resolution_steps.append({
                    "call_id": a.get("call_id"),
                    "outcome": a.get("outcome"),
                    "steps": steps
                })

    return {
        "failure_descriptions_by_type": failure_descriptions_by_type,
        "customer_verbatims": customer_verbatims,
        "agent_miss_details": agent_miss_details,
        "policy_gap_specifics": policy_gap_specifics,
        "failed_resolution_steps": failed_resolution_steps[:20]  # Limit sample
    }


def generate_report(analyses: list[dict]) -> dict:
    """Generate v3 Section A: Deterministic Metrics report."""
    if not analyses:
        return {"error": "No analyses to process"}

    n = len(analyses)

    # Get date range from call_ids or timestamps
    timestamps = [a.get("timestamp") for a in analyses if a.get("timestamp")]
    date_range = {
        "earliest": min(timestamps) if timestamps else None,
        "latest": max(timestamps) if timestamps else None
    }

    # Outcome distribution
    outcomes = Counter(a.get("outcome", "unclear") for a in analyses)

    # Resolution types (free text - show top 15)
    resolution_types = Counter(
        a.get("resolution_type", "unknown")
        for a in analyses if a.get("resolution_type")
    )

    # Quality scores
    effectiveness = [a.get("agent_effectiveness") for a in analyses]
    quality = [a.get("conversation_quality") for a in analyses]
    effort = [a.get("customer_effort") for a in analyses]

    # Failure analysis
    failures = [a for a in analyses if a.get("outcome") != "resolved"]
    failure_points = Counter(
        a.get("failure_point", "unknown")
        for a in failures if a.get("failure_point")
    )
    recoverable = sum(1 for a in failures if a.get("was_recoverable") is True)
    critical = sum(1 for a in analyses if a.get("critical_failure") is True)

    # Actionable flags
    escalation_requested = sum(1 for a in analyses if a.get("escalation_requested"))
    repeat_callers = sum(1 for a in analyses if a.get("repeat_caller_signals"))
    multi_intent = sum(1 for a in analyses if a.get("additional_intents"))

    # Training opportunities
    training_opps = Counter(
        a.get("training_opportunity")
        for a in analyses if a.get("training_opportunity")
    )

    # Policy gap breakdown (v3 feature)
    policy_gap_breakdown = extract_policy_gap_breakdown(analyses)

    # Conversation quality metrics (v3.6 feature)
    conversation_quality_metrics = compute_conversation_quality_metrics(analyses)

    # Calculate key rates
    resolved_count = outcomes.get("resolved", 0)
    escalated_count = outcomes.get("escalated", 0)
    abandoned_count = outcomes.get("abandoned", 0)

    # Containment = resolved + abandoned (not escalated to human)
    containment_count = resolved_count + abandoned_count

    # Validation warnings (v3.3)
    validation_warnings = validate_failure_consistency(analyses)

    return {
        "metadata": {
            "report_generated": datetime.now().isoformat(),
            "schema_version": "v3",
            "total_calls_analyzed": n,
            "date_range": date_range
        },

        "outcome_distribution": {
            k: {"count": v, "rate": safe_rate(v, n)}
            for k, v in outcomes.items()
        },

        "key_rates": {
            "success_rate": safe_rate(resolved_count, n),
            "containment_rate": safe_rate(containment_count, n),
            "escalation_rate": safe_rate(escalated_count, n),
            "failure_rate": safe_rate(n - resolved_count, n)
        },

        "quality_scores": {
            "agent_effectiveness": safe_stats(effectiveness),
            "conversation_quality": safe_stats(quality),
            "customer_effort": safe_stats(effort)
        },

        "failure_analysis": {
            "total_failures": len(failures),
            "failure_rate": safe_rate(len(failures), n),
            "by_failure_point": {
                k: {"count": v, "rate": safe_rate(v, len(failures))}
                for k, v in failure_points.most_common()
            },
            "recoverable_count": recoverable,
            "recoverable_rate": safe_rate(recoverable, len(failures)),
            "critical_failure_count": critical
        },

        "policy_gap_breakdown": policy_gap_breakdown,

        "conversation_quality": conversation_quality_metrics,

        "actionable_flags": {
            "escalation_requested": {"count": escalation_requested, "rate": safe_rate(escalation_requested, n)},
            "repeat_callers": {"count": repeat_callers, "rate": safe_rate(repeat_callers, n)},
            "multi_intent_calls": {"count": multi_intent, "rate": safe_rate(multi_intent, n)}
        },

        "training_priorities": dict(training_opps.most_common(10)),

        "resolution_types": dict(resolution_types.most_common(15)),

        "validation_warnings": validation_warnings
    }


def print_summary(report: dict) -> None:
    """Print human-readable summary of Section A metrics."""
    print("\n" + "=" * 60)
    print("VACATIA AI VOICE AGENT - ANALYTICS REPORT (v3 - Section A)")
    print("=" * 60)

    m = report.get("metadata", {})
    print(f"\nCalls Analyzed: {m.get('total_calls_analyzed', 0)}")
    print(f"Generated: {m.get('report_generated', 'N/A')}")

    # Key Rates
    print("\n" + "-" * 40)
    print("KEY RATES")
    print("-" * 40)
    rates = report.get("key_rates", {})
    for rate_name, value in rates.items():
        label = rate_name.replace("_", " ").title()
        if value is not None:
            print(f"  {label}: {value*100:.1f}%")

    # Outcome Distribution
    print("\n" + "-" * 40)
    print("OUTCOMES")
    print("-" * 40)
    for outcome, data in report.get("outcome_distribution", {}).items():
        print(f"  {outcome}: {data['count']} ({data['rate']*100:.1f}%)")

    # Quality Scores
    print("\n" + "-" * 40)
    print("QUALITY SCORES (1-5 scale)")
    print("-" * 40)
    q = report.get("quality_scores", {})
    for name, stats in q.items():
        label = name.replace("_", " ").title()
        if stats.get("mean"):
            print(f"  {label}: {stats['mean']:.2f} avg (median={stats['median']}, n={stats['n']})")

    # Failure Analysis
    print("\n" + "-" * 40)
    print("FAILURE ANALYSIS")
    print("-" * 40)
    f = report.get("failure_analysis", {})
    total_failures = f.get('total_failures', 0)
    failure_rate = f.get('failure_rate', 0)
    print(f"  Total Failures: {total_failures} ({failure_rate*100:.1f}%)" if failure_rate else f"  Total Failures: {total_failures}")
    print(f"  Recoverable: {f.get('recoverable_count', 0)}")
    print(f"  Critical: {f.get('critical_failure_count', 0)}")
    print("  By Type:")
    for fp, data in f.get("by_failure_point", {}).items():
        print(f"    {fp}: {data['count']} ({data['rate']*100:.1f}%)" if data.get('rate') else f"    {fp}: {data['count']}")

    # Policy Gap Breakdown
    print("\n" + "-" * 40)
    print("POLICY GAP BREAKDOWN")
    print("-" * 40)
    pgb = report.get("policy_gap_breakdown", {})
    by_cat = pgb.get("by_category", {})
    if by_cat:
        print("  By Category:")
        for cat, data in by_cat.items():
            print(f"    {cat}: {data['count']} ({data['rate']*100:.1f}%)" if data.get('rate') else f"    {cat}: {data['count']}")
    else:
        print("  No policy gaps recorded")

    top_gaps = pgb.get("top_specific_gaps", [])
    if top_gaps:
        print("  Top Specific Gaps:")
        for item in top_gaps[:5]:
            print(f"    - {item['gap']}: {item['count']}")

    top_asks = pgb.get("top_customer_asks", [])
    if top_asks:
        print("  Top Customer Asks:")
        for item in top_asks[:5]:
            print(f"    - {item['ask']}: {item['count']}")

    # Actionable Flags
    print("\n" + "-" * 40)
    print("ACTIONABLE FLAGS")
    print("-" * 40)
    a = report.get("actionable_flags", {})
    for flag, data in a.items():
        label = flag.replace("_", " ").title()
        rate = data.get('rate')
        print(f"  {label}: {data['count']} ({rate*100:.1f}%)" if rate else f"  {label}: {data['count']}")

    # Conversation Quality (v3.6)
    print("\n" + "-" * 40)
    print("CONVERSATION QUALITY (v3.6)")
    print("-" * 40)
    cq = report.get("conversation_quality", {})
    if cq:
        # Turn stats
        ts = cq.get("turn_stats", {})
        if ts.get("avg_turns"):
            print(f"  Avg Turns: {ts['avg_turns']} | Resolved: {ts.get('avg_turns_resolved', 'N/A')} | Failed: {ts.get('avg_turns_failed', 'N/A')}")
            if ts.get("avg_turns_to_failure"):
                print(f"  Avg Turns to Failure: {ts['avg_turns_to_failure']}")

        # Clarification stats
        cs = cq.get("clarification_stats", {})
        if cs.get("calls_with_clarifications"):
            print(f"  Calls with Clarifications: {cs['calls_with_clarifications']} ({cs.get('pct_calls_with_clarifications', 0)*100:.1f}%)")
            print(f"  Avg Clarifications/Call: {cs.get('avg_clarifications_per_call', 0)}")
            if cs.get("resolution_rate"):
                print(f"  Clarification Resolution Rate: {cs['resolution_rate']*100:.1f}%")
            by_type = cs.get("by_type", {})
            if by_type:
                print("  By Type:")
                for ctype, data in by_type.items():
                    print(f"    {ctype}: {data['count']} ({data.get('rate', 0)*100:.1f}%)")

        # Correction stats
        cors = cq.get("correction_stats", {})
        if cors.get("calls_with_corrections"):
            print(f"  Calls with Corrections: {cors['calls_with_corrections']} ({cors.get('pct_calls_with_corrections', 0)*100:.1f}%)")
            if cors.get("frustration_rate"):
                print(f"  Corrections with Frustration: {cors.get('with_frustration_signal', 0)} ({cors['frustration_rate']*100:.1f}%)")

        # Loop stats
        ls = cq.get("loop_stats", {})
        if ls.get("calls_with_loops"):
            print(f"  Calls with Loops: {ls['calls_with_loops']} ({ls.get('pct_calls_with_loops', 0)*100:.1f}%)")
            print(f"  Max Consecutive Repeats: {ls.get('max_consecutive_overall', 0)}")
    else:
        print("  No v3.6 conversation quality data")

    # Training Priorities
    print("\n" + "-" * 40)
    print("TRAINING PRIORITIES")
    print("-" * 40)
    t = report.get("training_priorities", {})
    if t:
        for skill, count in t.items():
            print(f"  {skill}: {count} calls")
    else:
        print("  None identified")

    # Top Resolution Types
    print("\n" + "-" * 40)
    print("TOP RESOLUTION TYPES")
    print("-" * 40)
    for rt, count in list(report.get("resolution_types", {}).items())[:8]:
        print(f"  {rt}: {count}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Compute v3.6 Section A: Deterministic Metrics (incl. conversation quality)")
    parser.add_argument("-i", "--input-dir", type=Path,
                        default=Path(__file__).parent.parent / "analyses",
                        help="Directory containing analysis JSON files")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "reports",
                        help="Output directory for report")
    parser.add_argument("-s", "--sampled-dir", type=Path,
                        default=Path(__file__).parent.parent / "sampled",
                        help="Directory containing manifest.csv for scope filtering (v3.3)")
    parser.add_argument("--no-scope-filter", action="store_true",
                        help="Disable manifest-based scope filtering (include all analyses)")
    parser.add_argument("--json-only", action="store_true",
                        help="Output only JSON")
    parser.add_argument("--schema-version", type=str, choices=["v2", "v3"], default="v3",
                        help="Schema version to process (default: v3)")

    args = parser.parse_args()

    # v3.3: Load manifest for scope coherence
    manifest_ids = None
    if not args.no_scope_filter:
        manifest_ids = load_manifest_ids(args.sampled_dir)
        if manifest_ids:
            print(f"Scope filter: Using manifest with {len(manifest_ids)} call IDs", file=sys.stderr)
        else:
            print("Scope filter: No manifest found, including all analyses", file=sys.stderr)

    print(f"Loading from: {args.input_dir}", file=sys.stderr)
    analyses = load_analyses(args.input_dir, args.schema_version, manifest_ids)
    print(f"Loaded {len(analyses)} {args.schema_version} analyses", file=sys.stderr)

    if not analyses:
        print(f"Error: No {args.schema_version} analysis files found", file=sys.stderr)
        return 1

    report = generate_report(analyses)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save Section A metrics
    report_path = args.output_dir / f"metrics_v3_{timestamp}.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump({"deterministic_metrics": report}, f, indent=2)
    print(f"Section A metrics saved: {report_path}", file=sys.stderr)

    if args.json_only:
        print(json.dumps({"deterministic_metrics": report}, indent=2))
    else:
        print_summary(report)
        print(f"\nFull report: {report_path}")

    return 0


if __name__ == "__main__":
    exit(main())
