#!/usr/bin/env python3
"""
Aggregate Metrics Calculator for Vacatia AI Voice Agent Analytics (v2)

Simplified metrics focused on actionable insights.
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


def load_analyses(analyses_dir: Path) -> list[dict]:
    """Load all v2 analysis JSON files from directory."""
    analyses = []
    for f in analyses_dir.iterdir():
        if f.is_file() and f.suffix == '.json':
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    # Only load v2 schema files (or unversioned for backward compat)
                    if data.get("schema_version", "v2") == "v2":
                        analyses.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {f}: {e}", file=sys.stderr)
    return analyses


def safe_rate(num: int, den: int) -> float | None:
    """Calculate rate safely."""
    return round(num / den, 4) if den > 0 else None


def safe_stats(values: list) -> dict:
    """Compute statistics for numeric values."""
    values = [v for v in values if v is not None and isinstance(v, (int, float))]
    if not values:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    return {
        "n": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "min": min(values),
        "max": max(values)
    }


def generate_report(analyses: list[dict]) -> dict:
    """Generate v2 metrics report."""
    if not analyses:
        return {"error": "No analyses to process"}

    n = len(analyses)

    # Outcome distribution
    outcomes = Counter(a.get("outcome", "unclear") for a in analyses)

    # Resolution types (free text - show top 10)
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

    # Training opportunities
    training_opps = Counter(
        a.get("training_opportunity")
        for a in analyses if a.get("training_opportunity")
    )

    # Multi-intent calls
    multi_intent = sum(1 for a in analyses if a.get("additional_intents"))

    return {
        "summary": {
            "total_calls": n,
            "schema_version": "v2",
            "generated": datetime.now().isoformat()
        },

        "outcomes": {
            "distribution": {k: {"count": v, "rate": safe_rate(v, n)} for k, v in outcomes.items()},
            "success_rate": safe_rate(outcomes.get("resolved", 0), n),
            "escalation_rate": safe_rate(outcomes.get("escalated", 0), n)
        },

        "resolution_types": dict(resolution_types.most_common(15)),

        "quality_scores": {
            "agent_effectiveness": safe_stats(effectiveness),
            "conversation_quality": safe_stats(quality),
            "customer_effort": safe_stats(effort)
        },

        "failure_analysis": {
            "total_failures": len(failures),
            "failure_rate": safe_rate(len(failures), n),
            "by_failure_point": dict(failure_points.most_common()),
            "recoverable_count": recoverable,
            "recoverable_rate": safe_rate(recoverable, len(failures)),
            "critical_failures": critical
        },

        "actionable_flags": {
            "escalation_requested": {"count": escalation_requested, "rate": safe_rate(escalation_requested, n)},
            "repeat_callers": {"count": repeat_callers, "rate": safe_rate(repeat_callers, n)},
            "multi_intent_calls": {"count": multi_intent, "rate": safe_rate(multi_intent, n)}
        },

        "training_priorities": dict(training_opps.most_common(10))
    }


def print_summary(report: dict) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 55)
    print("VACATIA AI VOICE AGENT - ANALYTICS REPORT (v2)")
    print("=" * 55)

    s = report.get("summary", {})
    print(f"\nCalls Analyzed: {s.get('total_calls', 0)}")

    # Outcomes
    print("\n" + "-" * 35)
    print("OUTCOMES")
    print("-" * 35)
    outcomes = report.get("outcomes", {})
    sr = outcomes.get("success_rate")
    print(f"  Success Rate: {sr*100:.1f}%" if sr else "  Success Rate: N/A")
    for outcome, data in outcomes.get("distribution", {}).items():
        print(f"  {outcome}: {data['count']} ({data['rate']*100:.1f}%)")

    # Quality
    print("\n" + "-" * 35)
    print("QUALITY SCORES (1-5)")
    print("-" * 35)
    q = report.get("quality_scores", {})
    for name, stats in q.items():
        label = name.replace("_", " ").title()
        if stats.get("mean"):
            print(f"  {label}: {stats['mean']:.1f} avg (n={stats['n']})")

    # Failures
    print("\n" + "-" * 35)
    print("FAILURE ANALYSIS")
    print("-" * 35)
    f = report.get("failure_analysis", {})
    print(f"  Total Failures: {f.get('total_failures', 0)} ({f.get('failure_rate', 0)*100:.1f}%)")
    print(f"  Recoverable: {f.get('recoverable_count', 0)}")
    print(f"  Critical: {f.get('critical_failures', 0)}")
    print("  By Type:")
    for fp, count in f.get("by_failure_point", {}).items():
        print(f"    {fp}: {count}")

    # Actionable
    print("\n" + "-" * 35)
    print("ACTIONABLE FLAGS")
    print("-" * 35)
    a = report.get("actionable_flags", {})
    for flag, data in a.items():
        label = flag.replace("_", " ").title()
        print(f"  {label}: {data['count']} ({data['rate']*100:.1f}%)" if data.get('rate') else f"  {label}: {data['count']}")

    # Training
    print("\n" + "-" * 35)
    print("TRAINING PRIORITIES")
    print("-" * 35)
    t = report.get("training_priorities", {})
    if t:
        for skill, count in t.items():
            print(f"  {skill}: {count} calls")
    else:
        print("  None identified")

    # Top resolution types
    print("\n" + "-" * 35)
    print("TOP RESOLUTION TYPES")
    print("-" * 35)
    for rt, count in list(report.get("resolution_types", {}).items())[:8]:
        print(f"  {rt}: {count}")

    print("\n" + "=" * 55)


def main():
    parser = argparse.ArgumentParser(description="Compute v2 aggregate metrics")
    parser.add_argument("-i", "--input-dir", type=Path,
                        default=Path(__file__).parent.parent.parent / "analyses",
                        help="Directory containing analysis JSON files")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path(__file__).parent.parent.parent / "reports",
                        help="Output directory for report")
    parser.add_argument("--json-only", action="store_true",
                        help="Output only JSON")

    args = parser.parse_args()

    print(f"Loading from: {args.input_dir}", file=sys.stderr)
    analyses = load_analyses(args.input_dir)
    print(f"Loaded {len(analyses)} v2 analyses", file=sys.stderr)

    if not analyses:
        print("Error: No v2 analysis files found", file=sys.stderr)
        return 1

    report = generate_report(analyses)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.output_dir / f"metrics_report_v2_{timestamp}.json"

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2)
    print(f"Report saved: {report_path}", file=sys.stderr)

    if args.json_only:
        print(json.dumps(report, indent=2))
    else:
        print_summary(report)
        print(f"\nFull report: {report_path}")

    return 0


if __name__ == "__main__":
    exit(main())
