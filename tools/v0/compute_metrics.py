#!/usr/bin/env python3
"""
Aggregate Metrics Calculator for Vacatia AI Voice Agent Analytics (v0)

Computes aggregate metrics from collection of analysis JSON files.
This is the original version with Level-0 metrics only.
"""

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any


def load_analyses(analyses_dir: Path) -> list[dict]:
    """Load all analysis JSON files from directory."""
    analyses = []
    for f in analyses_dir.iterdir():
        if f.is_file() and f.suffix == '.json':
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    analyses.append(json.load(fp))
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {f}: {e}", file=sys.stderr)
    return analyses


def safe_rate(numerator: int, denominator: int) -> float | None:
    """Calculate rate safely, returning None if denominator is 0."""
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def compute_level_0_metrics(analyses: list[dict]) -> dict:
    """Compute the three Level-0 metrics: ICR, ISSR, ACR."""
    total = len(analyses)

    # ICR: Intent Capture Rate
    intent_captured_count = sum(
        1 for a in analyses
        if a.get("funnel", {}).get("intent_captured", False)
    )
    icr = safe_rate(intent_captured_count, total)

    # ISSR: In-Scope Success Rate
    in_scope = [a for a in analyses if a.get("coverage") == "IN-SCOPE"]
    in_scope_success = sum(
        1 for a in in_scope if a.get("outcome") == "SUCCESS"
    )
    issr = safe_rate(in_scope_success, len(in_scope))

    # ACR: Action Completion Rate
    total_claimed = sum(len(a.get("actions_claimed", [])) for a in analyses)
    total_completed = sum(len(a.get("actions_with_completion_evidence", [])) for a in analyses)
    acr = safe_rate(total_completed, total_claimed)

    return {
        "icr": {
            "value": icr,
            "description": "Intent Capture Rate",
            "numerator": intent_captured_count,
            "denominator": total
        },
        "issr": {
            "value": issr,
            "description": "In-Scope Success Rate",
            "numerator": in_scope_success,
            "denominator": len(in_scope)
        },
        "acr": {
            "value": acr,
            "description": "Action Completion Rate",
            "numerator": total_completed,
            "denominator": total_claimed
        }
    }


def compute_funnel_analysis(analyses: list[dict]) -> dict:
    """Compute funnel stage metrics."""
    total = len(analyses)

    # Funnel stages
    stages = {
        "connect_greet": 0,
        "intent_captured": 0,
        "verification_attempted": 0,
        "verification_successful": 0,
        "solution_path_entered": 0
    }

    for a in analyses:
        funnel = a.get("funnel", {})
        for stage in stages:
            if funnel.get(stage):
                stages[stage] += 1

    # Closure type distribution
    closure_types = Counter(
        a.get("funnel", {}).get("closure_type", "unknown")
        for a in analyses
    )

    # Calculate drop-off rates
    funnel_flow = ["connect_greet", "intent_captured", "verification_attempted", "verification_successful", "solution_path_entered"]
    drop_offs = {}
    for i in range(len(funnel_flow) - 1):
        current = stages[funnel_flow[i]]
        next_stage = stages[funnel_flow[i + 1]]
        drop_offs[f"{funnel_flow[i]}_to_{funnel_flow[i+1]}"] = {
            "from": current,
            "to": next_stage,
            "drop_off_rate": safe_rate(current - next_stage, current) if current > 0 else None
        }

    return {
        "stage_counts": {k: {"count": v, "rate": safe_rate(v, total)} for k, v in stages.items()},
        "closure_type_distribution": dict(closure_types),
        "drop_off_analysis": drop_offs
    }


def compute_coverage_matrix(analyses: list[dict]) -> dict:
    """Compute coverage and outcome distributions."""
    total = len(analyses)

    # Coverage distribution
    coverage_counts = Counter(a.get("coverage", "UNKNOWN") for a in analyses)

    # Outcome distribution
    outcome_counts = Counter(a.get("outcome", "UNKNOWN") for a in analyses)

    # Cross-tabulation: coverage x outcome
    matrix = defaultdict(lambda: defaultdict(int))
    for a in analyses:
        cov = a.get("coverage", "UNKNOWN")
        out = a.get("outcome", "UNKNOWN")
        matrix[cov][out] += 1

    return {
        "coverage_distribution": {
            k: {"count": v, "rate": safe_rate(v, total)}
            for k, v in coverage_counts.items()
        },
        "outcome_distribution": {
            k: {"count": v, "rate": safe_rate(v, total)}
            for k, v in outcome_counts.items()
        },
        "coverage_outcome_matrix": {
            cov: dict(outcomes) for cov, outcomes in matrix.items()
        }
    }


def compute_verification_metrics(analyses: list[dict]) -> dict:
    """Compute verification-related metrics."""
    # Filter to calls where verification was attempted
    verification_attempted = [
        a for a in analyses
        if a.get("funnel", {}).get("verification_attempted", False)
    ]

    successful = sum(
        1 for a in verification_attempted
        if a.get("funnel", {}).get("verification_successful", False)
    )

    return {
        "total_attempted": len(verification_attempted),
        "successful": successful,
        "failed": len(verification_attempted) - successful,
        "success_rate": safe_rate(successful, len(verification_attempted))
    }


def compute_escalation_metrics(analyses: list[dict]) -> dict:
    """Compute human escalation metrics."""
    total = len(analyses)

    # Escalation requests
    requested = [a for a in analyses if a.get("human_escalation_requested", False)]

    # Escalation honored (only among those who requested)
    honored = sum(
        1 for a in requested
        if a.get("human_escalation_honored", False)
    )

    return {
        "total_calls": total,
        "escalation_requested": len(requested),
        "escalation_request_rate": safe_rate(len(requested), total),
        "escalation_honored": honored,
        "escalation_honor_rate": safe_rate(honored, len(requested))
    }


def compute_issue_breakdown(analyses: list[dict]) -> dict:
    """Compute issue frequency breakdown."""
    # Flatten all issues
    all_issues = []
    for a in analyses:
        all_issues.extend(a.get("issues", []))

    # Count by type
    type_counts = Counter(issue.get("type", "unknown") for issue in all_issues)

    # Count by severity
    severity_counts = Counter(issue.get("severity", "unknown") for issue in all_issues)

    # Calls with issues
    calls_with_issues = sum(1 for a in analyses if a.get("issues", []))

    # High severity issues
    high_severity = [i for i in all_issues if i.get("severity") == "high"]

    return {
        "total_issues": len(all_issues),
        "calls_with_issues": calls_with_issues,
        "calls_with_issues_rate": safe_rate(calls_with_issues, len(analyses)),
        "issues_per_call_avg": round(len(all_issues) / len(analyses), 2) if analyses else 0,
        "by_type": dict(type_counts),
        "by_severity": dict(severity_counts),
        "high_severity_issues": [
            {"type": i.get("type"), "description": i.get("description")}
            for i in high_severity[:10]  # Limit to top 10
        ]
    }


def compute_action_analysis(analyses: list[dict]) -> dict:
    """Compute action claim and completion analysis."""
    # Flatten all actions
    all_claimed = []
    all_completed = []
    for a in analyses:
        all_claimed.extend(a.get("actions_claimed", []))
        all_completed.extend(a.get("actions_with_completion_evidence", []))

    # Count claimed actions by type
    claimed_counts = Counter(all_claimed)
    completed_counts = Counter(all_completed)

    # Completion rates by action type
    completion_by_type = {}
    for action, claimed in claimed_counts.items():
        completed = completed_counts.get(action, 0)
        completion_by_type[action] = {
            "claimed": claimed,
            "completed": completed,
            "completion_rate": safe_rate(completed, claimed)
        }

    return {
        "total_actions_claimed": len(all_claimed),
        "total_actions_completed": len(all_completed),
        "unique_action_types": len(claimed_counts),
        "by_action_type": completion_by_type
    }


def compute_duration_metrics(analyses: list[dict]) -> dict:
    """Compute call duration proxy metrics."""
    line_counts = [a.get("call_duration_proxy", 0) for a in analyses]
    turn_counts = [a.get("turn_count", 0) for a in analyses]

    # Filter out zero values
    line_counts = [c for c in line_counts if c > 0]
    turn_counts = [c for c in turn_counts if c > 0]

    return {
        "line_count_stats": {
            "min": min(line_counts) if line_counts else 0,
            "max": max(line_counts) if line_counts else 0,
            "mean": round(statistics.mean(line_counts), 1) if line_counts else 0,
            "median": round(statistics.median(line_counts), 1) if line_counts else 0,
            "stdev": round(statistics.stdev(line_counts), 1) if len(line_counts) > 1 else 0
        },
        "turn_count_stats": {
            "min": min(turn_counts) if turn_counts else 0,
            "max": max(turn_counts) if turn_counts else 0,
            "mean": round(statistics.mean(turn_counts), 1) if turn_counts else 0,
            "median": round(statistics.median(turn_counts), 1) if turn_counts else 0,
            "stdev": round(statistics.stdev(turn_counts), 1) if len(turn_counts) > 1 else 0
        }
    }


def generate_report(analyses: list[dict]) -> dict:
    """Generate complete metrics report."""
    if not analyses:
        return {"error": "No analyses to process"}

    return {
        "summary": {
            "total_calls_analyzed": len(analyses),
            "analysis_timestamp": datetime.now().isoformat(),
            "schema_version": "v0",
            "call_ids": [a.get("call_id") for a in analyses]
        },
        "level_0_metrics": compute_level_0_metrics(analyses),
        "funnel_analysis": compute_funnel_analysis(analyses),
        "coverage_matrix": compute_coverage_matrix(analyses),
        "verification_metrics": compute_verification_metrics(analyses),
        "escalation_metrics": compute_escalation_metrics(analyses),
        "issue_breakdown": compute_issue_breakdown(analyses),
        "action_analysis": compute_action_analysis(analyses),
        "duration_metrics": compute_duration_metrics(analyses)
    }


def print_summary(report: dict) -> None:
    """Print a human-readable summary of the report."""
    print("\n" + "=" * 60)
    print("VACATIA AI VOICE AGENT - ANALYTICS REPORT (v0)")
    print("=" * 60)

    summary = report.get("summary", {})
    print(f"\nAnalyzed: {summary.get('total_calls_analyzed', 0)} calls")
    print(f"Generated: {summary.get('analysis_timestamp', 'N/A')}")

    # Level-0 Metrics
    print("\n" + "-" * 40)
    print("LEVEL-0 METRICS")
    print("-" * 40)
    l0 = report.get("level_0_metrics", {})
    for key, data in l0.items():
        value = data.get("value")
        desc = data.get("description", key.upper())
        num = data.get("numerator", 0)
        den = data.get("denominator", 0)
        pct = f"{value * 100:.1f}%" if value is not None else "N/A"
        print(f"{desc}: {pct} ({num}/{den})")

    # Coverage
    print("\n" + "-" * 40)
    print("COVERAGE DISTRIBUTION")
    print("-" * 40)
    cov = report.get("coverage_matrix", {}).get("coverage_distribution", {})
    for coverage, data in sorted(cov.items()):
        count = data.get("count", 0)
        rate = data.get("rate")
        pct = f"{rate * 100:.1f}%" if rate else "N/A"
        print(f"  {coverage}: {count} ({pct})")

    # Outcomes
    print("\n" + "-" * 40)
    print("OUTCOME DISTRIBUTION")
    print("-" * 40)
    out = report.get("coverage_matrix", {}).get("outcome_distribution", {})
    for outcome, data in sorted(out.items()):
        count = data.get("count", 0)
        rate = data.get("rate")
        pct = f"{rate * 100:.1f}%" if rate else "N/A"
        print(f"  {outcome}: {count} ({pct})")

    # Verification
    print("\n" + "-" * 40)
    print("VERIFICATION METRICS")
    print("-" * 40)
    ver = report.get("verification_metrics", {})
    print(f"  Attempted: {ver.get('total_attempted', 0)}")
    print(f"  Successful: {ver.get('successful', 0)}")
    rate = ver.get("success_rate")
    pct = f"{rate * 100:.1f}%" if rate else "N/A"
    print(f"  Success Rate: {pct}")

    # Escalation
    print("\n" + "-" * 40)
    print("ESCALATION METRICS")
    print("-" * 40)
    esc = report.get("escalation_metrics", {})
    print(f"  Requested: {esc.get('escalation_requested', 0)}")
    print(f"  Honored: {esc.get('escalation_honored', 0)}")
    rate = esc.get("escalation_honor_rate")
    pct = f"{rate * 100:.1f}%" if rate else "N/A"
    print(f"  Honor Rate: {pct}")

    # Top Issues
    print("\n" + "-" * 40)
    print("ISSUE BREAKDOWN")
    print("-" * 40)
    issues = report.get("issue_breakdown", {})
    print(f"  Total Issues: {issues.get('total_issues', 0)}")
    print(f"  Calls with Issues: {issues.get('calls_with_issues', 0)}")
    print("  By Type:")
    for issue_type, count in sorted(issues.get("by_type", {}).items(), key=lambda x: -x[1]):
        print(f"    {issue_type}: {count}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Compute aggregate metrics from analysis JSON files (v0)"
    )
    parser.add_argument(
        "-i", "--input-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "analyses",
        help="Directory containing analysis JSON files"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent.parent / "reports",
        help="Output directory for report"
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output only JSON, no console summary"
    )

    args = parser.parse_args()

    # Load analyses
    print(f"Loading analyses from: {args.input_dir}", file=sys.stderr)
    analyses = load_analyses(args.input_dir)
    print(f"Loaded {len(analyses)} analyses", file=sys.stderr)

    if not analyses:
        print("Error: No analysis files found", file=sys.stderr)
        return 1

    # Generate report
    report = generate_report(analyses)

    # Save report
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = args.output_dir / f"metrics_report_v0_{timestamp}.json"

    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"Report saved to: {report_path}", file=sys.stderr)

    # Output
    if args.json_only:
        print(json.dumps(report, indent=2, ensure_ascii=False))
    else:
        print_summary(report)
        print(f"\nFull report: {report_path}")

    return 0


if __name__ == "__main__":
    exit(main())
