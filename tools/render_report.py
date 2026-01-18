#!/usr/bin/env python3
"""
Markdown Report Renderer for Vacatia AI Voice Agent Analytics (v3)

Renders the combined Section A + Section B report as an executive-ready Markdown document.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def get_assessment_emoji(value: float, metric_type: str) -> str:
    """Get assessment emoji based on metric value and type."""
    if metric_type == "success_rate":
        if value >= 0.7:
            return "✅"
        elif value >= 0.5:
            return "⚠️"
        else:
            return "❌"
    elif metric_type == "escalation_rate":
        if value <= 0.15:
            return "✅"
        elif value <= 0.25:
            return "⚠️"
        else:
            return "❌"
    elif metric_type == "customer_effort":
        if value <= 2.5:
            return "✅"
        elif value <= 3.5:
            return "⚠️"
        else:
            return "❌"
    elif metric_type == "failure_rate":
        if value <= 0.3:
            return "✅"
        elif value <= 0.5:
            return "⚠️"
        else:
            return "❌"
    return ""


def render_markdown(report: dict) -> str:
    """Render the full report as Markdown."""
    metrics = report.get("deterministic_metrics", {})
    insights = report.get("llm_insights", {})

    metadata = metrics.get("metadata", {})
    total_calls = metadata.get("total_calls_analyzed", 0)
    generated = metadata.get("report_generated", datetime.now().isoformat())

    lines = []

    # Header
    lines.append("# Vacatia AI Voice Agent Performance Report")
    lines.append(f"**Generated:** {generated[:19].replace('T', ' ')} | **Calls Analyzed:** {total_calls}")
    lines.append("")

    # Executive Summary
    exec_summary = insights.get("executive_summary", "No executive summary available.")
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(exec_summary)
    lines.append("")

    # Key Metrics at a Glance
    lines.append("## Key Metrics at a Glance")
    lines.append("")
    lines.append("| Metric | Value | Assessment |")
    lines.append("|--------|-------|------------|")

    key_rates = metrics.get("key_rates", {})
    quality_scores = metrics.get("quality_scores", {})

    success_rate = key_rates.get("success_rate", 0)
    escalation_rate = key_rates.get("escalation_rate", 0)
    failure_rate = key_rates.get("failure_rate", 0)
    customer_effort = quality_scores.get("customer_effort", {}).get("mean", 0) or 0

    lines.append(f"| Success Rate | {success_rate*100:.1f}% | {get_assessment_emoji(success_rate, 'success_rate')} |")
    lines.append(f"| Escalation Rate | {escalation_rate*100:.1f}% | {get_assessment_emoji(escalation_rate, 'escalation_rate')} |")
    lines.append(f"| Failure Rate | {failure_rate*100:.1f}% | {get_assessment_emoji(failure_rate, 'failure_rate')} |")
    lines.append(f"| Customer Effort | {customer_effort:.2f}/5 | {get_assessment_emoji(customer_effort, 'customer_effort')} |")
    lines.append("")

    # Why Calls Are Failing
    lines.append("## Why Calls Are Failing")
    lines.append("")

    rca = insights.get("root_cause_analysis", {})
    primary_driver = rca.get("primary_driver", "Unknown")
    lines.append(f"**Primary Driver:** {primary_driver}")
    lines.append("")

    factors = rca.get("contributing_factors", [])
    if factors:
        lines.append("**Contributing Factors:**")
        for f in factors:
            lines.append(f"- {f}")
        lines.append("")

    evidence = rca.get("evidence", "")
    if evidence:
        lines.append(f"*Evidence: {evidence}*")
        lines.append("")

    # Failure Point Breakdown
    failure_analysis = metrics.get("failure_analysis", {})
    by_failure_point = failure_analysis.get("by_failure_point", {})

    if by_failure_point:
        lines.append("### Failure Point Breakdown")
        lines.append("")
        lines.append("| Failure Type | Count | % of Failures |")
        lines.append("|--------------|-------|---------------|")
        for fp, data in by_failure_point.items():
            rate = data.get("rate", 0) or 0
            lines.append(f"| {fp} | {data['count']} | {rate*100:.1f}% |")
        lines.append("")

    # Policy Gap Breakdown
    pgb = metrics.get("policy_gap_breakdown", {})
    by_category = pgb.get("by_category", {})

    if by_category:
        lines.append("### Policy Gap Breakdown")
        lines.append("")
        lines.append("| Category | Count | % of Gaps |")
        lines.append("|----------|-------|-----------|")
        for cat, data in by_category.items():
            rate = data.get("rate", 0) or 0
            lines.append(f"| {cat} | {data['count']} | {rate*100:.1f}% |")
        lines.append("")

    # Top Unmet Customer Needs
    top_asks = pgb.get("top_customer_asks", [])
    if top_asks:
        lines.append("### Top Unmet Customer Needs")
        lines.append("")
        for i, item in enumerate(top_asks[:5], 1):
            lines.append(f"{i}. **{item['ask']}** - {item['count']} calls")
        lines.append("")

    # Recommendations
    lines.append("## Recommendations")
    lines.append("")

    recs = insights.get("actionable_recommendations", [])
    p0_recs = [r for r in recs if r.get("priority") == "P0"]
    p1_recs = [r for r in recs if r.get("priority") == "P1"]
    p2_recs = [r for r in recs if r.get("priority") == "P2"]

    if p0_recs:
        lines.append("### P0 - Critical")
        lines.append("")
        for rec in p0_recs:
            lines.append(f"**{rec.get('recommendation', 'N/A')}**")
            lines.append(f"- Category: {rec.get('category', 'N/A')}")
            lines.append(f"- Expected Impact: {rec.get('expected_impact', 'N/A')}")
            lines.append(f"- Evidence: {rec.get('evidence', 'N/A')}")
            lines.append("")

    if p1_recs:
        lines.append("### P1 - High Priority")
        lines.append("")
        for rec in p1_recs:
            lines.append(f"**{rec.get('recommendation', 'N/A')}**")
            lines.append(f"- Category: {rec.get('category', 'N/A')}")
            lines.append(f"- Expected Impact: {rec.get('expected_impact', 'N/A')}")
            lines.append("")

    if p2_recs:
        lines.append("### P2 - Moderate Priority")
        lines.append("")
        for rec in p2_recs:
            lines.append(f"- {rec.get('recommendation', 'N/A')} ({rec.get('category', 'N/A')})")
        lines.append("")

    # Customer Voice
    lines.append("## Customer Voice")
    lines.append("")

    verbatims = insights.get("verbatim_highlights", {})

    if verbatims.get("most_frustrated"):
        lines.append("### Most Frustrated")
        lines.append(f'> "{verbatims["most_frustrated"]}"')
        lines.append("")

    if verbatims.get("most_common_ask"):
        lines.append("### Most Common Ask")
        lines.append(f'> "{verbatims["most_common_ask"]}"')
        lines.append("")

    if verbatims.get("biggest_miss"):
        lines.append("### Biggest Agent Miss")
        lines.append(f'> "{verbatims["biggest_miss"]}"')
        lines.append("")

    # Trend Narratives
    narratives = insights.get("trend_narratives", {})
    if narratives:
        lines.append("## Trend Analysis")
        lines.append("")

        if narratives.get("failure_patterns"):
            lines.append("### Failure Patterns")
            lines.append(narratives["failure_patterns"])
            lines.append("")

        if narratives.get("customer_experience"):
            lines.append("### Customer Experience")
            lines.append(narratives["customer_experience"])
            lines.append("")

        if narratives.get("agent_performance"):
            lines.append("### Agent Performance")
            lines.append(narratives["agent_performance"])
            lines.append("")

    # Quality Scores Detail
    lines.append("## Quality Scores")
    lines.append("")
    lines.append("| Score | Mean | Median | Std Dev | n |")
    lines.append("|-------|------|--------|---------|---|")

    for name, stats in quality_scores.items():
        label = name.replace("_", " ").title()
        mean = stats.get("mean") or "N/A"
        median = stats.get("median") or "N/A"
        std = stats.get("std") or "N/A"
        n = stats.get("n", 0)
        if isinstance(mean, (int, float)):
            lines.append(f"| {label} | {mean:.2f} | {median:.2f} | {std:.2f} | {n} |")
        else:
            lines.append(f"| {label} | {mean} | {median} | {std} | {n} |")
    lines.append("")

    # Training Priorities
    training = metrics.get("training_priorities", {})
    if training:
        lines.append("## Training Priorities")
        lines.append("")
        lines.append("| Skill Gap | Count |")
        lines.append("|-----------|-------|")
        for skill, count in training.items():
            lines.append(f"| {skill} | {count} |")
        lines.append("")

    # Actionable Flags
    flags = metrics.get("actionable_flags", {})
    if flags:
        lines.append("## Actionable Flags")
        lines.append("")
        lines.append("| Flag | Count | Rate |")
        lines.append("|------|-------|------|")
        for flag, data in flags.items():
            label = flag.replace("_", " ").title()
            rate = data.get("rate", 0) or 0
            lines.append(f"| {label} | {data['count']} | {rate*100:.1f}% |")
        lines.append("")

    # Appendix: Detailed Metrics
    lines.append("## Appendix: Detailed Metrics")
    lines.append("")
    lines.append("<details>")
    lines.append("<summary>Click to expand full metrics JSON</summary>")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(metrics, indent=2))
    lines.append("```")
    lines.append("")
    lines.append("</details>")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append(f"*Report generated by Vacatia AI Voice Agent Analytics Framework v3*")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Render v3 report as Markdown")
    parser.add_argument("-i", "--input", type=Path,
                        help="Path to full report JSON (with deterministic_metrics and llm_insights)")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "reports",
                        help="Output directory for Markdown report")
    parser.add_argument("--stdout", action="store_true",
                        help="Print to stdout instead of saving")

    args = parser.parse_args()

    # Find latest report file if not specified
    if not args.input:
        reports_dir = args.output_dir
        if reports_dir.exists():
            report_files = sorted(reports_dir.glob("report_v3_*.json"), reverse=True)
            if report_files:
                args.input = report_files[0]
                print(f"Using latest report: {args.input}", file=sys.stderr)

    if not args.input or not args.input.exists():
        print("Error: No report file found. Run generate_insights.py first.", file=sys.stderr)
        return 1

    # Load report
    with open(args.input, 'r', encoding='utf-8') as f:
        report = json.load(f)

    # Render Markdown
    markdown = render_markdown(report)

    if args.stdout:
        print(markdown)
    else:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = args.output_dir / f"executive_summary_v3_{timestamp}.md"

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(markdown)

        print(f"Markdown report saved: {output_path}", file=sys.stderr)
        print(f"\nReport: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
