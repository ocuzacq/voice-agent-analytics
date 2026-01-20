#!/usr/bin/env python3
"""
Markdown Report Renderer for Vacatia AI Voice Agent Analytics (v3.9.1)

Renders the combined Section A + Section B report as an executive-ready Markdown document.

v3.9.1 additions:
- Loop Subject Analysis: Table showing subject distribution per loop type
- High-Impact Patterns: LLM-generated insights on (loop_type, subject) combinations
- Subject clustering narrative
- Custom Analysis: Dedicated section for user-provided questions and LLM answers

v3.9 additions:
- Call Disposition Breakdown: Table showing distribution of call_disposition values
- Funnel Metrics: In-scope success rate, out-of-scope recovery rate, pre-intent rate
- Disposition Analysis: LLM-generated insights on disposition patterns

v3.8.5 additions:
- Backwards-compatible with both v3.8.5 (compact friction) and v3.8 formats
- All friction data parsed via compute_metrics.py helper functions
- Version references updated to v3.8.5

v3.8 additions:
- Agent Loops: Replaces repeated prompts with typed loop detection
- Loop Type Breakdown: Distribution by type (info_retry, intent_retry, deflection, etc.)
- Loop Type Analysis: LLM insights on loop patterns and recommendations

v3.7 additions:
- Clarification Cause Breakdown: Distribution by cause type (customer_refused, agent_misheard, etc.)
- Correction Severity Breakdown: Distribution by severity (minor, moderate, major)
- Cause/severity analysis insights from LLM

v3.6 additions:
- Conversation Quality: Dedicated section with turn stats, clarification friction, user corrections, loops
- Friction hotspots table with impact and recommendations
- Turn analysis insights

v3.5 additions:
- Training & Development: Narrative-first section with priorities, root causes, and actions
- Cross-Dimensional Patterns: Training gaps correlated with failures
- Emergent Patterns: LLM-discovered patterns not in standard categories
- Secondary Customer Needs: Clustered additional intents

v3.4 additions:
- Inline descriptions in 4th column (Context) for Key Metrics, Failure Types, Policy Gaps
- Sub-breakdowns for major failure types (≥5% of failures)

v3.3 additions:
- Clustered customer asks with semantic grouping
- Explanatory qualifiers for failure types
- Explanatory qualifiers for policy gap categories
- Supporting call_ids in recommendations
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

    # v3.9.1: Custom Analysis (user-provided questions)
    custom_analysis = report.get("custom_analysis", [])
    if custom_analysis:
        lines.append("## Custom Analysis")
        lines.append("")
        lines.append("*Answers to user-provided analytical questions:*")
        lines.append("")

        for i, qa in enumerate(custom_analysis, 1):
            question = qa.get("question", "N/A")
            answer = qa.get("answer", "N/A")
            confidence = qa.get("confidence", "medium")
            evidence = qa.get("evidence", [])

            lines.append(f"### {i}. {question}")
            lines.append("")
            lines.append(f"**Confidence:** {confidence.upper()}")
            lines.append("")
            lines.append(answer)
            lines.append("")

            if evidence:
                lines.append("**Supporting Evidence:**")
                for e in evidence:
                    lines.append(f"- {e}")
                lines.append("")

    # Key Metrics at a Glance
    lines.append("## Key Metrics at a Glance")
    lines.append("")
    lines.append("| Metric | Value | Assessment | Context |")
    lines.append("|--------|-------|------------|---------|")

    key_rates = metrics.get("key_rates", {})
    quality_scores = metrics.get("quality_scores", {})
    key_metrics_desc = insights.get("key_metrics_descriptions", {})

    success_rate = key_rates.get("success_rate", 0)
    escalation_rate = key_rates.get("escalation_rate", 0)
    failure_rate = key_rates.get("failure_rate", 0)
    customer_effort = quality_scores.get("customer_effort", {}).get("mean", 0) or 0

    lines.append(f"| Success Rate | {success_rate*100:.1f}% | {get_assessment_emoji(success_rate, 'success_rate')} | {key_metrics_desc.get('success_rate', '')} |")
    lines.append(f"| Escalation Rate | {escalation_rate*100:.1f}% | {get_assessment_emoji(escalation_rate, 'escalation_rate')} | {key_metrics_desc.get('escalation_rate', '')} |")
    lines.append(f"| Failure Rate | {failure_rate*100:.1f}% | {get_assessment_emoji(failure_rate, 'failure_rate')} | {key_metrics_desc.get('failure_rate', '')} |")
    lines.append(f"| Customer Effort | {customer_effort:.2f}/5 | {get_assessment_emoji(customer_effort, 'customer_effort')} | {key_metrics_desc.get('customer_effort', '')} |")
    lines.append("")

    # v3.9: Call Disposition Breakdown
    disposition_breakdown = metrics.get("disposition_breakdown", {})
    disp_analysis = insights.get("disposition_analysis", {})
    by_disposition = disposition_breakdown.get("by_disposition", {})

    if by_disposition:
        lines.append("## Call Disposition Breakdown")
        lines.append("")

        # Narrative from LLM
        if disp_analysis.get("narrative"):
            lines.append(disp_analysis["narrative"])
            lines.append("")

        lines.append("| Disposition | Count | % |")
        lines.append("|-------------|-------|---|")

        # Order dispositions in funnel order
        disposition_order = [
            "in_scope_success", "in_scope_partial", "in_scope_failed",
            "out_of_scope_handled", "out_of_scope_abandoned", "pre_intent", "unknown"
        ]
        for disp in disposition_order:
            if disp in by_disposition:
                data = by_disposition[disp]
                rate = data.get("rate", 0) or 0
                lines.append(f"| {disp.replace('_', ' ').title()} | {data['count']} | {rate*100:.1f}% |")

        # Any other dispositions not in the order
        for disp, data in by_disposition.items():
            if disp not in disposition_order:
                rate = data.get("rate", 0) or 0
                lines.append(f"| {disp} | {data['count']} | {rate*100:.1f}% |")

        lines.append("")

        # Funnel Metrics
        funnel_metrics = disposition_breakdown.get("funnel_metrics", {})
        if funnel_metrics:
            lines.append("### Funnel Metrics")
            lines.append("")

            in_scope_success = funnel_metrics.get("in_scope_success_rate")
            out_of_scope_recovery = funnel_metrics.get("out_of_scope_recovery_rate")
            pre_intent_rate = funnel_metrics.get("pre_intent_rate")

            if in_scope_success is not None:
                in_scope_total = funnel_metrics.get("in_scope_total", 0)
                lines.append(f"- **In-Scope Success Rate:** {in_scope_success*100:.1f}% ({in_scope_total} in-scope calls confirmed satisfaction)")
            if out_of_scope_recovery is not None:
                out_of_scope_total = funnel_metrics.get("out_of_scope_total", 0)
                lines.append(f"- **Out-of-Scope Recovery:** {out_of_scope_recovery*100:.1f}% ({out_of_scope_total} out-of-scope calls handled gracefully)")
            if pre_intent_rate is not None:
                lines.append(f"- **Pre-Intent Rate:** {pre_intent_rate*100:.1f}% (possible IVR/routing issues)")
            lines.append("")

        # Actionable insights from LLM
        actionable_insights = disp_analysis.get("actionable_insights", [])
        if actionable_insights:
            lines.append("### Disposition Insights")
            lines.append("")
            lines.append("| Disposition | Root Cause | Recommendation |")
            lines.append("|-------------|------------|----------------|")
            for ai in actionable_insights:
                disp = ai.get("disposition", "N/A")
                root_cause = ai.get("root_cause", "")
                rec = ai.get("recommendation", "")
                lines.append(f"| {disp.replace('_', ' ').title()} | {root_cause} | {rec} |")
            lines.append("")

        # Funnel health assessment
        funnel_health = disp_analysis.get("funnel_health", {})
        if funnel_health.get("assessment"):
            assessment = (funnel_health.get("assessment") or "").upper()
            emoji = "✅" if assessment == "HEALTHY" else "⚠️" if assessment == "NEEDS_ATTENTION" else "❌"
            lines.append(f"**Funnel Health:** {emoji} {assessment}")
            if funnel_health.get("explanation"):
                lines.append(f"_{funnel_health['explanation']}_")
            if funnel_health.get("priority_focus"):
                lines.append(f"\n**Priority Focus:** {funnel_health['priority_focus']}")
            lines.append("")

    # v3.6: Conversation Quality Section
    conv_quality = metrics.get("conversation_quality", {})
    cq_analysis = insights.get("conversation_quality_analysis", {})

    if conv_quality or cq_analysis:
        lines.append("## Conversation Quality")
        lines.append("")

        # Turn stats summary line
        turn_stats = conv_quality.get("turn_stats", {})
        if turn_stats.get("avg_turns"):
            avg = turn_stats.get("avg_turns", "N/A")
            resolved = turn_stats.get("avg_turns_resolved", "N/A")
            failed = turn_stats.get("avg_turns_failed", "N/A")
            ttf = turn_stats.get("avg_turns_to_failure", "N/A")
            lines.append(f"**Average Length:** {avg} turns | **Resolved:** {resolved} turns | **Failed:** {failed} turns | **Turns to Failure:** {ttf}")
            lines.append("")

        # v3.6: Narrative from LLM
        if cq_analysis.get("narrative"):
            lines.append(cq_analysis["narrative"])
            lines.append("")

        # Clarification Friction Table
        clar_stats = conv_quality.get("clarification_stats", {})
        if clar_stats.get("by_type"):
            lines.append("### Clarification Friction")
            lines.append("")
            lines.append("| Type | Count | % of Calls | Resolution Rate |")
            lines.append("|------|-------|------------|-----------------|")

            by_type = clar_stats.get("by_type", {})
            resolution_rate = clar_stats.get("resolution_rate")
            for ctype, data in by_type.items():
                count = data.get("count", 0)
                rate = data.get("rate", 0) or 0
                # Resolution rate is aggregate; we show it once
                lines.append(f"| {ctype.replace('_', ' ').title()} | {count} | {rate*100:.1f}% | - |")

            if resolution_rate is not None:
                lines.append("")
                lines.append(f"*Overall clarification resolution rate: {resolution_rate*100:.1f}%*")
            lines.append("")

            # v3.7: Clarification Cause Breakdown
            by_cause = clar_stats.get("by_cause", {})
            if by_cause:
                lines.append("#### By Cause (v3.7)")
                lines.append("")
                lines.append("| Cause | Count | % of Clarifications |")
                lines.append("|-------|-------|---------------------|")
                for cause, data in by_cause.items():
                    count = data.get("count", 0)
                    rate = data.get("rate", 0) or 0
                    lines.append(f"| {cause.replace('_', ' ').title()} | {count} | {rate*100:.1f}% |")
                lines.append("")

        # Friction Hotspots from LLM analysis
        friction_hotspots = cq_analysis.get("friction_hotspots", [])
        if friction_hotspots:
            lines.append("### Friction Hotspots")
            lines.append("")
            lines.append("| Pattern | Frequency | Impact | Recommendation |")
            lines.append("|---------|-----------|--------|----------------|")
            for fh in friction_hotspots:
                pattern = fh.get("pattern", "N/A")
                freq = fh.get("frequency", "N/A")
                impact = fh.get("impact", "N/A")
                rec = fh.get("recommendation", "N/A")
                lines.append(f"| {pattern} | {freq} | {impact} | {rec} |")
            lines.append("")

        # v3.7: Cause Analysis from LLM
        cause_analysis = cq_analysis.get("cause_analysis", {})
        if cause_analysis.get("insights"):
            lines.append("### Clarification Cause Analysis (v3.7)")
            lines.append("")
            if cause_analysis.get("narrative"):
                lines.append(cause_analysis["narrative"])
                lines.append("")
            lines.append("| Cause | Frequency | Correlation | Recommendation |")
            lines.append("|-------|-----------|-------------|----------------|")
            for ci in cause_analysis.get("insights", []):
                cause = ci.get("cause", "N/A")
                freq = ci.get("frequency", "N/A")
                corr = ci.get("correlation", "N/A")
                rec = ci.get("recommendation", "N/A")
                lines.append(f"| {cause.replace('_', ' ').title()} | {freq} | {corr} | {rec} |")
            lines.append("")

        # v3.7: Severity Analysis from LLM
        severity_analysis = cq_analysis.get("severity_analysis", {})
        if severity_analysis.get("insights"):
            lines.append("### Correction Severity Analysis (v3.7)")
            lines.append("")
            if severity_analysis.get("narrative"):
                lines.append(severity_analysis["narrative"])
                lines.append("")
            lines.append("| Severity | Frequency | Correlation | Recommendation |")
            lines.append("|----------|-----------|-------------|----------------|")
            for si in severity_analysis.get("insights", []):
                sev = si.get("severity", "N/A")
                freq = si.get("frequency", "N/A")
                corr = si.get("correlation", "N/A")
                rec = si.get("recommendation", "N/A")
                lines.append(f"| {sev.title()} | {freq} | {corr} | {rec} |")
            lines.append("")

        # Customer Corrections
        corr_stats = conv_quality.get("correction_stats", {})
        if corr_stats.get("calls_with_corrections"):
            lines.append("### Customer Corrections")
            lines.append("")
            calls = corr_stats.get("calls_with_corrections", 0)
            pct = corr_stats.get("pct_calls_with_corrections", 0) or 0
            frust = corr_stats.get("with_frustration_signal", 0)
            frust_rate = corr_stats.get("frustration_rate", 0) or 0
            lines.append(f"- **{calls} calls** ({pct*100:.1f}%) had customer corrections")
            lines.append(f"- **{frust}** ({frust_rate*100:.1f}%) showed frustration signals during correction")
            lines.append("")

            # v3.7: Correction Severity Breakdown
            by_severity = corr_stats.get("by_severity", {})
            if by_severity:
                lines.append("#### By Severity (v3.7)")
                lines.append("")
                lines.append("| Severity | Count | % of Corrections |")
                lines.append("|----------|-------|------------------|")
                for severity, data in by_severity.items():
                    count = data.get("count", 0)
                    rate = data.get("rate", 0) or 0
                    lines.append(f"| {severity.title()} | {count} | {rate*100:.1f}% |")
                lines.append("")

        # Agent Loops (v3.8: typed loop detection)
        loop_stats = conv_quality.get("loop_stats", {})
        if loop_stats.get("calls_with_loops"):
            lines.append("### Agent Loops")
            lines.append("")
            loops = loop_stats.get("calls_with_loops", 0)
            pct = loop_stats.get("pct_calls_with_loops", 0) or 0
            total_loops = loop_stats.get("total_loops", 0)
            avg_loops = loop_stats.get("avg_loops_per_call", 0)
            loop_density = loop_stats.get("loop_density")
            lines.append(f"- **{loops} calls** ({pct*100:.1f}%) had friction loops")
            lines.append(f"- **{total_loops} total loops** ({avg_loops} avg per affected call)")
            if loop_density:
                lines.append(f"- **Loop density:** {loop_density} loops/turn")
            lines.append("")

            # v3.8: Loop type breakdown
            by_type = loop_stats.get("by_type", {})
            if by_type:
                lines.append("#### By Type (v3.8)")
                lines.append("")
                lines.append("| Type | Count | % of Loops |")
                lines.append("|------|-------|------------|")
                for loop_type, data in by_type.items():
                    count = data.get("count", 0)
                    rate = data.get("rate", 0) or 0
                    lines.append(f"| {loop_type.replace('_', ' ').title()} | {count} | {rate*100:.1f}% |")
                lines.append("")

        # v3.8: Loop Type Analysis from LLM
        loop_analysis = cq_analysis.get("loop_type_analysis", {})
        if loop_analysis.get("insights"):
            lines.append("### Loop Type Analysis (v3.8)")
            lines.append("")
            if loop_analysis.get("narrative"):
                lines.append(loop_analysis["narrative"])
                lines.append("")
            lines.append("| Type | Frequency | Impact | Recommendation |")
            lines.append("|------|-----------|--------|----------------|")
            for li in loop_analysis.get("insights", []):
                loop_type = li.get("type", "N/A")
                freq = li.get("frequency", "N/A")
                impact = li.get("impact", "N/A")
                rec = li.get("recommendation", "N/A")
                lines.append(f"| {loop_type.replace('_', ' ').title()} | {freq} | {impact} | {rec} |")
            lines.append("")
            if loop_analysis.get("intent_retry_rate") or loop_analysis.get("deflection_rate"):
                lines.append("**Key Rates:**")
                if loop_analysis.get("intent_retry_rate"):
                    lines.append(f"- Intent Retry Rate: {loop_analysis['intent_retry_rate']}")
                if loop_analysis.get("deflection_rate"):
                    lines.append(f"- Deflection Rate: {loop_analysis['deflection_rate']}")
                lines.append("")

        # v3.9.1: Loop Subject Analysis
        loop_subject = insights.get("loop_subject_clusters", {})
        loop_subject_stats = loop_stats.get("by_subject", {})

        if loop_subject or loop_subject_stats:
            lines.append("### Loop Subject Analysis (v3.9.1)")
            lines.append("")

            if loop_subject.get("narrative"):
                lines.append(loop_subject["narrative"])
                lines.append("")

            # Show subject breakdown from deterministic metrics
            if loop_subject_stats:
                lines.append("#### Subject Breakdown by Loop Type")
                lines.append("")
                for loop_type, subjects in loop_subject_stats.items():
                    if subjects:
                        lines.append(f"**{loop_type.replace('_', ' ').title()}:**")
                        lines.append("")
                        lines.append("| Subject | Count | % |")
                        lines.append("|---------|-------|---|")
                        for subj, data in list(subjects.items())[:5]:
                            count = data.get("count", 0)
                            rate = data.get("rate", 0) or 0
                            lines.append(f"| {subj} | {count} | {rate*100:.1f}% |")
                        lines.append("")

            # Show high-impact patterns from LLM insights
            high_impact = loop_subject.get("high_impact_patterns", [])
            if high_impact:
                lines.append("#### High-Impact Patterns")
                lines.append("")
                lines.append("| Loop Type | Subject | Impact | Recommendation |")
                lines.append("|-----------|---------|--------|----------------|")
                for p in high_impact:
                    lt = p.get("loop_type", "N/A").replace("_", " ").title()
                    subj = p.get("subject", "N/A")
                    impact = p.get("impact", "N/A")
                    rec = p.get("recommendation", "N/A")
                    lines.append(f"| {lt} | {subj} | {impact} | {rec} |")
                lines.append("")

        # Efficiency Insights
        efficiency_insights = cq_analysis.get("efficiency_insights", [])
        if efficiency_insights:
            lines.append("### Efficiency Insights")
            lines.append("")
            for ei in efficiency_insights:
                lines.append(f"- {ei}")
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
        lines.append("| Failure Type | Count | % of Failures | Context |")
        lines.append("|--------------|-------|---------------|---------|")
        fp_desc = insights.get("failure_type_descriptions", {})
        for fp, data in by_failure_point.items():
            rate = data.get("rate", 0) or 0
            desc = fp_desc.get(fp, "")
            lines.append(f"| {fp} | {data['count']} | {rate*100:.1f}% | {desc} |")
        lines.append("")

        # v3.4: Render sub-breakdowns for major failure types (≥5% of failures)
        total_failures = sum(d.get("count", 0) for d in by_failure_point.values())
        threshold = total_failures * 0.05  # 5% of total failures
        major_breakdowns = insights.get("major_failure_breakdowns", {})
        for fp_type, breakdown in major_breakdowns.items():
            if fp_type in by_failure_point and by_failure_point[fp_type].get("count", 0) >= threshold:
                lines.append(f"#### {fp_type.replace('_', ' ').title()} Breakdown")
                lines.append("")
                lines.append("| Pattern | Count | Description |")
                lines.append("|---------|-------|-------------|")
                for p in breakdown.get("patterns", []):
                    lines.append(f"| {p.get('pattern', '')} | {p.get('count', 0)} | {p.get('description', '')} |")
                lines.append("")

    # Policy Gap Breakdown
    pgb = metrics.get("policy_gap_breakdown", {})
    by_category = pgb.get("by_category", {})

    if by_category:
        lines.append("### Policy Gap Breakdown")
        lines.append("")
        lines.append("| Category | Count | % of Gaps | Context |")
        lines.append("|----------|-------|-----------|---------|")
        pg_desc = insights.get("policy_gap_descriptions", {})
        for cat, data in by_category.items():
            rate = data.get("rate", 0) or 0
            desc = pg_desc.get(cat, "")
            lines.append(f"| {cat} | {data['count']} | {rate*100:.1f}% | {desc} |")
        lines.append("")

    # Top Unmet Customer Needs - v3.3: Use clustered asks when available
    customer_ask_clusters = insights.get("customer_ask_clusters", [])
    top_asks = pgb.get("top_customer_asks", [])

    if customer_ask_clusters:
        lines.append("### Top Unmet Customer Needs (Clustered)")
        lines.append("")
        for i, cluster in enumerate(customer_ask_clusters[:5], 1):
            label = cluster.get("canonical_label", "Unknown")
            count = cluster.get("total_count", 0)
            member_asks = cluster.get("member_asks", [])
            lines.append(f"{i}. **{label}** - {count} calls")
            if member_asks:
                examples = ", ".join(f'"{a}"' for a in member_asks[:3])
                lines.append(f"   *Examples: {examples}*")
        lines.append("")
    elif top_asks:
        # Fallback to raw asks
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
            # v3.3: Add supporting call_ids
            call_ids = rec.get("supporting_call_ids", [])
            if call_ids:
                lines.append(f"- Example calls: {', '.join(call_ids[:5])}")
            lines.append("")

    if p1_recs:
        lines.append("### P1 - High Priority")
        lines.append("")
        for rec in p1_recs:
            lines.append(f"**{rec.get('recommendation', 'N/A')}**")
            lines.append(f"- Category: {rec.get('category', 'N/A')}")
            lines.append(f"- Expected Impact: {rec.get('expected_impact', 'N/A')}")
            # v3.3: Add supporting call_ids
            call_ids = rec.get("supporting_call_ids", [])
            if call_ids:
                lines.append(f"- Example calls: {', '.join(call_ids[:5])}")
            lines.append("")

    if p2_recs:
        lines.append("### P2 - Moderate Priority")
        lines.append("")
        for rec in p2_recs:
            rec_text = f"- {rec.get('recommendation', 'N/A')} ({rec.get('category', 'N/A')})"
            # v3.3: Add supporting call_ids inline for P2
            call_ids = rec.get("supporting_call_ids", [])
            if call_ids:
                rec_text += f" - calls: {', '.join(call_ids[:3])}"
            lines.append(rec_text)
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

    # v3.5: Secondary Customer Needs (after Customer Voice)
    secondary = insights.get("secondary_intents_analysis", {})
    if secondary.get("clusters"):
        lines.append("## Secondary Customer Needs")
        lines.append("")
        if secondary.get("narrative"):
            lines.append(secondary["narrative"])
            lines.append("")
        lines.append("| Need | Count | Implication |")
        lines.append("|------|-------|-------------|")
        for c in secondary.get("clusters", [])[:5]:
            cluster = c.get("cluster", "N/A")
            count = c.get("count", 0)
            implication = c.get("implication", "")
            lines.append(f"| {cluster} | {count} | {implication} |")
        lines.append("")

    # v3.5: Emergent Patterns
    emergent = insights.get("emergent_patterns", [])
    if emergent:
        lines.append("## Emergent Patterns")
        lines.append("")
        lines.append("*Patterns discovered that don't fit standard categories:*")
        lines.append("")
        for p in emergent:
            name = p.get("name", "Unnamed")
            frequency = p.get("frequency", "Unknown")
            significance = p.get("significance", "")
            description = p.get("description", "")
            example_ids = p.get("example_call_ids", [])

            lines.append(f"### {name}")
            lines.append(f"**Frequency:** {frequency}")
            lines.append("")
            if significance:
                lines.append(f"**Significance:** {significance}")
                lines.append("")
            if description:
                lines.append(f"_{description}_")
                lines.append("")
            if example_ids:
                lines.append(f"Examples: {', '.join(example_ids[:3])}")
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

    # Training & Development (v3.5 - narrative first with priorities)
    training = metrics.get("training_priorities", {})
    training_analysis = insights.get("training_analysis", {})

    if training or training_analysis:
        lines.append("## Training & Development")
        lines.append("")

        # v3.5: Narrative first
        if training_analysis.get("narrative"):
            lines.append(training_analysis["narrative"])
            lines.append("")

        # Priority table with context (v3.5)
        top_priorities = training_analysis.get("top_priorities", [])
        if top_priorities:
            lines.append("### Priority Skills")
            lines.append("")
            lines.append("| Skill | Count | Root Cause | Recommended Action |")
            lines.append("|-------|-------|------------|-------------------|")
            for p in top_priorities[:5]:
                skill = p.get("skill", "N/A")
                count = p.get("count", 0)
                why = p.get("why", "")
                action = p.get("action", "")
                lines.append(f"| {skill} | {count} | {why} | {action} |")
            lines.append("")
        elif training:
            # Fallback to simple table if no LLM analysis
            lines.append("### Skill Gaps")
            lines.append("")
            lines.append("| Skill Gap | Count |")
            lines.append("|-----------|-------|")
            for skill, count in training.items():
                lines.append(f"| {skill} | {count} |")
            lines.append("")

        # v3.5: Cross-correlations
        correlations = training_analysis.get("cross_correlations", [])
        if correlations:
            lines.append("### Cross-Dimensional Patterns")
            lines.append("")
            for c in correlations:
                pattern = c.get("pattern", "N/A")
                count = c.get("count", 0)
                insight = c.get("insight", "")
                lines.append(f"- **{pattern}** ({count} calls): {insight}")
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
    lines.append(f"*Report generated by Vacatia AI Voice Agent Analytics Framework v3.9.1*")

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
