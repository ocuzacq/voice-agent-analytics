#!/usr/bin/env python3
"""
LLM Insights Generator for Vacatia AI Voice Agent Analytics (v3.1)

Generates Section B: LLM-powered insights by passing Section A metrics
and the condensed NL summary (from extract_nl_fields.py) to Gemini.

Primary input: nl_summary_v3_{timestamp}.json (from extract_nl_fields.py)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import google.generativeai as genai


INSIGHTS_SYSTEM_PROMPT = """You are a senior call center analytics consultant. Your job is to analyze voice agent performance metrics and natural language data to provide executive-ready insights and actionable recommendations.

You will receive:
1. Section A: Deterministic metrics (calculated via Python - factual, auditable)
2. Aggregated qualitative data: failure descriptions, customer quotes, agent misses

Your task: Generate Section B insights that synthesize the data into strategic recommendations.

## Output Format

Return ONLY valid JSON matching this structure:
{
  "executive_summary": "2-3 sentence high-level takeaway for leadership",

  "root_cause_analysis": {
    "primary_driver": "string (single biggest issue causing failures)",
    "contributing_factors": ["string", "string"],
    "evidence": "string (supporting data points from metrics)"
  },

  "actionable_recommendations": [
    {
      "priority": "P0 | P1 | P2",
      "category": "capability | training | prompt | process",
      "recommendation": "string (specific action to take)",
      "expected_impact": "string (e.g., 'Could resolve 18% of failures')",
      "evidence": "string (why this matters)"
    }
  ],

  "trend_narratives": {
    "failure_patterns": "string (narrative about what's failing and why)",
    "customer_experience": "string (narrative about customer friction points)",
    "agent_performance": "string (narrative about agent behavior patterns)"
  },

  "verbatim_highlights": {
    "most_frustrated": "string (worst customer quote showing pain)",
    "most_common_ask": "string (recurring unmet customer need)",
    "biggest_miss": "string (most impactful missed agent opportunity)"
  }
}

## Guidelines

1. **Be specific and actionable**: Avoid generic advice like "improve training". Instead: "Add verification bypass for returning customers to reduce 23% of auth failures"

2. **Ground in data**: Every recommendation should reference specific metrics. "Policy gaps account for 44% of failures, with capability_limit being the top category"

3. **Prioritize by impact**: P0 = critical/immediate (>20% of failures), P1 = high (10-20%), P2 = moderate (<10%)

4. **Use customer voice**: Include verbatim quotes that illustrate the problem - these are powerful for executive buy-in

5. **Connect dots**: Link failure patterns to customer experience to recommendations

6. **Be concise**: Executive summaries should be scannable in 30 seconds

Return ONLY the JSON object, no markdown code blocks or additional text.
"""


def configure_genai():
    """Configure the Google Generative AI client."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response."""
    text = text.strip()

    # Remove markdown code blocks if present
    if text.startswith('```'):
        first_newline = text.find('\n')
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON object in text
    brace_start = text.find('{')
    if brace_start != -1:
        depth = 0
        for i, char in enumerate(text[brace_start:], brace_start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i+1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Could not parse JSON from: {text[:500]}...")


def build_insights_prompt(metrics: dict, nl_summary: dict) -> str:
    """Build the prompt for LLM insight generation using v3.1 NL summary format."""

    # Format Section A metrics
    metrics_json = json.dumps(metrics, indent=2)

    # Format natural language fields from nl_summary_v3 structure
    nl_sections = []

    # Failures grouped by type (new v3.1 structure)
    if nl_summary.get("by_failure_type"):
        nl_sections.append("## Failures by Type")
        for fp_type, entries in nl_summary["by_failure_type"].items():
            nl_sections.append(f"\n### {fp_type} ({len(entries)} calls)")
            for entry in entries[:10]:  # Limit per type
                parts = [f"[{entry.get('outcome', 'unknown')}]"]
                if entry.get("description"):
                    parts.append(entry["description"])
                if entry.get("verbatim"):
                    parts.append(f'Customer: "{entry["verbatim"]}"')
                if entry.get("miss"):
                    parts.append(f'Miss: {entry["miss"]}')
                nl_sections.append(f"- {' | '.join(parts)}")

    # Customer verbatims (all quotes)
    if nl_summary.get("all_verbatims"):
        nl_sections.append("\n## Customer Verbatims (direct quotes)")
        for v in nl_summary["all_verbatims"][:15]:
            nl_sections.append(f'- [{v.get("outcome", "unknown")}] "{v.get("quote", "")}"')

    # Agent miss details
    if nl_summary.get("all_agent_misses"):
        nl_sections.append("\n## Agent Miss Details (coaching opportunities)")
        for m in nl_summary["all_agent_misses"][:15]:
            recoverable = "recoverable" if m.get("was_recoverable") else "not recoverable"
            nl_sections.append(f"- [{recoverable}] {m.get('miss', '')}")

    # Policy gap details (structured)
    if nl_summary.get("policy_gap_details"):
        nl_sections.append("\n## Policy Gap Details")
        for g in nl_summary["policy_gap_details"][:15]:
            nl_sections.append(
                f"- [{g.get('category', 'unknown')}] Gap: {g.get('gap', '')} | "
                f"Ask: {g.get('ask', '')} | Blocker: {g.get('blocker', '')}"
            )

    # Failed call flows
    if nl_summary.get("failed_call_flows"):
        nl_sections.append("\n## Sample Failed Call Flows")
        for call in nl_summary["failed_call_flows"][:5]:
            steps = call.get("steps", [])
            steps_str = " → ".join(steps[:10])
            nl_sections.append(f"- [{call.get('outcome', 'unknown')}] {steps_str}")

    nl_text = "\n".join(nl_sections)

    return f"""Analyze this voice agent performance data and generate Section B insights.

## SECTION A: DETERMINISTIC METRICS

```json
{metrics_json}
```

## AGGREGATED QUALITATIVE DATA

{nl_text}

---

Based on the above data, generate the Section B insights JSON following the schema in your instructions.
Focus on:
1. What's the #1 thing causing failures?
2. What are the top 3 actionable fixes?
3. What does the customer voice tell us?

Return ONLY the JSON object."""


def generate_insights(
    metrics_path: Path,
    nl_summary_path: Path | None,
    model_name: str = "gemini-2.5-flash"
) -> dict:
    """Generate Section B insights using LLM.

    Args:
        metrics_path: Path to Section A metrics JSON
        nl_summary_path: Path to NL summary from extract_nl_fields.py
        model_name: Gemini model to use
    """

    # Load Section A metrics
    with open(metrics_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)

    metrics = report_data.get("deterministic_metrics", report_data)

    # Load NL summary (required for v3.1)
    nl_summary = {}
    if nl_summary_path and nl_summary_path.exists():
        with open(nl_summary_path, 'r', encoding='utf-8') as f:
            nl_summary = json.load(f)
        print(f"Loaded NL summary: {nl_summary.get('metadata', {}).get('calls_with_nl_data', 0)} calls with NL data", file=sys.stderr)
    else:
        print("Warning: No NL summary provided. Insights will be based on metrics only.", file=sys.stderr)

    # Build prompt
    prompt = build_insights_prompt(metrics, nl_summary)

    # Call LLM
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=INSIGHTS_SYSTEM_PROMPT
    )

    response = model.generate_content(
        prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.3,
            max_output_tokens=4096,
        )
    )

    insights = extract_json_from_response(response.text)

    return insights


def combine_report(metrics_path: Path, insights: dict, output_path: Path) -> dict:
    """Combine Section A metrics and Section B insights into final report."""

    with open(metrics_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)

    metrics = report_data.get("deterministic_metrics", report_data)

    full_report = {
        "deterministic_metrics": metrics,
        "llm_insights": insights
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2)

    return full_report


def print_insights_summary(insights: dict) -> None:
    """Print human-readable summary of insights."""
    print("\n" + "=" * 60)
    print("VACATIA AI VOICE AGENT - INSIGHTS (v3 - Section B)")
    print("=" * 60)

    # Executive Summary
    print("\n" + "-" * 40)
    print("EXECUTIVE SUMMARY")
    print("-" * 40)
    print(f"  {insights.get('executive_summary', 'N/A')}")

    # Root Cause Analysis
    print("\n" + "-" * 40)
    print("ROOT CAUSE ANALYSIS")
    print("-" * 40)
    rca = insights.get("root_cause_analysis", {})
    print(f"  Primary Driver: {rca.get('primary_driver', 'N/A')}")
    factors = rca.get("contributing_factors", [])
    if factors:
        print("  Contributing Factors:")
        for f in factors:
            print(f"    - {f}")
    print(f"  Evidence: {rca.get('evidence', 'N/A')}")

    # Recommendations
    print("\n" + "-" * 40)
    print("RECOMMENDATIONS")
    print("-" * 40)
    recs = insights.get("actionable_recommendations", [])
    for rec in recs:
        priority = rec.get("priority", "?")
        category = rec.get("category", "?")
        print(f"\n  [{priority}] {rec.get('recommendation', 'N/A')}")
        print(f"       Category: {category}")
        print(f"       Impact: {rec.get('expected_impact', 'N/A')}")

    # Trend Narratives
    print("\n" + "-" * 40)
    print("TREND NARRATIVES")
    print("-" * 40)
    narratives = insights.get("trend_narratives", {})
    for name, narrative in narratives.items():
        label = name.replace("_", " ").title()
        print(f"\n  {label}:")
        print(f"    {narrative}")

    # Verbatim Highlights
    print("\n" + "-" * 40)
    print("CUSTOMER VOICE HIGHLIGHTS")
    print("-" * 40)
    verbatims = insights.get("verbatim_highlights", {})
    if verbatims.get("most_frustrated"):
        print(f"\n  Most Frustrated:")
        print(f'    "{verbatims["most_frustrated"]}"')
    if verbatims.get("most_common_ask"):
        print(f"\n  Most Common Ask:")
        print(f'    "{verbatims["most_common_ask"]}"')
    if verbatims.get("biggest_miss"):
        print(f"\n  Biggest Agent Miss:")
        print(f'    "{verbatims["biggest_miss"]}"')

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate v3.1 Section B: LLM Insights")
    parser.add_argument("-m", "--metrics", type=Path,
                        help="Path to Section A metrics JSON file")
    parser.add_argument("-n", "--nl-summary", type=Path,
                        help="Path to NL summary JSON from extract_nl_fields.py")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "reports",
                        help="Output directory for combined report")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        help="Gemini model to use")
    parser.add_argument("--json-only", action="store_true",
                        help="Output only JSON")

    args = parser.parse_args()

    reports_dir = args.output_dir

    # Find latest metrics file if not specified
    if not args.metrics:
        if reports_dir.exists():
            metrics_files = sorted(reports_dir.glob("metrics_v3_*.json"), reverse=True)
            if metrics_files:
                args.metrics = metrics_files[0]
                print(f"Using latest metrics: {args.metrics}", file=sys.stderr)

    if not args.metrics or not args.metrics.exists():
        print("Error: No metrics file found. Run compute_metrics.py first.", file=sys.stderr)
        return 1

    # Find latest NL summary if not specified
    if not args.nl_summary:
        if reports_dir.exists():
            nl_files = sorted(reports_dir.glob("nl_summary_v3_*.json"), reverse=True)
            if nl_files:
                args.nl_summary = nl_files[0]
                print(f"Using latest NL summary: {args.nl_summary}", file=sys.stderr)

    if not args.nl_summary or not args.nl_summary.exists():
        print("Warning: No NL summary found. Run extract_nl_fields.py first for richer insights.", file=sys.stderr)

    try:
        configure_genai()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Generating insights from: {args.metrics}", file=sys.stderr)

    try:
        insights = generate_insights(
            args.metrics,
            args.nl_summary,
            args.model
        )
    except Exception as e:
        print(f"Error generating insights: {e}", file=sys.stderr)
        return 1

    # Create output path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"report_v3_{timestamp}.json"

    # Combine and save
    full_report = combine_report(args.metrics, insights, output_path)
    print(f"Full report saved: {output_path}", file=sys.stderr)

    if args.json_only:
        print(json.dumps(full_report, indent=2))
    else:
        print_insights_summary(insights)
        print(f"\nFull report: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
