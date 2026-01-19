#!/usr/bin/env python3
"""
Report Review & Refinement Tool for Vacatia AI Voice Agent Analytics (v3.5.5)

Takes rendered markdown report, sends to Gemini 3 Pro for editorial review, outputs:
1. Refined report with review header and tightened prose
2. Pipeline suggestions for future improvements

v3.5.5 additions:
- Editorial review pass for quality assurance
- Inconsistency/gap detection
- Prose tightening (remove redundancies, improve clarity)
- Pipeline improvement suggestions

Usage:
    # Basic usage (uses latest rendered report)
    python3 tools/review_report.py

    # Specify input file
    python3 tools/review_report.py -i reports/executive_summary_v3_20250119.md

    # Custom output directory
    python3 tools/review_report.py -o ./refined_reports

    # Skip pipeline suggestions (review only)
    python3 tools/review_report.py --no-suggestions

    # Use different model
    python3 tools/review_report.py --model gemini-3-pro-preview
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

from google import genai
from google.genai import types


REVIEW_SYSTEM_PROMPT = """You are a senior analytics editor reviewing an executive report on AI voice agent performance.

Your tasks:
1. **Review for quality issues**:
   - Factual inconsistencies (numbers don't match, conflicting statements)
   - Logical gaps (conclusions not supported by evidence)
   - Missing connections (related insights not linked)
   - Unclear language or jargon

2. **Tighten the prose**:
   - Remove redundant statements (same insight said multiple ways)
   - Cut unnecessary qualifiers and hedging
   - Make recommendations more actionable and specific
   - Improve flow between sections

3. **Preserve accuracy**:
   - Do NOT change any numbers, percentages, or counts
   - Do NOT invent new insights not in the original
   - Keep all table data exactly as-is
   - Maintain all call_id references

4. **Create review header**:
   - Summarize key findings from your review (2-3 bullets)
   - Note any issues that couldn't be fixed (data gaps, unclear source data)

5. **Suggest pipeline improvements** (if requested):
   - What additional metrics would be valuable?
   - What insights are missing that could be generated?
   - What report structure changes would improve readability?
   - What data capture changes would improve future analysis?

## Output Format

Return a JSON object with two fields:

{
  "refined_report": "The complete refined markdown report with review header prepended",
  "pipeline_suggestions": {
    "summary": "1-2 sentence overview of improvement opportunities",
    "metrics_to_add": [
      {"metric": "name", "rationale": "why valuable", "data_source": "where it would come from"}
    ],
    "insights_to_add": [
      {"insight": "name", "rationale": "why valuable", "approach": "how to generate"}
    ],
    "report_structure": [
      {"change": "description", "rationale": "why"}
    ],
    "data_capture": [
      {"field": "name", "rationale": "why capture this", "where": "which tool"}
    ]
  }
}

CRITICAL RULES:
1. The refined_report must be COMPLETE valid markdown. Do not truncate.
2. Do NOT include an Appendix section - keep the report tight and executive-focused.
3. Focus refinements on the narrative sections, not the tables.
4. Be CONCISE - remove redundancy but don't add length.
5. Limit pipeline_suggestions to 2-3 items per category max.
"""


def get_genai_client() -> genai.Client:
    """Get configured Google GenAI client."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


# For backwards compatibility
def configure_genai():
    """Configure the Google Generative AI client (deprecated, use get_genai_client)."""
    get_genai_client()  # Just validate the key exists


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response, handling truncation gracefully."""
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

    # Handle truncation: try to salvage the refined_report even if JSON is incomplete
    if '"refined_report"' in text:
        # Find the start of the refined_report value
        report_match = text.find('"refined_report"')
        if report_match != -1:
            # Find the opening quote of the value
            colon_pos = text.find(':', report_match)
            if colon_pos != -1:
                # Skip whitespace after colon
                value_start = colon_pos + 1
                while value_start < len(text) and text[value_start] in ' \t\n':
                    value_start += 1

                if value_start < len(text) and text[value_start] == '"':
                    # Extract the string value, handling escapes
                    content_start = value_start + 1
                    # Find the end by looking for unescaped quote or end of text
                    i = content_start
                    while i < len(text):
                        if text[i] == '\\' and i + 1 < len(text):
                            i += 2  # Skip escaped character
                        elif text[i] == '"':
                            break
                        else:
                            i += 1

                    # Get the content (may be truncated)
                    raw_content = text[content_start:i]
                    # Unescape the string
                    try:
                        refined = json.loads(f'"{raw_content}"')
                    except json.JSONDecodeError:
                        # Try basic unescaping
                        refined = raw_content.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\')

                    if refined and len(refined) > 100:  # Reasonable minimum
                        print(f"Warning: JSON was truncated. Salvaged refined_report ({len(refined)} chars)", file=sys.stderr)
                        return {
                            "refined_report": refined + "\n\n---\n*Note: Report may be truncated due to output length limits.*",
                            "pipeline_suggestions": {}
                        }

    raise ValueError(f"Could not parse JSON from: {text[:500]}...")


def build_review_prompt(report_content: str, include_suggestions: bool = True) -> tuple[str, str]:
    """Build the prompt for LLM report review.

    Returns:
        Tuple of (prompt, appendix_content) - appendix is stripped and discarded
    """

    # Strip the Appendix section entirely - reviewed report is executive-focused
    appendix_marker = "## Appendix: Detailed Metrics"
    appendix_content = ""
    if appendix_marker in report_content:
        appendix_start = report_content.find(appendix_marker)
        # Find the footer marker
        footer_marker = "---\n*Report generated by"
        footer_pos = report_content.find(footer_marker, appendix_start)
        if footer_pos != -1:
            appendix_content = report_content[appendix_start:footer_pos]
            report_content = report_content[:appendix_start] + report_content[footer_pos:]
        else:
            appendix_content = report_content[appendix_start:]
            report_content = report_content[:appendix_start]

    task_section = """
Based on the report above, perform an editorial review and refinement.

NOTE: The Appendix has been removed. The reviewed report should be executive-focused without raw data dumps.

Your output MUST include:
1. `refined_report`: The markdown report with:
   - A review header prepended (see format below)
   - Tightened prose throughout
   - Improved flow and clarity
   - All original numbers/data preserved exactly
   - End with the footer line (no Appendix)
"""

    if include_suggestions:
        task_section += """
2. `pipeline_suggestions`: Actionable ideas for improving the analytics pipeline (2-3 items per category max)

"""
    else:
        task_section += """
2. `pipeline_suggestions`: Empty object {} (suggestions disabled)

"""

    task_section += """
## Review Header Format

The review header should be prepended to the refined report in this format:

---
## Report Review Summary

**Reviewed:** {timestamp} | **Model:** Gemini 3 Pro

### Refinements Made
- [Bullet 1: What you consolidated/clarified]
- [Bullet 2: What connections you made]
- [Bullet 3: What prose you tightened]

### Data Quality Notes
- [Any issues that couldn't be fixed]
- [Any unclear source data]

---

[Rest of refined report, ending with footer...]

Return ONLY the JSON object.
"""

    prompt = f"""Review and refine this executive report on AI voice agent performance.

## ORIGINAL REPORT

{report_content}

---

{task_section}"""

    return prompt, appendix_content


def review_report(
    report_path: Path,
    model_name: str = "gemini-3-pro-preview",
    include_suggestions: bool = True
) -> dict:
    """Review and refine a rendered report using LLM.

    Args:
        report_path: Path to rendered markdown report
        model_name: Gemini model to use
        include_suggestions: Whether to generate pipeline suggestions

    Returns:
        Dict with 'refined_report' and 'pipeline_suggestions'
    """

    # Load report content
    with open(report_path, 'r', encoding='utf-8') as f:
        report_content = f.read()

    # Build prompt (strips appendix - reviewed report won't include it)
    prompt, _ = build_review_prompt(report_content, include_suggestions)

    # Call LLM with new SDK
    client = get_genai_client()

    config = types.GenerateContentConfig(
        temperature=0.2,  # Lower temperature for editorial consistency
        max_output_tokens=100000,  # Very large limit to accommodate full report + suggestions
        # No thinking_config = uses model default (HIGH for Pro)
        system_instruction=REVIEW_SYSTEM_PROMPT,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )

    try:
        result = extract_json_from_response(response.text)
    except ValueError as e:
        # If still failing, try a second pass with just the review (no suggestions)
        if include_suggestions:
            print("Retrying with suggestions disabled due to truncation...", file=sys.stderr)
            prompt_no_suggestions, _ = build_review_prompt(report_content, include_suggestions=False)
            config_retry = types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=100000,
                system_instruction=REVIEW_SYSTEM_PROMPT,
            )
            response2 = client.models.generate_content(
                model=model_name,
                contents=prompt_no_suggestions,
                config=config_retry,
            )
            result = extract_json_from_response(response2.text)
        else:
            raise e

    # Note: Appendix is intentionally NOT restored - reviewed report is executive-focused
    # Original report with full appendix is preserved alongside

    return result


def render_suggestions_markdown(suggestions: dict, source_report: str, timestamp: str) -> str:
    """Render pipeline suggestions as markdown."""

    lines = [
        "# Pipeline Improvement Suggestions",
        "",
        f"**Generated:** {timestamp}",
        f"**Based on:** {source_report}",
        f"**Model:** Gemini 3 Pro",
        "",
    ]

    # Summary
    if suggestions.get("summary"):
        lines.extend([
            "## Summary",
            "",
            suggestions["summary"],
            "",
        ])

    # Metrics to Add
    metrics = suggestions.get("metrics_to_add", [])
    if metrics:
        lines.extend([
            "## Recommended Metrics to Add",
            "",
        ])
        for i, m in enumerate(metrics, 1):
            lines.extend([
                f"### {i}. {m.get('metric', 'Unnamed')}",
                f"**Rationale:** {m.get('rationale', 'N/A')}",
                f"**Data source:** {m.get('data_source', 'N/A')}",
                "",
            ])

    # Insights to Add
    insights = suggestions.get("insights_to_add", [])
    if insights:
        lines.extend([
            "## Recommended Insights to Add",
            "",
        ])
        for i, ins in enumerate(insights, 1):
            lines.extend([
                f"### {i}. {ins.get('insight', 'Unnamed')}",
                f"**Rationale:** {ins.get('rationale', 'N/A')}",
                f"**Approach:** {ins.get('approach', 'N/A')}",
                "",
            ])

    # Report Structure
    structure = suggestions.get("report_structure", [])
    if structure:
        lines.extend([
            "## Report Structure Suggestions",
            "",
        ])
        for s in structure:
            lines.append(f"- **{s.get('change', 'N/A')}** - {s.get('rationale', '')}")
        lines.append("")

    # Data Capture
    data_capture = suggestions.get("data_capture", [])
    if data_capture:
        lines.extend([
            "## Data Capture Improvements",
            "",
        ])
        for d in data_capture:
            lines.append(f"- **{d.get('field', 'N/A')}** ({d.get('where', 'unknown')}) - {d.get('rationale', '')}")
        lines.append("")

    lines.extend([
        "---",
        "*Generated by Vacatia AI Voice Agent Analytics Framework v3.5.5*",
    ])

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Review and refine rendered report (v3.5.5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage (uses latest rendered report)
    python3 tools/review_report.py

    # Specify input file
    python3 tools/review_report.py -i reports/executive_summary_v3_20250119.md

    # Skip pipeline suggestions
    python3 tools/review_report.py --no-suggestions
        """
    )

    parser.add_argument("-i", "--input", type=Path,
                        help="Path to rendered markdown report")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "reports",
                        help="Output directory for refined report and suggestions")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview",
                        help="Gemini model for report review (default: gemini-3-pro-preview)")
    parser.add_argument("--no-suggestions", action="store_true",
                        help="Skip pipeline suggestions (review only)")
    parser.add_argument("--stdout", action="store_true",
                        help="Print refined report to stdout instead of saving")

    args = parser.parse_args()

    reports_dir = args.output_dir

    # Find latest rendered report if not specified
    if not args.input:
        if reports_dir.exists():
            # Look for non-reviewed executive summaries
            report_files = sorted(
                [f for f in reports_dir.glob("executive_summary_v3_*.md")
                 if "_reviewed" not in f.name],
                reverse=True
            )
            if report_files:
                args.input = report_files[0]
                print(f"Using latest report: {args.input}", file=sys.stderr)

    if not args.input or not args.input.exists():
        print("Error: No report file found. Run render_report.py first.", file=sys.stderr)
        return 1

    try:
        configure_genai()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Reviewing report: {args.input}", file=sys.stderr)
    print(f"Model: {args.model}", file=sys.stderr)

    try:
        result = review_report(
            args.input,
            args.model,
            include_suggestions=not args.no_suggestions
        )
    except Exception as e:
        print(f"Error reviewing report: {e}", file=sys.stderr)
        return 1

    refined_report = result.get("refined_report", "")
    suggestions = result.get("pipeline_suggestions", {})

    if not refined_report:
        print("Error: No refined report returned from LLM", file=sys.stderr)
        return 1

    if args.stdout:
        print(refined_report)
    else:
        # Create output paths
        args.output_dir.mkdir(parents=True, exist_ok=True)

        # Extract timestamp from input filename or generate new one
        input_stem = args.input.stem
        if "_" in input_stem:
            # Try to extract timestamp from filename like executive_summary_v3_20250119_143022
            parts = input_stem.split("_")
            timestamp_parts = [p for p in parts if p.isdigit() and len(p) >= 6]
            timestamp_str = "_".join(timestamp_parts) if timestamp_parts else datetime.now().strftime("%Y%m%d_%H%M%S")
        else:
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Save refined report
        reviewed_path = args.output_dir / f"executive_summary_v3_{timestamp_str}_reviewed.md"
        with open(reviewed_path, 'w', encoding='utf-8') as f:
            f.write(refined_report)
        print(f"Reviewed report saved: {reviewed_path}", file=sys.stderr)

        # Save pipeline suggestions if generated
        if suggestions and not args.no_suggestions:
            suggestions_path = args.output_dir / f"pipeline_suggestions_v3_{timestamp_str}.md"
            suggestions_md = render_suggestions_markdown(
                suggestions,
                args.input.name,
                datetime.now().strftime("%Y-%m-%d %H:%M")
            )
            with open(suggestions_path, 'w', encoding='utf-8') as f:
                f.write(suggestions_md)
            print(f"Pipeline suggestions saved: {suggestions_path}", file=sys.stderr)

        print(f"\nReviewed report: {reviewed_path}")
        if suggestions and not args.no_suggestions:
            print(f"Pipeline suggestions: {suggestions_path}")

    return 0


if __name__ == "__main__":
    exit(main())
