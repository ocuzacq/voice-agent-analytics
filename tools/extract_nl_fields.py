#!/usr/bin/env python3
"""
NL Field Extractor for Vacatia AI Voice Agent Analytics (v3.1)

Extracts natural language fields from v3 analysis JSONs into a condensed
format optimized for LLM insight generation.

This dedicated script replaces the --export-nl-fields flag in compute_metrics.py,
providing:
1. Explicit architecture - extraction is a first-class pipeline step
2. Optimized output - ~70% smaller than full analyses
3. LLM-ready format - grouped by failure type with all context needed

Output: nl_summary_v3_{timestamp}.json
"""

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def load_v3_analyses(analyses_dir: Path) -> list[dict]:
    """Load only v3 schema analysis files."""
    analyses = []
    for f in analyses_dir.iterdir():
        if f.is_file() and f.suffix == '.json':
            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    if data.get("schema_version") == "v3":
                        analyses.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {f}: {e}", file=sys.stderr)
    return analyses


def extract_condensed_call(analysis: dict) -> dict:
    """Extract only the NL-relevant fields from a single analysis (~5-10 lines vs 18 fields)."""
    condensed = {
        "call_id": analysis.get("call_id"),
        "outcome": analysis.get("outcome"),
        "failure_point": analysis.get("failure_point"),
    }

    # Only include non-null NL fields
    if analysis.get("failure_description"):
        condensed["failure_description"] = analysis["failure_description"]

    if analysis.get("customer_verbatim"):
        condensed["customer_verbatim"] = analysis["customer_verbatim"]

    if analysis.get("agent_miss_detail"):
        condensed["agent_miss_detail"] = analysis["agent_miss_detail"]

    if analysis.get("policy_gap_detail"):
        condensed["policy_gap_detail"] = analysis["policy_gap_detail"]

    if analysis.get("resolution_steps"):
        condensed["resolution_steps"] = analysis["resolution_steps"]

    if analysis.get("was_recoverable") is not None:
        condensed["was_recoverable"] = analysis["was_recoverable"]

    return condensed


def extract_nl_summary(analyses: list[dict]) -> dict:
    """
    Extract and organize NL fields into LLM-optimized structure.

    Output structure:
    - by_failure_type: Grouped failure data for pattern analysis
    - all_verbatims: Customer quotes for voice-of-customer insights
    - all_agent_misses: Coaching opportunities
    - policy_gap_details: Structured gap analysis
    - failed_call_flows: Resolution step sequences for failed calls
    """

    # Group by failure type
    by_failure_type = defaultdict(list)
    all_verbatims = []
    all_agent_misses = []
    policy_gap_details = []
    failed_call_flows = []

    calls_with_nl_data = 0

    for a in analyses:
        call_id = a.get("call_id")
        outcome = a.get("outcome")
        failure_point = a.get("failure_point", "none")

        has_nl_data = False

        # Group by failure type (excluding "none")
        if failure_point and failure_point != "none":
            entry = {
                "call_id": call_id,
                "outcome": outcome,
            }
            if a.get("failure_description"):
                entry["description"] = a["failure_description"]
                has_nl_data = True
            if a.get("customer_verbatim"):
                entry["verbatim"] = a["customer_verbatim"]
                has_nl_data = True
            if a.get("agent_miss_detail"):
                entry["miss"] = a["agent_miss_detail"]
                has_nl_data = True

            by_failure_type[failure_point].append(entry)

        # Collect all verbatims
        if a.get("customer_verbatim"):
            all_verbatims.append({
                "call_id": call_id,
                "outcome": outcome,
                "quote": a["customer_verbatim"]
            })
            has_nl_data = True

        # Collect all agent misses
        if a.get("agent_miss_detail"):
            all_agent_misses.append({
                "call_id": call_id,
                "was_recoverable": a.get("was_recoverable"),
                "miss": a["agent_miss_detail"]
            })
            has_nl_data = True

        # Collect policy gap details
        if a.get("policy_gap_detail"):
            detail = a["policy_gap_detail"]
            policy_gap_details.append({
                "call_id": call_id,
                "category": detail.get("category"),
                "gap": detail.get("specific_gap"),
                "ask": detail.get("customer_ask"),
                "blocker": detail.get("blocker")
            })
            has_nl_data = True

        # Collect failed call flows
        if outcome != "resolved" and a.get("resolution_steps"):
            failed_call_flows.append({
                "call_id": call_id,
                "outcome": outcome,
                "failure_point": failure_point,
                "steps": a["resolution_steps"]
            })
            has_nl_data = True

        if has_nl_data:
            calls_with_nl_data += 1

    return {
        "metadata": {
            "extracted_at": datetime.now().isoformat(),
            "total_calls": len(analyses),
            "calls_with_nl_data": calls_with_nl_data
        },
        "by_failure_type": dict(by_failure_type),
        "all_verbatims": all_verbatims,
        "all_agent_misses": all_agent_misses,
        "policy_gap_details": policy_gap_details,
        "failed_call_flows": failed_call_flows
    }


def print_summary(nl_summary: dict) -> None:
    """Print human-readable summary of extracted NL data."""
    print("\n" + "=" * 60)
    print("NL FIELD EXTRACTION SUMMARY (v3.1)")
    print("=" * 60)

    meta = nl_summary.get("metadata", {})
    print(f"\nTotal calls analyzed: {meta.get('total_calls', 0)}")
    print(f"Calls with NL data: {meta.get('calls_with_nl_data', 0)}")
    print(f"Extracted at: {meta.get('extracted_at', 'N/A')}")

    # By failure type
    print("\n" + "-" * 40)
    print("BY FAILURE TYPE")
    print("-" * 40)
    by_type = nl_summary.get("by_failure_type", {})
    if by_type:
        for fp_type, entries in by_type.items():
            print(f"  {fp_type}: {len(entries)} calls")
    else:
        print("  No failure data")

    # Verbatims
    print("\n" + "-" * 40)
    print("CUSTOMER VERBATIMS")
    print("-" * 40)
    verbatims = nl_summary.get("all_verbatims", [])
    print(f"  Total quotes extracted: {len(verbatims)}")
    for v in verbatims[:3]:  # Show first 3
        print(f'  - "{v["quote"][:60]}..."' if len(v.get("quote", "")) > 60 else f'  - "{v.get("quote", "")}"')
    if len(verbatims) > 3:
        print(f"  ... and {len(verbatims) - 3} more")

    # Agent misses
    print("\n" + "-" * 40)
    print("AGENT MISSES")
    print("-" * 40)
    misses = nl_summary.get("all_agent_misses", [])
    print(f"  Total coaching opportunities: {len(misses)}")
    recoverable = sum(1 for m in misses if m.get("was_recoverable"))
    print(f"  Recoverable calls: {recoverable}")

    # Policy gaps
    print("\n" + "-" * 40)
    print("POLICY GAP DETAILS")
    print("-" * 40)
    gaps = nl_summary.get("policy_gap_details", [])
    print(f"  Total gaps documented: {len(gaps)}")
    if gaps:
        # Count by category
        from collections import Counter
        categories = Counter(g.get("category") for g in gaps if g.get("category"))
        for cat, count in categories.most_common():
            print(f"    {cat}: {count}")

    # Failed flows
    print("\n" + "-" * 40)
    print("FAILED CALL FLOWS")
    print("-" * 40)
    flows = nl_summary.get("failed_call_flows", [])
    print(f"  Flows documented: {len(flows)}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Extract NL fields from v3 analyses for LLM insight generation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python3 tools/extract_nl_fields.py

    # Custom input/output
    python3 tools/extract_nl_fields.py -i ./analyses -o ./reports

    # Limit sample size (for testing)
    python3 tools/extract_nl_fields.py --limit 50
        """
    )

    parser.add_argument("-i", "--input-dir", type=Path,
                        default=Path(__file__).parent.parent / "analyses",
                        help="Directory containing analysis JSON files")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "reports",
                        help="Output directory for NL summary")
    parser.add_argument("--limit", type=int,
                        help="Limit number of analyses to process (for testing)")
    parser.add_argument("--json-only", action="store_true",
                        help="Output only JSON, no summary")

    args = parser.parse_args()

    print(f"Loading v3 analyses from: {args.input_dir}", file=sys.stderr)
    analyses = load_v3_analyses(args.input_dir)

    if not analyses:
        print("Error: No v3 schema analysis files found", file=sys.stderr)
        return 1

    print(f"Found {len(analyses)} v3 analyses", file=sys.stderr)

    # Apply limit if specified
    if args.limit and args.limit < len(analyses):
        analyses = analyses[:args.limit]
        print(f"Limited to {args.limit} analyses", file=sys.stderr)

    # Extract NL summary
    nl_summary = extract_nl_summary(analyses)

    # Save output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"nl_summary_v3_{timestamp}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(nl_summary, f, indent=2)

    print(f"NL summary saved: {output_path}", file=sys.stderr)

    # Calculate size reduction
    original_size = sum(
        len(json.dumps(a)) for a in analyses
    )
    condensed_size = len(json.dumps(nl_summary))
    reduction = (1 - condensed_size / original_size) * 100 if original_size > 0 else 0
    print(f"Size reduction: {reduction:.1f}% ({original_size} → {condensed_size} bytes)", file=sys.stderr)

    if args.json_only:
        print(json.dumps(nl_summary, indent=2))
    else:
        print_summary(nl_summary)
        print(f"\nOutput: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
