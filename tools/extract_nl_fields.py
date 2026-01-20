#!/usr/bin/env python3
"""
NL Field Extractor for Vacatia AI Voice Agent Analytics (v3.9.1)

Extracts natural language fields from v3 analysis JSONs into a condensed
format optimized for LLM insight generation.

v3.9.1 additions:
- loop_subjects: Extracts subject field from loops for granular analysis
- loop_subject_pairs: (loop_type, subject) pairs for LLM semantic clustering

v3.9 additions:
- call_disposition extraction for funnel analysis insights
- disposition_summary: Groups calls by disposition for LLM narrative generation

v3.8.5 additions:
- Backwards-compatible extraction from both v3.8.5 (friction) and v3.8 formats
- Extracts from friction.clarifications, friction.corrections, friction.loops
- Maps short enum values to canonical values for consistent aggregation
- Loops now include turn numbers (t array) when available

v3.8 additions:
- loop_events: Now extracts from agent_loops with type + context
- Each loop event includes: type (enum), context (description), call outcome
- Supports friction loop analysis by type (info_retry, intent_retry, etc.)

v3.7 additions:
- clarification_events now include: cause (enum) + context (string)
- correction_events now include: severity (enum) + context (string)
- These structured fields enable reliable aggregation + nuanced analysis

v3.6 additions:
- clarification_events: Calls with clarification requests + outcomes
- correction_events: Calls with user corrections + frustration signals
- conversation_turn_outliers: Unusually long or short calls for analysis

v3.5 additions:
- training_details: Training opportunities with associated failure context
- all_additional_intents: Secondary customer intents for clustering

v3.3 additions:
- Scope coherence: Respects manifest.csv for run isolation

This dedicated script replaces the --export-nl-fields flag in compute_metrics.py,
providing:
1. Explicit architecture - extraction is a first-class pipeline step
2. Optimized output - ~70% smaller than full analyses
3. LLM-ready format - grouped by failure type with all context needed

Output: nl_summary_v3_{timestamp}.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
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
            filename = row.get("filename", "")
            if filename:
                call_ids.add(Path(filename).stem)
    return call_ids


def load_v3_analyses(analyses_dir: Path, manifest_ids: set[str] | None = None) -> list[dict]:
    """Load only v3 schema analysis files.

    Args:
        analyses_dir: Directory containing analysis JSON files
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
                    # Accept v3.x versions (v3, v3.5, v3.6, etc.)
                    version = data.get("schema_version", "")
                    if version.startswith("v3"):
                        analyses.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {f}: {e}", file=sys.stderr)

    if skipped_not_in_manifest > 0:
        print(f"Scope filter: Skipped {skipped_not_in_manifest} analyses not in manifest", file=sys.stderr)

    return analyses


# v3.8.5 Enum mapping: short values → canonical values
CLARIFICATION_TYPE_MAP = {
    "name": "name_spelling", "phone": "phone_confirmation",
    "intent": "intent_clarification", "repeat": "repeat_request", "verify": "verification_retry",
    "name_spelling": "name_spelling", "phone_confirmation": "phone_confirmation",
    "intent_clarification": "intent_clarification", "repeat_request": "repeat_request",
    "verification_retry": "verification_retry",
}

CLARIFICATION_CAUSE_MAP = {
    "misheard": "agent_misheard", "unclear": "customer_unclear",
    "refused": "customer_refused", "tech": "tech_issue", "ok": "successful",
    "agent_misheard": "agent_misheard", "customer_unclear": "customer_unclear",
    "customer_refused": "customer_refused", "tech_issue": "tech_issue", "successful": "successful",
}


def get_conversation_turns(analysis: dict) -> int | None:
    """Get conversation turns from v3.8.5 (friction.turns) or v3.8 (conversation_turns)."""
    friction = analysis.get("friction")
    if friction and isinstance(friction, dict):
        turns = friction.get("turns")
        if turns is not None:
            return turns
    return analysis.get("conversation_turns")


def extract_clarification_events(analysis: dict) -> list[dict]:
    """Extract clarification events from both v3.8.5 and v3.8 formats."""
    events = []
    call_id = analysis.get("call_id")
    outcome = analysis.get("outcome")

    # v3.8.5 format: friction.clarifications
    friction = analysis.get("friction")
    if friction and isinstance(friction, dict):
        for c in friction.get("clarifications", []):
            events.append({
                "call_id": call_id,
                "type": CLARIFICATION_TYPE_MAP.get(c.get("type"), c.get("type")),
                "turn": c.get("t"),
                "resolved": None,  # v3.8.5 doesn't track resolved
                "cause": CLARIFICATION_CAUSE_MAP.get(c.get("cause"), c.get("cause")),
                "context": c.get("ctx"),
                "outcome": outcome
            })
        return events

    # v3.8 format: clarification_requests.details
    clar = analysis.get("clarification_requests")
    if clar and isinstance(clar, dict) and clar.get("count", 0) > 0:
        for d in clar.get("details", []):
            events.append({
                "call_id": call_id,
                "type": d.get("type"),
                "turn": d.get("turn"),
                "resolved": d.get("resolved"),
                "cause": d.get("cause"),
                "context": d.get("context"),
                "outcome": outcome
            })

    return events


def extract_correction_events(analysis: dict) -> list[dict]:
    """Extract correction events from both v3.8.5 and v3.8 formats."""
    events = []
    call_id = analysis.get("call_id")
    outcome = analysis.get("outcome")

    # v3.8.5 format: friction.corrections
    friction = analysis.get("friction")
    if friction and isinstance(friction, dict):
        for c in friction.get("corrections", []):
            sev = c.get("sev")
            events.append({
                "call_id": call_id,
                "what": None,  # v3.8.5 doesn't track what_was_wrong
                "turn": c.get("t"),
                "frustrated": sev == "major",  # Infer from severity
                "severity": sev,
                "context": c.get("ctx"),
                "outcome": outcome
            })
        return events

    # v3.8 format: user_corrections.details
    corr = analysis.get("user_corrections")
    if corr and isinstance(corr, dict) and corr.get("count", 0) > 0:
        for d in corr.get("details", []):
            events.append({
                "call_id": call_id,
                "what": d.get("what_was_wrong"),
                "turn": d.get("turn"),
                "frustrated": d.get("frustration_signal"),
                "severity": d.get("severity"),
                "context": d.get("context"),
                "outcome": outcome
            })

    return events


def extract_loop_events(analysis: dict) -> list[dict]:
    """Extract loop events from both v3.9.1/v3.8.5 and v3.8 formats.

    v3.9.1: Includes subject field for granular loop analysis.
    """
    events = []
    call_id = analysis.get("call_id")
    outcome = analysis.get("outcome")

    # v3.9.1/v3.8.5 format: friction.loops
    friction = analysis.get("friction")
    if friction and isinstance(friction, dict):
        for l in friction.get("loops", []):
            events.append({
                "call_id": call_id,
                "turns": l.get("t", []),  # v3.8.5+: array of turn numbers
                "type": l.get("type"),
                "subject": l.get("subject"),  # v3.9.1: what is being looped on
                "context": l.get("ctx"),
                "outcome": outcome
            })
        return events

    # v3.8 format: agent_loops.details
    loops = analysis.get("agent_loops")
    if loops and isinstance(loops, dict) and loops.get("count", 0) > 0:
        for d in loops.get("details", []):
            events.append({
                "call_id": call_id,
                "turns": None,  # v3.8 has no turn numbers in loops
                "type": d.get("type"),
                "subject": None,  # v3.8 has no subject
                "context": d.get("context"),
                "outcome": outcome
            })
        return events

    # v3.7 format: repeated_prompts (legacy)
    rp = analysis.get("repeated_prompts")
    if rp and isinstance(rp, dict) and rp.get("count", 0) > 0:
        events.append({
            "call_id": call_id,
            "count": rp.get("count"),
            "max_consecutive": rp.get("max_consecutive"),
            "type": None,  # Legacy format
            "subject": None,
            "context": None,
            "outcome": outcome
        })

    return events


def extract_condensed_call(analysis: dict) -> dict:
    """Extract only the NL-relevant fields from a single analysis (~5-10 lines vs 18 fields)."""
    condensed = {
        "call_id": analysis.get("call_id"),
        "outcome": analysis.get("outcome"),
        "failure_point": analysis.get("failure_point"),
    }

    # v3.9: Include call disposition
    if analysis.get("call_disposition"):
        condensed["call_disposition"] = analysis["call_disposition"]

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

    # v3.4: Extract all customer_asks for LLM semantic clustering
    # This provides the raw list so LLM can cluster semantically similar asks
    all_customer_asks = []
    for detail in policy_gap_details:
        ask = detail.get("ask")
        if ask and ask.strip():
            all_customer_asks.append(ask.strip())

    # v3.5: Training details with context
    # Extracts training opportunities with associated failure info for narrative generation
    training_details = []
    for a in analyses:
        training_opp = a.get("training_opportunity")
        if training_opp:
            training_details.append({
                "call_id": a.get("call_id"),
                "opportunity": training_opp,
                "outcome": a.get("outcome"),
                "failure_point": a.get("failure_point"),
                "failure_description": a.get("failure_description"),
                "agent_miss_detail": a.get("agent_miss_detail")
            })

    # v3.5: Additional intents (secondary customer needs)
    all_additional_intents = []
    for a in analyses:
        if a.get("additional_intents"):
            all_additional_intents.append({
                "call_id": a.get("call_id"),
                "outcome": a.get("outcome"),
                "intent": a.get("additional_intents")
            })

    # v3.8.5: Clarification events (using backwards-compatible extraction)
    clarification_events = []
    for a in analyses:
        clarification_events.extend(extract_clarification_events(a))

    # v3.8.5: Correction events (using backwards-compatible extraction)
    correction_events = []
    for a in analyses:
        correction_events.extend(extract_correction_events(a))

    # v3.8.5/v3.9.1: Loop events (using backwards-compatible extraction)
    loop_events = []
    for a in analyses:
        loop_events.extend(extract_loop_events(a))

    # v3.9.1: Extract (type, subject) pairs for LLM semantic clustering
    loop_subject_pairs = []
    for event in loop_events:
        loop_type = event.get("type")
        subject = event.get("subject")
        if loop_type and subject:
            loop_subject_pairs.append({
                "call_id": event.get("call_id"),
                "loop_type": loop_type,
                "subject": subject,
                "outcome": event.get("outcome")
            })

    # v3.8.5: Conversation turn outliers (using backwards-compatible extraction)
    # Identify unusually long or short calls for analysis
    turn_data = [(a.get("call_id"), get_conversation_turns(a), a.get("outcome"))
                 for a in analyses if get_conversation_turns(a)]
    turn_values = [t[1] for t in turn_data if t[1] is not None]
    turn_outliers = []
    if len(turn_values) >= 5:
        import statistics
        avg = statistics.mean(turn_values)
        std = statistics.stdev(turn_values) if len(turn_values) > 1 else 0
        threshold_high = avg + 2 * std
        threshold_low = max(1, avg - 1.5 * std)  # At least 1 turn

        for call_id, turns, outcome in turn_data:
            if turns is not None:
                if turns >= threshold_high:
                    turn_outliers.append({
                        "call_id": call_id,
                        "turns": turns,
                        "type": "long",
                        "outcome": outcome
                    })
                elif turns <= threshold_low:
                    turn_outliers.append({
                        "call_id": call_id,
                        "turns": turns,
                        "type": "short",
                        "outcome": outcome
                    })

    # v3.9: Extract disposition summary for LLM narrative generation
    disposition_summary = defaultdict(list)
    for a in analyses:
        disposition = a.get("call_disposition")
        if disposition:
            entry = {
                "call_id": a.get("call_id"),
                "outcome": a.get("outcome"),
            }
            if a.get("failure_description"):
                entry["description"] = a["failure_description"]
            if a.get("customer_verbatim"):
                entry["verbatim"] = a["customer_verbatim"]
            if a.get("summary"):
                entry["summary"] = a["summary"]
            disposition_summary[disposition].append(entry)

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
        "failed_call_flows": failed_call_flows,
        "all_customer_asks": all_customer_asks,  # v3.4: Raw list for LLM clustering
        "training_details": training_details,  # v3.5: Training opportunities with context
        "all_additional_intents": all_additional_intents,  # v3.5: Secondary customer intents
        "clarification_events": clarification_events,  # v3.6: Clarification request details
        "correction_events": correction_events,  # v3.6: User correction details
        "loop_events": loop_events,  # v3.6: Repeated prompt events
        "loop_subject_pairs": loop_subject_pairs,  # v3.9.1: (type, subject) pairs for clustering
        "turn_outliers": turn_outliers,  # v3.6: Unusually long/short calls
        "disposition_summary": dict(disposition_summary)  # v3.9: Calls by disposition
    }


def print_summary(nl_summary: dict) -> None:
    """Print human-readable summary of extracted NL data."""
    print("\n" + "=" * 60)
    print("NL FIELD EXTRACTION SUMMARY (v3.9.1)")
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

    # v3.5: Training details
    print("\n" + "-" * 40)
    print("TRAINING OPPORTUNITIES (v3.5)")
    print("-" * 40)
    training = nl_summary.get("training_details", [])
    print(f"  Total training opportunities: {len(training)}")
    if training:
        # Count by type
        from collections import Counter
        types = Counter(t.get("opportunity") for t in training if t.get("opportunity"))
        for opp_type, count in types.most_common(5):
            print(f"    {opp_type}: {count}")

    # v3.5: Additional intents
    print("\n" + "-" * 40)
    print("ADDITIONAL INTENTS (v3.5)")
    print("-" * 40)
    intents = nl_summary.get("all_additional_intents", [])
    print(f"  Calls with secondary intents: {len(intents)}")
    if intents:
        for i in intents[:3]:
            intent = i.get("intent", "")
            if len(intent) > 60:
                intent = intent[:60] + "..."
            print(f'    - "{intent}"')
        if len(intents) > 3:
            print(f"    ... and {len(intents) - 3} more")

    # v3.8.5: Conversation Quality Events
    print("\n" + "-" * 40)
    print("CONVERSATION QUALITY (v3.8.5)")
    print("-" * 40)

    # Clarification events
    clar_events = nl_summary.get("clarification_events", [])
    print(f"  Clarification events: {len(clar_events)}")
    if clar_events:
        from collections import Counter
        types = Counter(e.get("type") for e in clar_events if e.get("type"))
        for ctype, count in types.most_common(3):
            print(f"    {ctype}: {count}")
        # v3.7: Show cause distribution
        causes = Counter(e.get("cause") for e in clar_events if e.get("cause"))
        if causes:
            print("  By Cause (v3.7):")
            for cause, count in causes.most_common():
                print(f"    {cause}: {count}")

    # Correction events
    corr_events = nl_summary.get("correction_events", [])
    print(f"  Correction events: {len(corr_events)}")
    frustrated = sum(1 for e in corr_events if e.get("frustrated"))
    if corr_events:
        print(f"    With frustration signal: {frustrated}")
        # v3.7: Show severity distribution
        from collections import Counter
        severities = Counter(e.get("severity") for e in corr_events if e.get("severity"))
        if severities:
            print("  By Severity (v3.7):")
            for severity, count in severities.most_common():
                print(f"    {severity}: {count}")

    # Loop events (v3.9.1: with type + subject + context + turns)
    loop_events = nl_summary.get("loop_events", [])
    print(f"  Loop events: {len(loop_events)}")
    if loop_events:
        # v3.8.5+: Show type distribution
        from collections import Counter
        types = Counter(e.get("type") for e in loop_events if e.get("type"))
        if types:
            print("  By Type:")
            for loop_type, count in types.most_common():
                print(f"    {loop_type}: {count}")
        else:
            # Legacy format
            max_cons = max((e.get("max_consecutive") or 0) for e in loop_events)
            print(f"    Worst consecutive repeats: {max_cons}")

    # v3.9.1: Loop subject pairs
    loop_subject_pairs = nl_summary.get("loop_subject_pairs", [])
    if loop_subject_pairs:
        print(f"  Loop subject pairs (v3.9.1): {len(loop_subject_pairs)}")
        from collections import Counter
        type_subject_pairs = Counter(
            (p.get("loop_type"), p.get("subject"))
            for p in loop_subject_pairs
        )
        for (loop_type, subject), count in type_subject_pairs.most_common(5):
            print(f"    {loop_type}/{subject}: {count}")

    # Turn outliers
    turn_outliers = nl_summary.get("turn_outliers", [])
    if turn_outliers:
        long_calls = [e for e in turn_outliers if e.get("type") == "long"]
        short_calls = [e for e in turn_outliers if e.get("type") == "short"]
        print(f"  Turn outliers: {len(long_calls)} long, {len(short_calls)} short")

    # v3.9: Disposition Summary
    print("\n" + "-" * 40)
    print("DISPOSITION SUMMARY (v3.9)")
    print("-" * 40)
    disp_summary = nl_summary.get("disposition_summary", {})
    if disp_summary:
        total_with_disp = sum(len(entries) for entries in disp_summary.values())
        print(f"  Total calls with disposition: {total_with_disp}")
        for disp, entries in sorted(disp_summary.items(), key=lambda x: -len(x[1])):
            print(f"    {disp}: {len(entries)} calls")
    else:
        print("  No disposition data (pre-v3.9 analyses)")

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
    parser.add_argument("-s", "--sampled-dir", type=Path,
                        default=Path(__file__).parent.parent / "sampled",
                        help="Directory containing manifest.csv for scope filtering (v3.3)")
    parser.add_argument("--no-scope-filter", action="store_true",
                        help="Disable manifest-based scope filtering (include all analyses)")
    parser.add_argument("--limit", type=int,
                        help="Limit number of analyses to process (for testing)")
    parser.add_argument("--json-only", action="store_true",
                        help="Output only JSON, no summary")

    args = parser.parse_args()

    # v3.3: Load manifest for scope coherence
    manifest_ids = None
    if not args.no_scope_filter:
        manifest_ids = load_manifest_ids(args.sampled_dir)
        if manifest_ids:
            print(f"Scope filter: Using manifest with {len(manifest_ids)} call IDs", file=sys.stderr)
        else:
            print("Scope filter: No manifest found, including all analyses", file=sys.stderr)

    print(f"Loading v3 analyses from: {args.input_dir}", file=sys.stderr)
    analyses = load_v3_analyses(args.input_dir, manifest_ids)

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
