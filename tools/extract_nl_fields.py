#!/usr/bin/env python3
"""
NL Field Extractor for Vacatia AI Voice Agent Analytics (v4.1)

Extracts natural language fields from v4/v3 analysis JSONs into a condensed
format optimized for LLM insight generation.

v4.1 additions:
- Run-based isolation support via --run-dir argument
- Reads from run's analyses/, writes to run's reports/
- Backwards compatible with legacy flat directory mode

v4.0 additions:
- intent_data: Extracts intent, intent_context, secondary_intent for caller needs analysis
- sentiment_data: Extracts sentiment_start, sentiment_end for emotional journey tracking
- Updated field mappings for v4.0 renamed fields (verbatim, coaching, etc.)
- Flattened friction extraction (clarifications, corrections, loops at top-level)
- Disposition extraction unified (v4.0 disposition vs v3.x call_disposition)

v3.9.1 additions:
- loop_subjects: Extracts subject field from loops for granular analysis
- loop_subject_pairs: (loop_type, subject) pairs for LLM semantic clustering

v3.9 additions:
- call_disposition extraction for funnel analysis insights
- disposition_summary: Groups calls by disposition for LLM narrative generation

v3.8.5 additions:
- Backwards-compatible extraction from both v3.8.5 (friction) and v3.8 formats
- Extracts from friction.clarifications, friction.corrections, friction.loops

This dedicated script provides:
1. Explicit architecture - extraction is a first-class pipeline step
2. Optimized output - ~70% smaller than full analyses
3. LLM-ready format - grouped by failure type with all context needed

Output: nl_summary_v4_{timestamp}.json
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from run_utils import (
    add_run_arguments, resolve_run_from_args, get_run_paths,
    prompt_for_run, confirm_or_select_run, require_explicit_run_noninteractive
)


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


def load_analyses(analyses_dir: Path, manifest_ids: set[str] | None = None) -> list[dict]:
    """Load v4.x and v3.x schema analysis files.

    Args:
        analyses_dir: Directory containing analysis JSON files
        manifest_ids: Optional set of call_ids to filter by (scope coherence)
    """
    analyses = []
    skipped_not_in_manifest = 0
    version_counts = {}

    for f in analyses_dir.iterdir():
        if f.is_file() and f.suffix == '.json':
            # Scope coherence - filter by manifest if provided
            if manifest_ids is not None and f.stem not in manifest_ids:
                skipped_not_in_manifest += 1
                continue

            try:
                with open(f, 'r', encoding='utf-8') as fp:
                    data = json.load(fp)
                    # Accept v4.x and v3.x versions
                    version = data.get("schema_version", "")
                    if version.startswith("v4") or version.startswith("v3"):
                        analyses.append(data)
                        version_counts[version] = version_counts.get(version, 0) + 1
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {f}: {e}", file=sys.stderr)

    if skipped_not_in_manifest > 0:
        print(f"Scope filter: Skipped {skipped_not_in_manifest} analyses not in manifest", file=sys.stderr)

    if version_counts:
        versions_str = ", ".join(f"{v}: {c}" for v, c in sorted(version_counts.items()))
        print(f"Schema versions: {versions_str}", file=sys.stderr)

    return analyses


# Keep old function name for backwards compatibility
load_v3_analyses = load_analyses


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


def get_outcome(analysis: dict) -> str:
    """Get outcome/disposition from v4.0 or v3.x format."""
    # v4.0 uses disposition
    if "disposition" in analysis:
        return analysis["disposition"]
    # v3.x uses outcome or call_disposition
    return analysis.get("outcome") or analysis.get("call_disposition")


def extract_clarification_events(analysis: dict) -> list[dict]:
    """Extract clarification events from v4.0 (top-level), v3.8.5, and v3.8 formats."""
    events = []
    call_id = analysis.get("call_id")
    outcome = get_outcome(analysis)

    # v4.0 format: top-level clarifications array
    if "clarifications" in analysis and isinstance(analysis["clarifications"], list):
        for c in analysis["clarifications"]:
            events.append({
                "call_id": call_id,
                "type": CLARIFICATION_TYPE_MAP.get(c.get("type"), c.get("type")),
                "turn": c.get("turn") or c.get("t"),
                "resolved": None,
                "cause": CLARIFICATION_CAUSE_MAP.get(c.get("cause"), c.get("cause")),
                "context": c.get("note") or c.get("ctx"),
                "outcome": outcome
            })
        return events

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
    """Extract correction events from v4.0 (top-level), v3.8.5, and v3.8 formats."""
    events = []
    call_id = analysis.get("call_id")
    outcome = get_outcome(analysis)

    # v4.0 format: top-level corrections array
    if "corrections" in analysis and isinstance(analysis["corrections"], list):
        for c in analysis["corrections"]:
            sev = c.get("severity") or c.get("sev")
            events.append({
                "call_id": call_id,
                "what": None,
                "turn": c.get("turn") or c.get("t"),
                "frustrated": sev == "major",
                "severity": sev,
                "context": c.get("note") or c.get("ctx"),
                "outcome": outcome
            })
        return events

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
    """Extract loop events from v4.0 (top-level), v3.9.1/v3.8.5, and v3.8 formats.

    v4.0/v3.9.1: Includes subject field for granular loop analysis.
    """
    events = []
    call_id = analysis.get("call_id")
    outcome = get_outcome(analysis)

    # v4.0 format: top-level loops array
    if "loops" in analysis and isinstance(analysis["loops"], list):
        for l in analysis["loops"]:
            events.append({
                "call_id": call_id,
                "turns": l.get("turns") or l.get("t", []),  # v4.0 uses "turns"
                "type": l.get("type"),
                "subject": l.get("subject"),
                "context": l.get("note") or l.get("ctx"),
                "outcome": outcome
            })
        return events

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
    """Extract only the NL-relevant fields from a single analysis (~5-10 lines vs 18 fields).

    Handles both v4.0 and v3.x field names.
    """
    condensed = {
        "call_id": analysis.get("call_id"),
    }

    # v4.0 uses disposition, v3.x uses outcome + call_disposition
    disposition = analysis.get("disposition") or analysis.get("call_disposition")
    if disposition:
        condensed["disposition"] = disposition
    elif analysis.get("outcome"):
        condensed["outcome"] = analysis["outcome"]

    # v4.0 uses failure_type, v3.x uses failure_point
    failure_type = analysis.get("failure_type") or analysis.get("failure_point")
    if failure_type:
        condensed["failure_type"] = failure_type

    # v4.0: Include intent data
    if analysis.get("intent"):
        condensed["intent"] = analysis["intent"]
    if analysis.get("intent_context"):
        condensed["intent_context"] = analysis["intent_context"]

    # v4.0: Include sentiment data
    if analysis.get("sentiment_start"):
        condensed["sentiment_start"] = analysis["sentiment_start"]
    if analysis.get("sentiment_end"):
        condensed["sentiment_end"] = analysis["sentiment_end"]

    # Only include non-null NL fields (with v4.0/v3.x compatibility)
    # v4.0 uses failure_detail, v3.x uses failure_description
    failure_detail = analysis.get("failure_detail") or analysis.get("failure_description")
    if failure_detail:
        condensed["failure_detail"] = failure_detail

    # v4.0 uses verbatim, v3.x uses customer_verbatim
    verbatim = analysis.get("verbatim") or analysis.get("customer_verbatim")
    if verbatim:
        condensed["verbatim"] = verbatim

    # v4.0 uses coaching, v3.x uses agent_miss_detail
    coaching = analysis.get("coaching") or analysis.get("agent_miss_detail")
    if coaching:
        condensed["coaching"] = coaching

    # v4.0 uses policy_gap, v3.x uses policy_gap_detail
    policy_gap = analysis.get("policy_gap") or analysis.get("policy_gap_detail")
    if policy_gap:
        condensed["policy_gap"] = policy_gap

    # v4.0 uses steps, v3.x uses resolution_steps
    steps = analysis.get("steps") or analysis.get("resolution_steps")
    if steps:
        condensed["steps"] = steps

    # v4.0 uses failure_recoverable, v3.x uses was_recoverable
    recoverable = analysis.get("failure_recoverable") if "failure_recoverable" in analysis else analysis.get("was_recoverable")
    if recoverable is not None:
        condensed["recoverable"] = recoverable

    return condensed


def extract_nl_summary(analyses: list[dict]) -> dict:
    """
    Extract and organize NL fields into LLM-optimized structure.

    Output structure (v4.0):
    - by_failure_type: Grouped failure data for pattern analysis
    - all_verbatims: Customer quotes for voice-of-customer insights
    - all_coaching: Coaching opportunities (renamed from all_agent_misses)
    - policy_gap_details: Structured gap analysis
    - failed_call_flows: Resolution step sequences for failed calls
    - intent_data: Intent and context pairs for caller needs analysis (v4.0)
    - sentiment_data: Sentiment journeys for emotional tracking (v4.0)
    """

    # Group by failure type
    by_failure_type = defaultdict(list)
    all_verbatims = []
    all_coaching = []  # v4.0 renamed from all_agent_misses
    policy_gap_details = []
    failed_call_flows = []

    # v4.0: Intent and sentiment data
    intent_data = []
    sentiment_data = []

    calls_with_nl_data = 0

    for a in analyses:
        call_id = a.get("call_id")
        # v4.0 uses disposition, v3.x uses outcome
        outcome = get_outcome(a)
        # v4.0 uses failure_type, v3.x uses failure_point
        failure_type = a.get("failure_type") or a.get("failure_point", "none")

        has_nl_data = False

        # v4.0: Extract intent data
        if a.get("intent"):
            intent_entry = {
                "call_id": call_id,
                "intent": a["intent"],
                "outcome": outcome
            }
            if a.get("intent_context"):
                intent_entry["context"] = a["intent_context"]
            intent_data.append(intent_entry)
            has_nl_data = True

        # v4.0: Extract sentiment data
        if a.get("sentiment_start") or a.get("sentiment_end"):
            sentiment_data.append({
                "call_id": call_id,
                "start": a.get("sentiment_start"),
                "end": a.get("sentiment_end"),
                "outcome": outcome
            })
            has_nl_data = True

        # Group by failure type (excluding "none")
        if failure_type and failure_type != "none":
            entry = {
                "call_id": call_id,
                "outcome": outcome,
            }
            # v4.0 uses failure_detail, v3.x uses failure_description
            failure_detail = a.get("failure_detail") or a.get("failure_description")
            if failure_detail:
                entry["description"] = failure_detail
                has_nl_data = True
            # v4.0 uses verbatim, v3.x uses customer_verbatim
            verbatim = a.get("verbatim") or a.get("customer_verbatim")
            if verbatim:
                entry["verbatim"] = verbatim
                has_nl_data = True
            # v4.0 uses coaching, v3.x uses agent_miss_detail
            coaching = a.get("coaching") or a.get("agent_miss_detail")
            if coaching:
                entry["coaching"] = coaching
                has_nl_data = True

            by_failure_type[failure_type].append(entry)

        # Collect all verbatims (v4.0 uses verbatim, v3.x uses customer_verbatim)
        verbatim = a.get("verbatim") or a.get("customer_verbatim")
        if verbatim:
            all_verbatims.append({
                "call_id": call_id,
                "outcome": outcome,
                "quote": verbatim
            })
            has_nl_data = True

        # Collect all coaching (v4.0 uses coaching, v3.x uses agent_miss_detail)
        coaching = a.get("coaching") or a.get("agent_miss_detail")
        if coaching:
            # v4.0 uses failure_recoverable, v3.x uses was_recoverable
            recoverable = a.get("failure_recoverable") if "failure_recoverable" in a else a.get("was_recoverable")
            all_coaching.append({
                "call_id": call_id,
                "recoverable": recoverable,
                "coaching": coaching
            })
            has_nl_data = True

        # Collect policy gap details (v4.0 uses policy_gap, v3.x uses policy_gap_detail)
        policy_gap = a.get("policy_gap") or a.get("policy_gap_detail")
        if policy_gap:
            policy_gap_details.append({
                "call_id": call_id,
                "category": policy_gap.get("category"),
                "gap": policy_gap.get("specific_gap"),
                "ask": policy_gap.get("customer_ask"),
                "blocker": policy_gap.get("blocker")
            })
            has_nl_data = True

        # Collect failed call flows (v4.0 uses steps, v3.x uses resolution_steps)
        steps = a.get("steps") or a.get("resolution_steps")
        is_failed = outcome not in ("in_scope_success", "in_scope_partial", "resolved")
        if is_failed and steps:
            failed_call_flows.append({
                "call_id": call_id,
                "outcome": outcome,
                "failure_type": failure_type,
                "steps": steps
            })
            has_nl_data = True

        if has_nl_data:
            calls_with_nl_data += 1

    # Extract all customer_asks for LLM semantic clustering
    # This provides the raw list so LLM can cluster semantically similar asks
    all_customer_asks = []
    for detail in policy_gap_details:
        ask = detail.get("ask")
        if ask and ask.strip():
            all_customer_asks.append(ask.strip())

    # Training details with context (v3.x only - merged into coaching in v4.0)
    # Extracts training opportunities with associated failure info for narrative generation
    training_details = []
    for a in analyses:
        training_opp = a.get("training_opportunity")
        if training_opp:
            training_details.append({
                "call_id": a.get("call_id"),
                "opportunity": training_opp,
                "outcome": get_outcome(a),
                "failure_type": a.get("failure_type") or a.get("failure_point"),
                "failure_detail": a.get("failure_detail") or a.get("failure_description"),
                "coaching": a.get("coaching") or a.get("agent_miss_detail")
            })

    # Secondary intents (v4.0 uses secondary_intent, v3.x uses additional_intents)
    all_secondary_intents = []
    for a in analyses:
        secondary = a.get("secondary_intent") or a.get("additional_intents")
        if secondary:
            all_secondary_intents.append({
                "call_id": a.get("call_id"),
                "outcome": get_outcome(a),
                "intent": secondary
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

    # Extract disposition summary for LLM narrative generation (v4.0/v3.9+)
    disposition_summary = defaultdict(list)
    for a in analyses:
        # v4.0 uses disposition, v3.x uses call_disposition
        disposition = a.get("disposition") or a.get("call_disposition")
        if disposition:
            entry = {
                "call_id": a.get("call_id"),
            }
            # v4.0 uses failure_detail, v3.x uses failure_description
            failure_detail = a.get("failure_detail") or a.get("failure_description")
            if failure_detail:
                entry["description"] = failure_detail
            # v4.0 uses verbatim, v3.x uses customer_verbatim
            verbatim = a.get("verbatim") or a.get("customer_verbatim")
            if verbatim:
                entry["verbatim"] = verbatim
            if a.get("summary"):
                entry["summary"] = a["summary"]
            # v4.0: Include intent
            if a.get("intent"):
                entry["intent"] = a["intent"]
            disposition_summary[disposition].append(entry)

    # Count schema versions
    v4_count = sum(1 for a in analyses if a.get("schema_version", "").startswith("v4"))
    v3_count = sum(1 for a in analyses if a.get("schema_version", "").startswith("v3"))

    return {
        "metadata": {
            "extracted_at": datetime.now().isoformat(),
            "schema_version": "v4.0",
            "total_calls": len(analyses),
            "calls_with_nl_data": calls_with_nl_data,
            "source_versions": {"v4": v4_count, "v3": v3_count}
        },
        "by_failure_type": dict(by_failure_type),
        "all_verbatims": all_verbatims,
        "all_coaching": all_coaching,  # v4.0 renamed from all_agent_misses
        "all_agent_misses": all_coaching,  # v3.x compatibility alias
        "policy_gap_details": policy_gap_details,
        "failed_call_flows": failed_call_flows,
        "all_customer_asks": all_customer_asks,  # Raw list for LLM clustering
        "training_details": training_details,  # Training opportunities with context
        "all_secondary_intents": all_secondary_intents,  # v4.0 renamed from all_additional_intents
        "all_additional_intents": all_secondary_intents,  # v3.x compatibility alias
        "clarification_events": clarification_events,
        "correction_events": correction_events,
        "loop_events": loop_events,
        "loop_subject_pairs": loop_subject_pairs,  # (type, subject) pairs for clustering
        "turn_outliers": turn_outliers,
        "disposition_summary": dict(disposition_summary),
        # v4.0: New data sections
        "intent_data": intent_data,
        "sentiment_data": sentiment_data
    }


def print_summary(nl_summary: dict) -> None:
    """Print human-readable summary of extracted NL data."""
    print("\n" + "=" * 60)
    print("NL FIELD EXTRACTION SUMMARY (v4.0)")
    print("=" * 60)

    meta = nl_summary.get("metadata", {})
    print(f"\nTotal calls analyzed: {meta.get('total_calls', 0)}")
    print(f"Calls with NL data: {meta.get('calls_with_nl_data', 0)}")
    print(f"Extracted at: {meta.get('extracted_at', 'N/A')}")
    versions = meta.get("source_versions", {})
    if versions:
        print(f"Source versions: v4={versions.get('v4', 0)}, v3={versions.get('v3', 0)}")

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

    # v4.0: Intent Data
    intent_data = nl_summary.get("intent_data", [])
    if intent_data:
        print("\n" + "-" * 40)
        print("INTENT DATA (v4.0)")
        print("-" * 40)
        print(f"  Calls with intent: {len(intent_data)}")
        with_context = sum(1 for i in intent_data if i.get("context"))
        print(f"  With context: {with_context}")
        # Show top intents
        from collections import Counter
        intents = Counter(i.get("intent", "").lower() for i in intent_data if i.get("intent"))
        print("  Top intents:")
        for intent, count in intents.most_common(5):
            print(f"    {intent}: {count}")

    # v4.0: Sentiment Data
    sentiment_data = nl_summary.get("sentiment_data", [])
    if sentiment_data:
        print("\n" + "-" * 40)
        print("SENTIMENT DATA (v4.0)")
        print("-" * 40)
        print(f"  Calls with sentiment: {len(sentiment_data)}")
        from collections import Counter
        start_dist = Counter(s.get("start") for s in sentiment_data if s.get("start"))
        end_dist = Counter(s.get("end") for s in sentiment_data if s.get("end"))
        if start_dist:
            print("  Start sentiment:")
            for sent, count in start_dist.most_common():
                print(f"    {sent}: {count}")
        if end_dist:
            print("  End sentiment:")
            for sent, count in end_dist.most_common():
                print(f"    {sent}: {count}")

    # Coaching (renamed from Agent misses)
    print("\n" + "-" * 40)
    print("COACHING OPPORTUNITIES")
    print("-" * 40)
    coaching = nl_summary.get("all_coaching", []) or nl_summary.get("all_agent_misses", [])
    print(f"  Total coaching opportunities: {len(coaching)}")
    recoverable = sum(1 for m in coaching if m.get("recoverable") or m.get("was_recoverable"))
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
        description="Extract NL fields from v4/v3 analyses for LLM insight generation",
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

    # v4.1: Run-based isolation
    add_run_arguments(parser)

    args = parser.parse_args()

    # v4.1: Resolve run directory from --run-dir or --run-id
    project_dir = Path(__file__).parent.parent
    run_dir, run_id, source = resolve_run_from_args(args, project_dir)

    # Non-interactive mode requires explicit --run-id or --run-dir
    require_explicit_run_noninteractive(source)

    # Interactive run selection/confirmation
    if source in (".last_run", "$LAST_RUN"):
        # Implicit source - ask for confirmation
        run_dir, run_id = confirm_or_select_run(project_dir, run_dir, run_id, source)
    elif run_dir is None:
        # No run specified - show selection menu
        run_dir, run_id = prompt_for_run(project_dir)

    if run_dir:
        paths = get_run_paths(run_dir, project_dir)
        args.input_dir = paths["analyses_dir"]
        args.output_dir = paths["reports_dir"]
        args.sampled_dir = paths["sampled_dir"]
        print(f"Using run: {run_id} ({run_dir})", file=sys.stderr)

    # v3.3: Load manifest for scope coherence
    manifest_ids = None
    if not args.no_scope_filter:
        manifest_ids = load_manifest_ids(args.sampled_dir)
        if manifest_ids:
            print(f"Scope filter: Using manifest with {len(manifest_ids)} call IDs", file=sys.stderr)
        else:
            print("Scope filter: No manifest found, including all analyses", file=sys.stderr)

    print(f"Loading analyses from: {args.input_dir}", file=sys.stderr)
    analyses = load_analyses(args.input_dir, manifest_ids)

    if not analyses:
        print("Error: No v4/v3 schema analysis files found", file=sys.stderr)
        return 1

    print(f"Found {len(analyses)} analyses", file=sys.stderr)

    # Apply limit if specified
    if args.limit and args.limit < len(analyses):
        analyses = analyses[:args.limit]
        print(f"Limited to {args.limit} analyses", file=sys.stderr)

    # Extract NL summary
    nl_summary = extract_nl_summary(analyses)

    # Save output
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"nl_summary_v4_{timestamp}.json"

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
