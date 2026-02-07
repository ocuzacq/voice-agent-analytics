#!/usr/bin/env python3
"""
Aggregate Metrics Calculator for Vacatia AI Voice Agent Analytics (v5.0)

Computes Section A: Deterministic Metrics - all Python-calculated, reproducible and auditable.

v5.0 changes:
- Orthogonal disposition model: call_scope × call_outcome (replaces single disposition enum)
- New scope-outcome cross-tabulation replaces disposition_breakdown
- Call funnel uses call_scope (no_request / out_of_scope / in_scope)
- In-scope outcomes use call_outcome + conditional qualifiers
  (escalation_trigger, abandon_stage, resolution_confirmed)
- Handle time grouped by scope and outcome
- Backward compatible with v3.x/v4.x via get_disposition() bridge

v4.4: Handle time / AHT (duration_seconds field, per-disposition breakdown)
v4.1: Run-based isolation support via --run-dir argument
v4.0: Intent/sentiment tracking, flattened friction, unified disposition
v3.9: Call disposition classification, funnel metrics
v3.8.5: Compact friction parsing

Note: NL field extraction is now handled by the dedicated extract_nl_fields.py script.
"""

import argparse
import csv
import json
import statistics
import sys
from collections import Counter
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
            # Extract call_id from filename (e.g., "abc123.txt" -> "abc123")
            filename = row.get("filename", "")
            if filename:
                call_ids.add(Path(filename).stem)
    return call_ids


def load_analyses(analyses_dir: Path, schema_version: str = "v4", manifest_ids: set[str] | None = None) -> list[dict]:
    """Load analysis JSON files from directory matching the specified schema version.

    Args:
        analyses_dir: Directory containing analysis JSON files
        schema_version: Schema version to filter by ("v4", "v3", or "v2")
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
                    version = data.get("schema_version", "v2")

                    # Track version distribution
                    version_counts[version] = version_counts.get(version, 0) + 1

                    # v4.0+: Accept v5.x, v4.x, v3.x, or v2 (all backwards-compatible)
                    if schema_version == "v4":
                        if version.startswith("v5") or version.startswith("v4") or version.startswith("v3") or version == "v2":
                            analyses.append(data)
                    # v3: Accept v3.x or v2
                    elif schema_version == "v3":
                        if version.startswith("v3") or version == "v2":
                            analyses.append(data)
                    elif schema_version == "v2" and version == "v2":
                        analyses.append(data)
            except (json.JSONDecodeError, IOError) as e:
                print(f"Warning: Could not load {f}: {e}", file=sys.stderr)

    if skipped_not_in_manifest > 0:
        print(f"Scope filter: Skipped {skipped_not_in_manifest} analyses not in manifest", file=sys.stderr)

    if version_counts:
        versions_str = ", ".join(f"{v}: {c}" for v, c in sorted(version_counts.items()))
        print(f"Schema versions found: {versions_str}", file=sys.stderr)

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


# v3.8.5 Enum mapping: short values → canonical values for aggregation
CLARIFICATION_TYPE_MAP = {
    # v3.8.5 short values
    "name": "name_spelling",
    "phone": "phone_confirmation",
    "intent": "intent_clarification",
    "repeat": "repeat_request",
    "verify": "verification_retry",
    # v3.8 long values (pass through)
    "name_spelling": "name_spelling",
    "phone_confirmation": "phone_confirmation",
    "intent_clarification": "intent_clarification",
    "repeat_request": "repeat_request",
    "verification_retry": "verification_retry",
}

CLARIFICATION_CAUSE_MAP = {
    # v3.8.5 short values
    "misheard": "agent_misheard",
    "unclear": "customer_unclear",
    "refused": "customer_refused",
    "tech": "tech_issue",
    "ok": "successful",
    # v3.8 long values (pass through)
    "agent_misheard": "agent_misheard",
    "customer_unclear": "customer_unclear",
    "customer_refused": "customer_refused",
    "tech_issue": "tech_issue",
    "successful": "successful",
}


def parse_clarifications(analysis: dict) -> list[tuple]:
    """
    Parse clarifications from v4.0 (top-level), v3.8.5 (compact), and v3.8 (verbose) formats.

    Returns: list of (turn, type, cause, context) tuples with canonical enum values.
    """
    results = []

    # v4.0 format: top-level clarifications array
    if "clarifications" in analysis and isinstance(analysis["clarifications"], list):
        for c in analysis["clarifications"]:
            turn = c.get("turn") or c.get("t")  # v4.0 uses "turn", v3.x uses "t"
            ctype = CLARIFICATION_TYPE_MAP.get(c.get("type"), c.get("type"))
            cause = CLARIFICATION_CAUSE_MAP.get(c.get("cause"), c.get("cause"))
            ctx = c.get("note") or c.get("ctx")  # v4.0 uses "note", v3.x uses "ctx"
            results.append((turn, ctype, cause, ctx))
        return results

    # v3.8.5 format: friction.clarifications
    friction = analysis.get("friction")
    if friction and isinstance(friction, dict):
        for c in friction.get("clarifications", []):
            turn = c.get("t")
            ctype = CLARIFICATION_TYPE_MAP.get(c.get("type"), c.get("type"))
            cause = CLARIFICATION_CAUSE_MAP.get(c.get("cause"), c.get("cause"))
            ctx = c.get("ctx")
            results.append((turn, ctype, cause, ctx))
        return results

    # v3.8 format: clarification_requests.details
    clar = analysis.get("clarification_requests")
    if clar and isinstance(clar, dict):
        for d in clar.get("details", []):
            turn = d.get("turn")
            ctype = CLARIFICATION_TYPE_MAP.get(d.get("type"), d.get("type"))
            cause = CLARIFICATION_CAUSE_MAP.get(d.get("cause"), d.get("cause"))
            ctx = d.get("context")
            results.append((turn, ctype, cause, ctx))

    return results


def parse_corrections(analysis: dict) -> list[tuple]:
    """
    Parse corrections from v4.0 (top-level), v3.8.5 (compact), and v3.8 (verbose) formats.

    Returns: list of (turn, severity, context, frustrated) tuples.
    """
    results = []

    # v4.0 format: top-level corrections array
    if "corrections" in analysis and isinstance(analysis["corrections"], list):
        for c in analysis["corrections"]:
            turn = c.get("turn") or c.get("t")  # v4.0 uses "turn"
            sev = c.get("severity") or c.get("sev")  # v4.0 uses "severity"
            ctx = c.get("note") or c.get("ctx")  # v4.0 uses "note"
            # Infer frustrated from severity
            frustrated = sev == "major"
            results.append((turn, sev, ctx, frustrated))
        return results

    # v3.8.5 format: friction.corrections
    friction = analysis.get("friction")
    if friction and isinstance(friction, dict):
        for c in friction.get("corrections", []):
            turn = c.get("t")
            sev = c.get("sev")
            ctx = c.get("ctx")
            # v3.8.5 doesn't track frustrated separately - infer from severity
            frustrated = sev == "major"
            results.append((turn, sev, ctx, frustrated))
        return results

    # v3.8 format: user_corrections.details
    corr = analysis.get("user_corrections")
    if corr and isinstance(corr, dict):
        for d in corr.get("details", []):
            turn = d.get("turn")
            sev = d.get("severity")
            ctx = d.get("context")
            frustrated = d.get("frustration_signal", False)
            results.append((turn, sev, ctx, frustrated))

    return results


def parse_loops(analysis: dict) -> list[tuple]:
    """
    Parse loops from v4.0 (top-level), v3.9.1/v3.8.5 (compact), and v3.8 (verbose) formats.

    Returns: list of (turns, type, subject, context) tuples.
    - turns: list of turn numbers (v4.0/v3.8.5+) or None (v3.8)
    - subject: what is being looped on (v4.0/v3.9.1) or None (earlier versions)
    """
    results = []

    # v4.0 format: top-level loops array
    if "loops" in analysis and isinstance(analysis["loops"], list):
        for l in analysis["loops"]:
            turns = l.get("turns") or l.get("t", [])  # v4.0 uses "turns"
            loop_type = l.get("type")
            subject = l.get("subject")
            ctx = l.get("note") or l.get("ctx")  # v4.0 uses "note"
            results.append((turns, loop_type, subject, ctx))
        return results

    # v3.9.1/v3.8.5 format: friction.loops
    friction = analysis.get("friction")
    if friction and isinstance(friction, dict):
        for l in friction.get("loops", []):
            turns = l.get("t", [])  # Array of turn numbers
            loop_type = l.get("type")
            subject = l.get("subject")  # v3.9.1: what is being looped on
            ctx = l.get("ctx")
            results.append((turns, loop_type, subject, ctx))
        return results

    # v3.8 format: agent_loops.details
    loops = analysis.get("agent_loops")
    if loops and isinstance(loops, dict):
        for d in loops.get("details", []):
            loop_type = d.get("type")
            ctx = d.get("context")
            results.append((None, loop_type, None, ctx))  # v3.8 has no turn numbers or subject
        return results

    # v3.7 format: repeated_prompts (legacy)
    rp = analysis.get("repeated_prompts")
    if rp and isinstance(rp, dict):
        count = rp.get("count", 0)
        if count > 0:
            # Legacy format: single entry with count info
            results.append((None, None, None, f"Legacy: {count} repeats"))

    return results


def get_conversation_turns(analysis: dict) -> int | None:
    """Get conversation turns from v4.0 (top-level), v3.8.5 (friction.turns), or v3.8 (conversation_turns)."""
    # v4.0 format: top-level turns
    if "turns" in analysis and isinstance(analysis["turns"], (int, float)):
        return analysis["turns"]

    # v3.8.5 format
    friction = analysis.get("friction")
    if friction and isinstance(friction, dict):
        turns = friction.get("turns")
        if turns is not None:
            return turns

    # v3.8 format
    return analysis.get("conversation_turns")


def get_turns_to_failure(analysis: dict) -> int | None:
    """Get turns_to_failure from v4.0 (top-level), v3.8.5 (friction.derailed_at), or v3.8 (turns_to_failure)."""
    # v4.0 format: top-level derailed_at
    if "derailed_at" in analysis:
        return analysis["derailed_at"]

    # v3.8.5 format
    friction = analysis.get("friction")
    if friction and isinstance(friction, dict):
        return friction.get("derailed_at")

    # v3.8 format
    return analysis.get("turns_to_failure")


def validate_failure_consistency(analyses: list[dict]) -> dict:
    """
    Flag failure_type inconsistencies (v5.0/v4.0/v3.x).

    For non-completed calls, failure_type should typically be populated.
    """
    warnings = []
    for a in analyses:
        failure_type = a.get("failure_type") or a.get("failure_point")

        # v5.0: escalated calls should have failure_type
        if a.get("call_outcome") == "escalated" and not failure_type:
            warnings.append({
                "call_id": a.get("call_id"),
                "call_scope": a.get("call_scope"),
                "call_outcome": "escalated",
                "issue": "failure_type missing for escalated call"
            })
        # v4.0: Failed dispositions should have failure_type
        elif "disposition" in a:
            disposition = get_disposition(a)
            if disposition in ("in_scope_failed", "out_of_scope_failed", "escalated"):
                if not failure_type:
                    warnings.append({
                        "call_id": a.get("call_id"),
                        "disposition": disposition,
                        "issue": "failure_type missing for failed disposition"
                    })
        # v3.x compatibility
        else:
            outcome = a.get("outcome")
            if outcome in ("abandoned", "escalated", "unclear") and failure_type == "none":
                warnings.append({
                    "call_id": a.get("call_id"),
                    "outcome": outcome,
                    "issue": "failure_type='none' invalid for non-resolved call"
                })

    return {"failure_consistency_warnings": warnings}


def extract_policy_gap_breakdown(analyses: list[dict]) -> dict:
    """Extract policy gap analysis from v4.0 (policy_gap) or v3.x (policy_gap_detail) fields."""
    # Filter to policy gap failures (v4.0 uses failure_type, v3.x uses failure_point)
    policy_gaps = []
    for a in analyses:
        failure_type = a.get("failure_type") or a.get("failure_point")
        if failure_type == "policy_gap":
            policy_gaps.append(a)

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
        # v4.0 uses policy_gap, v3.x uses policy_gap_detail
        detail = a.get("policy_gap") or a.get("policy_gap_detail")
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


def get_disposition(analysis: dict) -> str | None:
    """Get disposition with v5.0/v4.0/v3.x compatibility.

    v5.0: Synthesizes legacy disposition from call_scope × call_outcome.
    v4.0: Uses unified "disposition" field.
    v3.x: Uses "call_disposition".
    """
    # v5.0: synthesize from orthogonal dimensions
    if "call_scope" in analysis and "call_outcome" in analysis:
        scope = analysis["call_scope"]
        outcome = analysis["call_outcome"]
        if scope == "no_request":
            return "pre_intent"
        elif scope == "out_of_scope":
            return "out_of_scope_handled" if outcome == "completed" else "out_of_scope_failed"
        elif scope in ("in_scope", "mixed"):
            if outcome == "completed":
                return "in_scope_success"
            elif outcome == "escalated":
                return "escalated"
            elif outcome == "abandoned":
                return "in_scope_failed"

    # v4.0 format: unified disposition field
    if "disposition" in analysis:
        return analysis["disposition"]

    # v3.x format: call_disposition
    return analysis.get("call_disposition")


def compute_disposition_breakdown(analyses: list[dict]) -> dict:
    """
    Compute call disposition breakdown (v5.0/v4.0/v3.9+).

    v5.0: Returns scope × outcome cross-tabulation with key containment metrics.
    v4.x/v3.x: Returns legacy disposition-based breakdown.
    """
    n = len(analyses)
    if n == 0:
        return {}

    has_v5 = any("call_scope" in a for a in analyses)

    if has_v5:
        # v5.0: Scope and outcome distributions
        scopes = Counter(a.get("call_scope") for a in analyses)
        outcomes = Counter(a.get("call_outcome") for a in analyses)

        # Cross-tab: scope × outcome
        cross_tab = {}
        for a in analyses:
            scope = a.get("call_scope", "unknown")
            outcome = a.get("call_outcome", "unknown")
            key = f"{scope}:{outcome}"
            cross_tab[key] = cross_tab.get(key, 0) + 1

        # Key metrics
        in_scope_count = sum(1 for a in analyses if a.get("call_scope") in ("in_scope", "mixed"))
        in_scope_completed = sum(1 for a in analyses
                                  if a.get("call_scope") in ("in_scope", "mixed")
                                  and a.get("call_outcome") == "completed")
        total_escalated = outcomes.get("escalated", 0)

        return {
            "by_scope": {
                k: {"count": v, "rate": safe_rate(v, n)}
                for k, v in scopes.most_common()
                if k is not None
            },
            "by_outcome": {
                k: {"count": v, "rate": safe_rate(v, n)}
                for k, v in outcomes.most_common()
                if k is not None
            },
            "cross_tab": {
                k: {"count": v, "rate": safe_rate(v, n)}
                for k, v in sorted(cross_tab.items(), key=lambda x: -x[1])
            },
            "funnel_metrics": {
                "containment_rate": safe_rate(in_scope_completed, in_scope_count),
                "escalation_rate": safe_rate(total_escalated, n),
                "in_scope_total": in_scope_count,
                "calls_with_scope": n - scopes.get(None, 0),
            }
        }
    else:
        # v4.x/v3.x: Legacy disposition-based breakdown
        valid_dispositions = [
            "pre_intent", "out_of_scope_handled", "out_of_scope_abandoned",
            "out_of_scope_failed", "in_scope_success", "in_scope_partial",
            "in_scope_failed", "escalated"
        ]

        dispositions = Counter()
        for a in analyses:
            disposition = get_disposition(a)
            if disposition and disposition in valid_dispositions:
                dispositions[disposition] += 1
            else:
                dispositions["unknown"] = dispositions.get("unknown", 0) + 1

        in_scope_total = (
            dispositions.get("in_scope_success", 0) +
            dispositions.get("in_scope_partial", 0) +
            dispositions.get("in_scope_failed", 0)
        )
        in_scope_confirmed = dispositions.get("in_scope_success", 0)
        out_of_scope_total = (
            dispositions.get("out_of_scope_handled", 0) +
            dispositions.get("out_of_scope_abandoned", 0) +
            dispositions.get("out_of_scope_failed", 0)
        )
        out_of_scope_handled = dispositions.get("out_of_scope_handled", 0)
        escalated_count = dispositions.get("escalated", 0)

        return {
            "by_disposition": {
                k: {"count": v, "rate": safe_rate(v, n)}
                for k, v in dispositions.most_common()
            },
            "funnel_metrics": {
                "containment_rate": safe_rate(in_scope_confirmed, in_scope_total),
                "out_of_scope_recovery_rate": safe_rate(out_of_scope_handled, out_of_scope_total),
                "escalation_rate": safe_rate(escalated_count, n),
                "in_scope_total": in_scope_total,
                "out_of_scope_total": out_of_scope_total,
                "calls_with_disposition": n - dispositions.get("unknown", 0)
            }
        }


def compute_intent_stats(analyses: list[dict]) -> dict:
    """
    Compute intent statistics from v4.0 analyses (v4.0 only).

    v4.0 adds:
    - intent: Primary customer request (normalized phrase)
    - intent_context: Why they need it (underlying situation)
    - secondary_intent: Additional request beyond main intent
    """
    # Count analyses with intent data
    intents = []
    intent_contexts = []
    secondary_intents = []

    for a in analyses:
        if a.get("intent"):
            intents.append(a["intent"].lower().strip())
        if a.get("intent_context"):
            intent_contexts.append(a["intent_context"])
        # v4.0 uses secondary_intent, v3.x uses additional_intents
        secondary = a.get("secondary_intent") or a.get("additional_intents")
        if secondary:
            secondary_intents.append(secondary)

    if not intents:
        return {}

    # Count intent frequency
    intent_counts = Counter(intents)

    return {
        "total_with_intent": len(intents),
        "total_with_context": len(intent_contexts),
        "total_with_secondary": len(secondary_intents),
        "top_intents": [
            {"intent": intent, "count": count}
            for intent, count in intent_counts.most_common(15)
        ],
        "coverage_rate": safe_rate(len(intents), len(analyses))
    }


def compute_sentiment_stats(analyses: list[dict]) -> dict:
    """
    Compute sentiment statistics from v4.0 analyses (v4.0 only).

    v4.0 adds:
    - sentiment_start: Customer mood at conversation start
    - sentiment_end: Customer mood at conversation end

    Values: positive, neutral, frustrated, angry (start)
            satisfied, neutral, frustrated, angry (end)
    """
    start_sentiments = Counter()
    end_sentiments = Counter()
    journeys = Counter()  # (start, end) pairs

    for a in analyses:
        start = a.get("sentiment_start")
        end = a.get("sentiment_end")

        if start:
            start_sentiments[start] += 1
        if end:
            end_sentiments[end] += 1
        if start and end:
            journeys[(start, end)] += 1

    if not start_sentiments and not end_sentiments:
        return {}

    total_with_sentiment = sum(start_sentiments.values())

    return {
        "start_distribution": {
            k: {"count": v, "rate": safe_rate(v, total_with_sentiment)}
            for k, v in start_sentiments.most_common()
        },
        "end_distribution": {
            k: {"count": v, "rate": safe_rate(v, total_with_sentiment)}
            for k, v in end_sentiments.most_common()
        },
        "top_journeys": [
            {"from": start, "to": end, "count": count}
            for (start, end), count in journeys.most_common(10)
        ],
        "improvement_rate": safe_rate(
            sum(1 for a in analyses if _sentiment_improved(a)),
            total_with_sentiment
        ),
        "degradation_rate": safe_rate(
            sum(1 for a in analyses if _sentiment_degraded(a)),
            total_with_sentiment
        ),
        "total_with_sentiment": total_with_sentiment
    }


def _sentiment_improved(analysis: dict) -> bool:
    """Check if sentiment improved from start to end."""
    start = analysis.get("sentiment_start")
    end = analysis.get("sentiment_end")
    if not start or not end:
        return False

    # Sentiment levels (higher = better)
    levels = {"angry": 0, "frustrated": 1, "neutral": 2, "positive": 3, "satisfied": 3}
    return levels.get(end, 2) > levels.get(start, 2)


def _sentiment_degraded(analysis: dict) -> bool:
    """Check if sentiment degraded from start to end."""
    start = analysis.get("sentiment_start")
    end = analysis.get("sentiment_end")
    if not start or not end:
        return False

    levels = {"angry": 0, "frustrated": 1, "neutral": 2, "positive": 3, "satisfied": 3}
    return levels.get(end, 2) < levels.get(start, 2)


def compute_handle_time_stats(analyses: list[dict]) -> dict:
    """
    Compute handle time statistics from v4.4+ duration_seconds field.

    Returns overall AHT stats plus breakdowns by scope and outcome.
    v5.0: Groups by call_scope and call_outcome.
    v4.x: Falls back to legacy disposition grouping.
    """
    durations = [a["duration_seconds"] for a in analyses if a.get("duration_seconds") is not None]

    if not durations:
        return {}

    result = {"overall": safe_stats(durations)}

    # v5.0: Group by call_scope and call_outcome
    has_v5 = any("call_scope" in a for a in analyses)
    if has_v5:
        by_scope = {}
        for scope in ("in_scope", "out_of_scope", "mixed", "no_request"):
            scope_durations = [
                a["duration_seconds"] for a in analyses
                if a.get("call_scope") == scope and a.get("duration_seconds") is not None
            ]
            if scope_durations:
                by_scope[scope] = safe_stats(scope_durations)
        result["by_scope"] = by_scope

        by_outcome = {}
        for outcome in ("completed", "escalated", "abandoned"):
            outcome_durations = [
                a["duration_seconds"] for a in analyses
                if a.get("call_outcome") == outcome and a.get("duration_seconds") is not None
            ]
            if outcome_durations:
                by_outcome[outcome] = safe_stats(outcome_durations)
        result["by_outcome"] = by_outcome
    else:
        # Legacy: group by disposition
        disposition_values = [
            "pre_intent", "out_of_scope_handled", "out_of_scope_failed",
            "in_scope_success", "in_scope_partial", "in_scope_failed", "escalated"
        ]
        by_disposition = {}
        for disp in disposition_values:
            disp_durations = [
                a["duration_seconds"] for a in analyses
                if get_disposition(a) == disp and a.get("duration_seconds") is not None
            ]
            if disp_durations:
                by_disposition[disp] = safe_stats(disp_durations)
        result["by_disposition"] = by_disposition

    return result


def compute_call_funnel(analyses: list[dict]) -> dict:
    """
    Compute MECE call funnel from entry (v5.0 dashboard).

    v5.0: Uses call_scope for scope split, call_outcome for outcome split.
    v4.x fallback: Uses legacy disposition values.
    """
    n = len(analyses)
    if n == 0:
        return {}

    has_v5 = any("call_scope" in a for a in analyses)

    if has_v5:
        # v5.0: Scope-based funnel
        no_request = [a for a in analyses if a.get("call_scope") == "no_request"]
        request_made = [a for a in analyses if a.get("call_scope") != "no_request"]

        no_request_count = len(no_request)
        request_made_count = len(request_made)

        # No-request breakdown by abandon_stage
        abandon_stages = Counter(a.get("abandon_stage") for a in no_request)

        # Request-made scope breakdown
        in_scope = [a for a in request_made if a.get("call_scope") == "in_scope"]
        out_of_scope = [a for a in request_made if a.get("call_scope") == "out_of_scope"]
        mixed_scope = [a for a in request_made if a.get("call_scope") == "mixed"]

        in_scope_count = len(in_scope)
        out_of_scope_count = len(out_of_scope)
        mixed_count = len(mixed_scope)

        return {
            "total": n,
            "no_request": {
                "count": no_request_count,
                "rate": safe_rate(no_request_count, n),
                "by_stage": {
                    k: {"count": v, "rate": safe_rate(v, no_request_count)}
                    for k, v in abandon_stages.most_common()
                    if k is not None
                }
            },
            "request_made": {
                "count": request_made_count,
                "rate": safe_rate(request_made_count, n),
                "in_scope": {
                    "count": in_scope_count,
                    "rate": safe_rate(in_scope_count, request_made_count),
                },
                "out_of_scope": {
                    "count": out_of_scope_count,
                    "rate": safe_rate(out_of_scope_count, request_made_count),
                },
                "mixed": {
                    "count": mixed_count,
                    "rate": safe_rate(mixed_count, request_made_count),
                },
            },
            "_invariants": {
                "no_request_plus_request_eq_total": no_request_count + request_made_count == n,
                "scope_sum_eq_request_made": in_scope_count + out_of_scope_count + mixed_count == request_made_count,
            }
        }
    else:
        # v4.x fallback: disposition-based funnel
        pre_intent_calls = [a for a in analyses if get_disposition(a) == "pre_intent"]
        intent_captured_calls = [a for a in analyses if get_disposition(a) != "pre_intent"]

        pre_intent_count = len(pre_intent_calls)
        intent_captured_count = len(intent_captured_calls)

        pre_intent_subtypes = Counter(a.get("pre_intent_subtype") for a in pre_intent_calls)

        oos_dispositions = {"out_of_scope_handled", "out_of_scope_failed"}
        is_dispositions = {"in_scope_success", "in_scope_partial", "in_scope_failed", "escalated"}

        out_of_scope_count = sum(1 for a in intent_captured_calls if get_disposition(a) in oos_dispositions)
        in_scope_count = sum(1 for a in intent_captured_calls if get_disposition(a) in is_dispositions)

        return {
            "total": n,
            "no_request": {
                "count": pre_intent_count,
                "rate": safe_rate(pre_intent_count, n),
                "by_stage": {
                    k: {"count": v, "rate": safe_rate(v, pre_intent_count)}
                    for k, v in pre_intent_subtypes.most_common()
                    if k is not None
                }
            },
            "request_made": {
                "count": intent_captured_count,
                "rate": safe_rate(intent_captured_count, n),
                "in_scope": {
                    "count": in_scope_count,
                    "rate": safe_rate(in_scope_count, intent_captured_count),
                },
                "out_of_scope": {
                    "count": out_of_scope_count,
                    "rate": safe_rate(out_of_scope_count, intent_captured_count),
                },
                "mixed": {
                    "count": 0,
                    "rate": None,
                },
            },
            "_invariants": {
                "no_request_plus_request_eq_total": pre_intent_count + intent_captured_count == n,
                "scope_sum_eq_request_made": out_of_scope_count + in_scope_count == intent_captured_count,
            }
        }


def compute_in_scope_outcomes(analyses: list[dict]) -> dict:
    """
    Compute in-scope outcome breakdown (v5.0 dashboard).

    v5.0: Base = calls with call_scope in {in_scope, mixed}.
          Outcomes split by call_outcome (completed/escalated/abandoned).
    v4.x fallback: Uses legacy disposition values.
    """
    has_v5 = any("call_scope" in a for a in analyses)

    if has_v5:
        # v5.0: in_scope and mixed-scope calls
        in_scope = [a for a in analyses if a.get("call_scope") in ("in_scope", "mixed")]
        is_count = len(in_scope)
        if is_count == 0:
            return {}

        # Completed
        completed = [a for a in in_scope if a.get("call_outcome") == "completed"]
        completed_count = len(completed)
        confirmed_count = sum(1 for a in completed if a.get("resolution_confirmed") is True)
        unconfirmed_count = sum(1 for a in completed if a.get("resolution_confirmed") is False)

        # Escalated
        escalated = [a for a in in_scope if a.get("call_outcome") == "escalated"]
        escalated_count = len(escalated)
        trigger_counts = Counter(a.get("escalation_trigger") for a in escalated)

        # Abandoned
        abandoned = [a for a in in_scope if a.get("call_outcome") == "abandoned"]
        abandoned_count = len(abandoned)
        stage_counts = Counter(a.get("abandon_stage") for a in abandoned)

        return {
            "in_scope_total": is_count,
            "completed": {
                "count": completed_count,
                "rate": safe_rate(completed_count, is_count),
                "confirmed": {"count": confirmed_count, "rate": safe_rate(confirmed_count, completed_count)},
                "unconfirmed": {"count": unconfirmed_count, "rate": safe_rate(unconfirmed_count, completed_count)},
            },
            "escalated": {
                "count": escalated_count,
                "rate": safe_rate(escalated_count, is_count),
                "by_trigger": {
                    k: {"count": v, "rate": safe_rate(v, escalated_count)}
                    for k, v in trigger_counts.most_common()
                    if k is not None
                },
            },
            "abandoned": {
                "count": abandoned_count,
                "rate": safe_rate(abandoned_count, is_count),
                "by_stage": {
                    k: {"count": v, "rate": safe_rate(v, abandoned_count)}
                    for k, v in stage_counts.most_common()
                    if k is not None
                },
            },
            "_invariant": {
                "sum_eq_total": completed_count + escalated_count + abandoned_count == is_count,
            }
        }
    else:
        # v4.x fallback
        is_dispositions = {"in_scope_success", "in_scope_partial", "in_scope_failed", "escalated"}
        in_scope = [a for a in analyses if get_disposition(a) in is_dispositions]
        is_count = len(in_scope)
        if is_count == 0:
            return {}

        resolved = [a for a in in_scope if get_disposition(a) in {"in_scope_success", "in_scope_partial"}]
        resolved_count = len(resolved)
        confirmed_count = sum(1 for a in resolved if a.get("resolution_confirmed") is True)
        unconfirmed_count = sum(1 for a in resolved if a.get("resolution_confirmed") is False)

        escalated = [a for a in in_scope if get_disposition(a) == "escalated"]
        escalated_count = len(escalated)

        abandoned = [a for a in in_scope if get_disposition(a) == "in_scope_failed" and a.get("ended_by") == "customer"]
        abandoned_count = len(abandoned)

        failed_other = [a for a in in_scope if get_disposition(a) == "in_scope_failed" and a.get("ended_by") != "customer"]
        failed_other_count = len(failed_other)

        return {
            "in_scope_total": is_count,
            "completed": {
                "count": resolved_count,
                "rate": safe_rate(resolved_count, is_count),
                "confirmed": {"count": confirmed_count, "rate": safe_rate(confirmed_count, resolved_count)},
                "unconfirmed": {"count": unconfirmed_count, "rate": safe_rate(unconfirmed_count, resolved_count)},
            },
            "escalated": {
                "count": escalated_count,
                "rate": safe_rate(escalated_count, is_count),
                "by_trigger": {},
            },
            "abandoned": {
                "count": abandoned_count + failed_other_count,
                "rate": safe_rate(abandoned_count + failed_other_count, is_count),
                "by_stage": {},
            },
            "_invariant": {
                "sum_eq_total": resolved_count + escalated_count + abandoned_count + failed_other_count == is_count,
            }
        }


def compute_action_performance(analyses: list[dict]) -> dict:
    """
    Compute per-action-type performance metrics (v4.5 dashboard).

    Counts attempted, success, retry, failed, unknown per action type.
    """
    action_stats = {}  # action_type -> {attempted, success, retry, failed, unknown}

    for a in analyses:
        for act in a.get("actions", []):
            action_type = act.get("action", "unknown")
            outcome = act.get("outcome", "unknown")

            if action_type not in action_stats:
                action_stats[action_type] = Counter()
            action_stats[action_type]["attempted"] += 1
            action_stats[action_type][outcome] += 1

    if not action_stats:
        return {}

    result = {}
    for action_type, counts in sorted(action_stats.items(), key=lambda x: -x[1]["attempted"]):
        attempted = counts["attempted"]
        result[action_type] = {
            "attempted": attempted,
            "success": counts.get("success", 0),
            "retry": counts.get("retry", 0),
            "failed": counts.get("failed", 0),
            "unknown": counts.get("unknown", 0),
            "success_rate": safe_rate(counts.get("success", 0), attempted),
            "retry_rate": safe_rate(counts.get("retry", 0), attempted),
            "failure_rate": safe_rate(counts.get("failed", 0), attempted),
        }

    # Overall stats
    total_actions = sum(c["attempted"] for c in action_stats.values())
    total_success = sum(c.get("success", 0) for c in action_stats.values())
    calls_with_actions = sum(1 for a in analyses if a.get("actions"))

    return {
        "by_action": result,
        "overall": {
            "total_actions": total_actions,
            "calls_with_actions": calls_with_actions,
            "overall_success_rate": safe_rate(total_success, total_actions),
            "action_types_seen": len(action_stats),
        }
    }


def compute_transfer_quality(analyses: list[dict]) -> dict:
    """
    Compute transfer quality metrics (v4.5 dashboard).

    Base = all calls with transfer_destination not null.
    """
    transfers = [a for a in analyses if a.get("transfer_destination") is not None]
    total_transfers = len(transfers)

    if total_transfers == 0:
        return {}

    # By destination
    destinations = Counter(a.get("transfer_destination") for a in transfers)

    # Queue detection rate
    queue_detected = sum(1 for a in transfers if a.get("transfer_queue_detected") is True)

    return {
        "total_transfers": total_transfers,
        "by_destination": {
            dest: {"count": count, "rate": safe_rate(count, total_transfers)}
            for dest, count in destinations.most_common()
        },
        "queue_detected": {
            "count": queue_detected,
            "rate": safe_rate(queue_detected, total_transfers),
        }
    }


def compute_conversation_quality_metrics(analyses: list[dict]) -> dict:
    """
    Compute conversation quality metrics from v3.8.5/v3.8/v3.6 fields.

    v3.8.5: Uses parse_clarifications(), parse_corrections(), parse_loops() for
    backwards-compatible parsing of both compact (friction) and verbose formats.

    Aggregates:
    - Turn statistics (avg, median, by outcome, turns-to-failure)
    - Clarification request statistics (by type, by cause)
    - User correction statistics (frustration rate, by severity)
    - Loop detection statistics (calls with loops, by type, loop density)
    """
    n = len(analyses)
    if n == 0:
        return {}

    # === Turn Statistics ===
    all_turns = []
    resolved_turns = []
    failed_turns = []
    turns_to_failure_list = []

    for a in analyses:
        turns = get_conversation_turns(a)
        if turns is not None and isinstance(turns, (int, float)):
            all_turns.append(turns)
            if a.get("outcome") == "resolved":
                resolved_turns.append(turns)
            else:
                failed_turns.append(turns)

        ttf = get_turns_to_failure(a)
        if ttf is not None and isinstance(ttf, (int, float)):
            turns_to_failure_list.append(ttf)

    turn_stats = {
        "avg_turns": round(statistics.mean(all_turns), 1) if all_turns else None,
        "median_turns": round(statistics.median(all_turns), 1) if all_turns else None,
        "avg_turns_resolved": round(statistics.mean(resolved_turns), 1) if resolved_turns else None,
        "avg_turns_failed": round(statistics.mean(failed_turns), 1) if failed_turns else None,
        "avg_turns_to_failure": round(statistics.mean(turns_to_failure_list), 1) if turns_to_failure_list else None,
        "calls_with_turn_data": len(all_turns)
    }

    # === Clarification Statistics (v3.8.5: uses parse_clarifications) ===
    clarification_counts = []
    clarification_types = Counter()
    clarification_causes = Counter()
    clarification_total_details = 0
    calls_with_clarifications = 0

    for a in analyses:
        clarifications = parse_clarifications(a)
        if clarifications:
            calls_with_clarifications += 1
            clarification_counts.append(len(clarifications))

            for turn, ctype, cause, ctx in clarifications:
                clarification_total_details += 1
                if ctype:
                    clarification_types[ctype] += 1
                if cause:
                    clarification_causes[cause] += 1

    # v3.8.5: resolution_rate not tracked (resolved boolean removed)
    clarification_stats = {
        "calls_with_clarifications": calls_with_clarifications,
        "pct_calls_with_clarifications": safe_rate(calls_with_clarifications, n),
        "avg_clarifications_per_call": round(statistics.mean(clarification_counts), 2) if clarification_counts else 0,
        "by_type": {
            ctype: {"count": count, "rate": safe_rate(count, n)}
            for ctype, count in clarification_types.most_common()
        },
        "by_cause": {
            cause: {"count": count, "rate": safe_rate(count, clarification_total_details)}
            for cause, count in clarification_causes.most_common()
        },
        "resolution_rate": None  # v3.8.5: removed resolved tracking
    }

    # === Correction Statistics (v3.8.5: uses parse_corrections) ===
    correction_counts = []
    corrections_with_frustration = 0
    correction_severities = Counter()
    total_corrections = 0
    calls_with_corrections = 0

    for a in analyses:
        corrections = parse_corrections(a)
        if corrections:
            calls_with_corrections += 1
            correction_counts.append(len(corrections))

            for turn, severity, ctx, frustrated in corrections:
                total_corrections += 1
                if frustrated:
                    corrections_with_frustration += 1
                if severity:
                    correction_severities[severity] += 1

    correction_stats = {
        "calls_with_corrections": calls_with_corrections,
        "pct_calls_with_corrections": safe_rate(calls_with_corrections, n),
        "avg_corrections_per_call": round(statistics.mean(correction_counts), 2) if correction_counts else 0,
        "with_frustration_signal": corrections_with_frustration,
        "frustration_rate": safe_rate(corrections_with_frustration, total_corrections) if total_corrections > 0 else None,
        "by_severity": {
            severity: {"count": count, "rate": safe_rate(count, total_corrections)}
            for severity, count in correction_severities.most_common()
        }
    }

    # === Loop Statistics (v3.9.1: uses parse_loops with subject) ===
    calls_with_loops = 0
    all_loop_counts = []
    loop_types = Counter()
    loop_subjects = Counter()  # v3.9.1: subject aggregation
    loop_type_subjects = {}  # v3.9.1: subjects by loop type
    total_loops = 0
    loops_with_subject = 0
    total_turns_in_calls_with_loops = 0

    for a in analyses:
        loops = parse_loops(a)
        if loops:
            calls_with_loops += 1
            all_loop_counts.append(len(loops))
            total_loops += len(loops)

            # Track turns for loop density calculation
            turns = get_conversation_turns(a)
            if turns and isinstance(turns, (int, float)):
                total_turns_in_calls_with_loops += turns

            for turn_list, loop_type, subject, ctx in loops:
                if loop_type:
                    loop_types[loop_type] += 1

                    # v3.9.1: Track subject by loop type
                    if subject:
                        loops_with_subject += 1
                        loop_subjects[subject] += 1
                        if loop_type not in loop_type_subjects:
                            loop_type_subjects[loop_type] = Counter()
                        loop_type_subjects[loop_type][subject] += 1

    # Calculate loop density (loops per turn for calls with loops)
    loop_density = None
    if total_turns_in_calls_with_loops > 0:
        loop_density = round(total_loops / total_turns_in_calls_with_loops, 4)

    # v3.9.1: Build subject stats by loop type
    by_subject = {}
    for loop_type, subjects in loop_type_subjects.items():
        type_total = sum(subjects.values())
        by_subject[loop_type] = {
            subj: {"count": count, "rate": safe_rate(count, type_total)}
            for subj, count in subjects.most_common()
        }

    loop_stats = {
        "calls_with_loops": calls_with_loops,
        "pct_calls_with_loops": safe_rate(calls_with_loops, n),
        "avg_loops_per_call": round(statistics.mean(all_loop_counts), 2) if all_loop_counts else 0,
        "total_loops": total_loops,
        "loop_density": loop_density,
        "by_type": {
            loop_type: {"count": count, "rate": safe_rate(count, total_loops)}
            for loop_type, count in loop_types.most_common()
        },
        # v3.9.1: Subject statistics
        "loops_with_subject": loops_with_subject,
        "by_subject": by_subject,
        "top_subjects": [
            {"subject": subj, "count": count}
            for subj, count in loop_subjects.most_common(10)
        ]
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


def get_quality_scores(analysis: dict) -> tuple:
    """Get quality scores with v4.0/v3.x compatibility.

    Returns: (effectiveness, quality, effort) scores
    """
    # v4.0 uses shorter names
    effectiveness = analysis.get("effectiveness") or analysis.get("agent_effectiveness")
    quality = analysis.get("quality") or analysis.get("conversation_quality")
    effort = analysis.get("effort") or analysis.get("customer_effort")
    return (effectiveness, quality, effort)


def get_failure_info(analysis: dict) -> tuple:
    """Get failure info with v4.0/v3.x compatibility.

    Returns: (failure_type, failure_detail, recoverable, critical)
    """
    # v4.0 uses failure_type, v3.x uses failure_point
    failure_type = analysis.get("failure_type") or analysis.get("failure_point")
    failure_detail = analysis.get("failure_detail") or analysis.get("failure_description")
    recoverable = analysis.get("failure_recoverable") if "failure_recoverable" in analysis else analysis.get("was_recoverable")
    critical = analysis.get("failure_critical") if "failure_critical" in analysis else analysis.get("critical_failure")
    return (failure_type, failure_detail, recoverable, critical)


def is_success(analysis: dict) -> bool:
    """Check if call was successful with v5.0/v4.0/v3.x compatibility."""
    # v5.0: completed outcome
    if "call_outcome" in analysis:
        return analysis["call_outcome"] == "completed"

    # v4.0 uses disposition
    disposition = get_disposition(analysis)
    if disposition:
        return disposition in ("in_scope_success", "in_scope_partial")

    # Fallback to outcome for v3.x
    outcome = analysis.get("outcome")
    return outcome == "resolved"


def generate_report(analyses: list[dict]) -> dict:
    """Generate Section A: Deterministic Metrics report (v5.0 with v4.x/v3.x backwards compat)."""
    if not analyses:
        return {"error": "No analyses to process"}

    n = len(analyses)

    # Detect schema version distribution
    v5_count = sum(1 for a in analyses if a.get("schema_version", "").startswith("v5"))
    v4_count = sum(1 for a in analyses if a.get("schema_version", "").startswith("v4"))
    v3_count = sum(1 for a in analyses if a.get("schema_version", "").startswith("v3"))

    # Get date range from call_ids or timestamps
    timestamps = [a.get("timestamp") for a in analyses if a.get("timestamp")]
    date_range = {
        "earliest": min(timestamps) if timestamps else None,
        "latest": max(timestamps) if timestamps else None
    }

    # Outcome distribution (v3.x compatibility)
    outcomes = Counter(a.get("outcome", "unclear") for a in analyses if a.get("outcome"))

    # Resolution types (free text - show top 15)
    # v4.0 uses "resolution", v3.x uses "resolution_type"
    resolution_types = Counter()
    for a in analyses:
        res = a.get("resolution") or a.get("resolution_type")
        if res:
            resolution_types[res] += 1

    # Quality scores (v4.0/v3.x compatible)
    effectiveness = []
    quality = []
    effort = []
    for a in analyses:
        e, q, ef = get_quality_scores(a)
        if e is not None:
            effectiveness.append(e)
        if q is not None:
            quality.append(q)
        if ef is not None:
            effort.append(ef)

    # Failure analysis (v4.0/v3.x compatible)
    failures = [a for a in analyses if not is_success(a)]
    failure_types = Counter()
    recoverable_count = 0
    critical_count = 0

    for a in failures:
        ft, fd, rec, crit = get_failure_info(a)
        if ft and ft != "none":
            failure_types[ft] += 1
        if rec is True:
            recoverable_count += 1
        if crit is True:
            critical_count += 1

    # Also count critical for all analyses
    for a in analyses:
        _, _, _, crit = get_failure_info(a)
        if crit is True:
            critical_count += 1

    # Actionable flags (v5.0/v4.0/v3.x compatible)
    # v5.0: escalation_requested removed (covered by escalation_trigger=customer_requested)
    escalation_requested = sum(1 for a in analyses
                                if a.get("escalation_requested")
                                or a.get("escalation_trigger") == "customer_requested")
    repeat_callers = sum(1 for a in analyses if a.get("repeat_caller") or a.get("repeat_caller_signals"))
    multi_intent = sum(1 for a in analyses if a.get("secondary_intent") or a.get("additional_intents"))

    # Training opportunities (v3.x only - merged into coaching in v4.0)
    training_opps = Counter(
        a.get("training_opportunity")
        for a in analyses if a.get("training_opportunity")
    )

    # Policy gap breakdown (v4.0 uses policy_gap, v3.x uses policy_gap_detail)
    policy_gap_breakdown = extract_policy_gap_breakdown(analyses)

    # Call disposition breakdown (v4.0/v3.9+)
    disposition_breakdown = compute_disposition_breakdown(analyses)

    # Conversation quality metrics (v3.6+)
    conversation_quality_metrics = compute_conversation_quality_metrics(analyses)

    # v4.0: Intent statistics
    intent_stats = compute_intent_stats(analyses)

    # v4.0: Sentiment statistics
    sentiment_stats = compute_sentiment_stats(analyses)

    # v4.4: Handle time statistics
    handle_time_stats = compute_handle_time_stats(analyses)

    # Calculate key rates
    has_v5 = any("call_scope" in a for a in analyses)
    if has_v5:
        # v5.0: Use call_outcome directly
        completed_count = sum(1 for a in analyses if a.get("call_outcome") == "completed")
        escalated_count = sum(1 for a in analyses if a.get("call_outcome") == "escalated")
        abandoned_count = sum(1 for a in analyses if a.get("call_outcome") == "abandoned")
        # Containment = all calls NOT escalated to human
        containment_count = completed_count + abandoned_count
        resolved_count = completed_count
    else:
        disp_by = disposition_breakdown.get("by_disposition", {})
        if disp_by:
            success_count = disp_by.get("in_scope_success", {}).get("count", 0)
            partial_count = disp_by.get("in_scope_partial", {}).get("count", 0)
            resolved_count = success_count + partial_count
            escalated_count = disp_by.get("escalated", {}).get("count", 0)
        else:
            resolved_count = outcomes.get("resolved", 0)
            escalated_count = outcomes.get("escalated", 0)
        abandoned_count = outcomes.get("abandoned", 0)
        containment_count = resolved_count + abandoned_count

    # Validation warnings
    validation_warnings = validate_failure_consistency(analyses)

    report = {
        "metadata": {
            "report_generated": datetime.now().isoformat(),
            "schema_version": "v5.0",
            "total_calls_analyzed": n,
            "date_range": date_range,
            "source_versions": {
                "v5": v5_count,
                "v4": v4_count,
                "v3": v3_count,
                "other": n - v5_count - v4_count - v3_count
            }
        },

        "key_rates": {
            "success_rate": safe_rate(resolved_count, n),
            "containment_rate": safe_rate(containment_count, n),
            "escalation_rate": safe_rate(escalated_count, n),
            "failure_rate": safe_rate(n - resolved_count, n)
        },

        "quality_scores": {
            "effectiveness": safe_stats(effectiveness),
            "quality": safe_stats(quality),
            "effort": safe_stats(effort)
        },

        "failure_analysis": {
            "total_failures": len(failures),
            "failure_rate": safe_rate(len(failures), n),
            "by_failure_type": {
                k: {"count": v, "rate": safe_rate(v, len(failures))}
                for k, v in failure_types.most_common()
            },
            "recoverable_count": recoverable_count,
            "recoverable_rate": safe_rate(recoverable_count, len(failures)),
            "critical_failure_count": critical_count
        },

        "policy_gap_breakdown": policy_gap_breakdown,

        "disposition_breakdown": disposition_breakdown,

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

    # v3.x backwards compatibility: include outcome_distribution if present
    if outcomes:
        report["outcome_distribution"] = {
            k: {"count": v, "rate": safe_rate(v, n)}
            for k, v in outcomes.items()
        }

    # v4.0: Add intent and sentiment stats if available
    if intent_stats:
        report["intent_stats"] = intent_stats

    if sentiment_stats:
        report["sentiment_stats"] = sentiment_stats

    # v4.4: Handle time stats
    if handle_time_stats:
        report["handle_time"] = handle_time_stats

    # v4.5: Dashboard metrics
    call_funnel = compute_call_funnel(analyses)
    in_scope_outcomes = compute_in_scope_outcomes(analyses)
    action_performance = compute_action_performance(analyses)
    transfer_quality = compute_transfer_quality(analyses)

    if call_funnel or in_scope_outcomes or action_performance or transfer_quality:
        report["dashboard"] = {}
        if call_funnel:
            report["dashboard"]["call_funnel"] = call_funnel
        if in_scope_outcomes:
            report["dashboard"]["in_scope_outcomes"] = in_scope_outcomes
        if action_performance:
            report["dashboard"]["action_performance"] = action_performance
        if transfer_quality:
            report["dashboard"]["transfer_quality"] = transfer_quality

    return report


def print_summary(report: dict) -> None:
    """Print human-readable summary of Section A metrics."""
    print("\n" + "=" * 60)
    print("VACATIA AI VOICE AGENT - ANALYTICS REPORT (v5.0 - Section A)")
    print("=" * 60)

    m = report.get("metadata", {})
    print(f"\nCalls Analyzed: {m.get('total_calls_analyzed', 0)}")
    print(f"Generated: {m.get('report_generated', 'N/A')}")
    versions = m.get("source_versions", {})
    if versions:
        parts = []
        for v in ("v5", "v4", "v3"):
            c = versions.get(v, 0)
            if c > 0:
                parts.append(f"{v}={c}")
        if parts:
            print(f"Source Versions: {', '.join(parts)}")

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
    # v4.0 uses by_failure_type, v3.x uses by_failure_point
    by_type = f.get("by_failure_type", {}) or f.get("by_failure_point", {})
    for fp, data in by_type.items():
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

    # Disposition Breakdown
    print("\n" + "-" * 40)
    print("CALL DISPOSITION")
    print("-" * 40)
    db = report.get("disposition_breakdown", {})
    by_scope = db.get("by_scope", {})
    by_outcome = db.get("by_outcome", {})
    by_disp = db.get("by_disposition", {})

    if by_scope:
        # v5.0: scope × outcome
        print("  By Scope:")
        for scope, data in by_scope.items():
            print(f"    {scope}: {data['count']} ({data['rate']*100:.1f}%)" if data.get('rate') else f"    {scope}: {data['count']}")
        print("  By Outcome:")
        for outcome, data in by_outcome.items():
            print(f"    {outcome}: {data['count']} ({data['rate']*100:.1f}%)" if data.get('rate') else f"    {outcome}: {data['count']}")
        cross = db.get("cross_tab", {})
        if cross:
            print("  Cross-Tab (scope:outcome):")
            for combo, data in cross.items():
                print(f"    {combo}: {data['count']} ({data['rate']*100:.1f}%)" if data.get('rate') else f"    {combo}: {data['count']}")
    elif by_disp:
        # v4.x/v3.x: legacy disposition
        print("  By Disposition:")
        for disp, data in by_disp.items():
            print(f"    {disp}: {data['count']} ({data['rate']*100:.1f}%)" if data.get('rate') else f"    {disp}: {data['count']}")

    funnel = db.get("funnel_metrics", {})
    if funnel:
        print("\n  Funnel Metrics:")
        if funnel.get("containment_rate") is not None:
            print(f"    Containment Rate: {funnel['containment_rate']*100:.1f}% ({funnel.get('in_scope_total', 0)} in-scope calls)")
        if funnel.get("escalation_rate") is not None:
            print(f"    Escalation Rate: {funnel['escalation_rate']*100:.1f}%")
        if funnel.get("out_of_scope_recovery_rate") is not None:
            print(f"    Out-of-Scope Recovery: {funnel['out_of_scope_recovery_rate']*100:.1f}%")

    # Actionable Flags
    print("\n" + "-" * 40)
    print("ACTIONABLE FLAGS")
    print("-" * 40)
    a = report.get("actionable_flags", {})
    for flag, data in a.items():
        label = flag.replace("_", " ").title()
        rate = data.get('rate')
        print(f"  {label}: {data['count']} ({rate*100:.1f}%)" if rate else f"  {label}: {data['count']}")

    # Conversation Quality (v3.8.5)
    print("\n" + "-" * 40)
    print("CONVERSATION QUALITY (v3.8.5)")
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
            # v3.7: By cause
            by_cause = cs.get("by_cause", {})
            if by_cause:
                print("  By Cause (v3.7):")
                for cause, data in by_cause.items():
                    print(f"    {cause}: {data['count']} ({data.get('rate', 0)*100:.1f}%)")

        # Correction stats
        cors = cq.get("correction_stats", {})
        if cors.get("calls_with_corrections"):
            print(f"  Calls with Corrections: {cors['calls_with_corrections']} ({cors.get('pct_calls_with_corrections', 0)*100:.1f}%)")
            if cors.get("frustration_rate"):
                print(f"  Corrections with Frustration: {cors.get('with_frustration_signal', 0)} ({cors['frustration_rate']*100:.1f}%)")
            # v3.7: By severity
            by_severity = cors.get("by_severity", {})
            if by_severity:
                print("  By Severity (v3.7):")
                for severity, data in by_severity.items():
                    print(f"    {severity}: {data['count']} ({data.get('rate', 0)*100:.1f}%)")

        # Loop stats (v3.9.1: agent_loops with typed detection + subject)
        ls = cq.get("loop_stats", {})
        if ls.get("calls_with_loops"):
            print(f"  Calls with Loops: {ls['calls_with_loops']} ({ls.get('pct_calls_with_loops', 0)*100:.1f}%)")
            print(f"  Total Loops: {ls.get('total_loops', 0)} | Avg per Call: {ls.get('avg_loops_per_call', 0)}")
            if ls.get("loop_density"):
                print(f"  Loop Density: {ls['loop_density']} loops/turn")
            # v3.8: By type distribution
            by_type = ls.get("by_type", {})
            if by_type:
                print("  By Type:")
                for loop_type, data in by_type.items():
                    print(f"    {loop_type}: {data['count']} ({data.get('rate', 0)*100:.1f}%)")
            # v3.9.1: Subject stats
            loops_with_subject = ls.get("loops_with_subject", 0)
            if loops_with_subject > 0:
                print(f"  Loops with Subject (v3.9.1): {loops_with_subject}")
                top_subjects = ls.get("top_subjects", [])
                if top_subjects:
                    print("  Top Subjects:")
                    for item in top_subjects[:5]:
                        print(f"    {item.get('subject', 'N/A')}: {item.get('count', 0)}")
    else:
        print("  No v3.8.5 conversation quality data")

    # v4.0: Intent Statistics
    intent_stats = report.get("intent_stats", {})
    if intent_stats:
        print("\n" + "-" * 40)
        print("INTENT STATISTICS (v4.0)")
        print("-" * 40)
        print(f"  Calls with intent: {intent_stats.get('total_with_intent', 0)} ({intent_stats.get('coverage_rate', 0)*100:.1f}%)")
        print(f"  Calls with context: {intent_stats.get('total_with_context', 0)}")
        print(f"  Calls with secondary intent: {intent_stats.get('total_with_secondary', 0)}")
        top_intents = intent_stats.get("top_intents", [])
        if top_intents:
            print("  Top Intents:")
            for item in top_intents[:5]:
                print(f"    {item['intent']}: {item['count']}")

    # v4.0: Sentiment Statistics
    sentiment_stats = report.get("sentiment_stats", {})
    if sentiment_stats:
        print("\n" + "-" * 40)
        print("SENTIMENT STATISTICS (v4.0)")
        print("-" * 40)
        print(f"  Calls with sentiment: {sentiment_stats.get('total_with_sentiment', 0)}")
        improvement = sentiment_stats.get('improvement_rate')
        degradation = sentiment_stats.get('degradation_rate')
        if improvement is not None:
            print(f"  Improved: {improvement*100:.1f}%")
        if degradation is not None:
            print(f"  Degraded: {degradation*100:.1f}%")
        start_dist = sentiment_stats.get("start_distribution", {})
        if start_dist:
            print("  Start Sentiment:")
            for sent, data in start_dist.items():
                print(f"    {sent}: {data['count']} ({data['rate']*100:.1f}%)")
        end_dist = sentiment_stats.get("end_distribution", {})
        if end_dist:
            print("  End Sentiment:")
            for sent, data in end_dist.items():
                print(f"    {sent}: {data['count']} ({data['rate']*100:.1f}%)")

    # Handle Time Statistics
    handle_time = report.get("handle_time", {})
    if handle_time:
        print("\n" + "-" * 40)
        print("HANDLE TIME / AHT")
        print("-" * 40)
        overall = handle_time.get("overall", {})
        if overall.get("mean"):
            print(f"  Overall: {overall['mean']:.1f}s mean, {overall['median']:.1f}s median (n={overall['n']}, range {overall['min']:.0f}-{overall['max']:.0f}s)")
        by_scope = handle_time.get("by_scope", {})
        if by_scope:
            print("  By Scope:")
            for scope, stats in by_scope.items():
                if stats.get("mean"):
                    print(f"    {scope}: {stats['mean']:.1f}s mean, {stats['median']:.1f}s median (n={stats['n']})")
        by_outcome = handle_time.get("by_outcome", {})
        if by_outcome:
            print("  By Outcome:")
            for outcome, stats in by_outcome.items():
                if stats.get("mean"):
                    print(f"    {outcome}: {stats['mean']:.1f}s mean, {stats['median']:.1f}s median (n={stats['n']})")
        by_disp = handle_time.get("by_disposition", {})
        if by_disp:
            print("  By Disposition:")
            for disp, stats in by_disp.items():
                if stats.get("mean"):
                    print(f"    {disp}: {stats['mean']:.1f}s mean, {stats['median']:.1f}s median (n={stats['n']})")

    # Dashboard Metrics
    dashboard = report.get("dashboard", {})
    if dashboard:
        print("\n" + "-" * 40)
        print("DASHBOARD (v5.0)")
        print("-" * 40)

        # Call Funnel
        cf = dashboard.get("call_funnel", {})
        if cf:
            print(f"  Call Funnel (N={cf.get('total', 0)}):")
            nr = cf.get("no_request", {})
            rm = cf.get("request_made", {})
            print(f"    No Request: {nr.get('count', 0)} ({(nr.get('rate') or 0)*100:.1f}%)")
            stages = nr.get("by_stage", {})
            for st, data in stages.items():
                print(f"      {st}: {data['count']}")
            print(f"    Request Made: {rm.get('count', 0)} ({(rm.get('rate') or 0)*100:.1f}%)")
            isc = rm.get("in_scope", {})
            oos = rm.get("out_of_scope", {})
            mx = rm.get("mixed", {})
            parts = [f"In-Scope: {isc.get('count', 0)}", f"Out-of-Scope: {oos.get('count', 0)}"]
            if mx.get("count", 0) > 0:
                parts.append(f"Mixed: {mx['count']}")
            print(f"      {' | '.join(parts)}")

        # In-Scope Outcomes
        iso = dashboard.get("in_scope_outcomes", {})
        if iso:
            print(f"\n  In-Scope Outcomes (N={iso.get('in_scope_total', 0)}):")
            comp = iso.get("completed", {})
            print(f"    Completed: {comp.get('count', 0)} ({(comp.get('rate') or 0)*100:.1f}%)")
            conf = comp.get("confirmed", {})
            unconf = comp.get("unconfirmed", {})
            print(f"      Confirmed: {conf.get('count', 0)} | Unconfirmed: {unconf.get('count', 0)}")
            esc = iso.get("escalated", {})
            print(f"    Escalated: {esc.get('count', 0)} ({(esc.get('rate') or 0)*100:.1f}%)")
            triggers = esc.get("by_trigger", {})
            if triggers:
                trigger_parts = [f"{k}: {v['count']}" for k, v in triggers.items()]
                print(f"      {' | '.join(trigger_parts)}")
            ab = iso.get("abandoned", {})
            print(f"    Abandoned: {ab.get('count', 0)} ({(ab.get('rate') or 0)*100:.1f}%)")
            stages = ab.get("by_stage", {})
            if stages:
                stage_parts = [f"{k}: {v['count']}" for k, v in stages.items()]
                print(f"      {' | '.join(stage_parts)}")

        # Action Performance
        ap = dashboard.get("action_performance", {})
        if ap:
            overall = ap.get("overall", {})
            print(f"\n  Action Performance ({overall.get('total_actions', 0)} actions, {overall.get('action_types_seen', 0)} types):")
            for action_type, data in ap.get("by_action", {}).items():
                sr = data.get("success_rate")
                sr_str = f"{sr*100:.0f}%" if sr is not None else "N/A"
                print(f"    {action_type}: {data['attempted']} attempted, {sr_str} success")

        # Transfer Quality
        tq = dashboard.get("transfer_quality", {})
        if tq:
            print(f"\n  Transfer Quality ({tq.get('total_transfers', 0)} transfers):")
            for dest, data in tq.get("by_destination", {}).items():
                print(f"    {dest}: {data['count']} ({(data.get('rate') or 0)*100:.0f}%)")
            qd = tq.get("queue_detected", {})
            print(f"    Queue detected: {qd.get('count', 0)} ({(qd.get('rate') or 0)*100:.0f}%)")

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
    parser = argparse.ArgumentParser(description="Compute v4.0 Section A: Deterministic Metrics (intent + sentiment)")
    parser.add_argument("-i", "--input-dir", type=Path,
                        default=Path(__file__).parent.parent / "analyses",
                        help="Directory containing analysis JSON files")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "reports",
                        help="Output directory for report")
    parser.add_argument("-s", "--sampled-dir", type=Path,
                        default=Path(__file__).parent.parent / "sampled",
                        help="Directory containing manifest.csv for scope filtering")
    parser.add_argument("--no-scope-filter", action="store_true",
                        help="Disable manifest-based scope filtering (include all analyses)")
    parser.add_argument("--json-only", action="store_true",
                        help="Output only JSON")
    parser.add_argument("--schema-version", type=str, choices=["v2", "v3", "v4"], default="v4",
                        help="Schema version to process (default: v4, accepts v3.x/v4.x)")

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

    print(f"Loading from: {args.input_dir}", file=sys.stderr)
    analyses = load_analyses(args.input_dir, args.schema_version, manifest_ids)
    print(f"Loaded {len(analyses)} {args.schema_version} analyses", file=sys.stderr)

    if not analyses:
        print(f"Error: No {args.schema_version} analysis files found", file=sys.stderr)
        return 1

    report = generate_report(analyses)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save Section A metrics (v4.0 output format)
    report_path = args.output_dir / f"metrics_v4_{timestamp}.json"
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
