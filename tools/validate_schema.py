#!/usr/bin/env python3
"""
v5.0 Schema Validator for Voice Agent Analytics

Validates analysis JSONs against v5.0 field constraints:
- Field presence (call_scope, call_outcome, conditional qualifiers)
- Type validation (boolean, enum, array, null)
- Conditional consistency (cross-field rules)
- Enum value validation (known values only)
- Funnel MECE invariants (optional, with --check-funnel)

Also supports v4.5 analyses for backward compatibility.

Usage:
    python3 tools/validate_schema.py runs/run_XXXX/analyses/
    python3 tools/validate_schema.py runs/run_XXXX/analyses/ --check-funnel
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# === v5.0 Enums ===

VALID_SCOPES = {"in_scope", "out_of_scope", "mixed", "no_request"}
VALID_OUTCOMES = {"completed", "escalated", "abandoned"}
VALID_ESCALATION_TRIGGERS = {"customer_requested", "scope_limit", "task_failure", "policy_routing"}
VALID_ABANDON_STAGES = {"pre_greeting", "pre_intent", "mid_task", "post_delivery"}

VALID_ACTION_TYPES = {
    "account_lookup", "verification", "send_payment_link", "send_portal_link",
    "send_autopay_link", "send_rental_link", "send_clubhouse_link",
    "send_rci_link", "transfer", "other"
}
VALID_ACTION_OUTCOMES = {"success", "failed", "retry", "unknown"}
VALID_TRANSFER_DESTINATIONS = {"concierge", "specific_department", "ivr", "unknown"}

# === v4.5 Enums (backward compat) ===

VALID_DISPOSITIONS_V4 = {
    "pre_intent", "out_of_scope_handled", "out_of_scope_failed",
    "in_scope_success", "in_scope_partial", "in_scope_failed", "escalated"
}


def validate_v5_analysis(data: dict, filename: str) -> list[str]:
    """Validate a v5.0 analysis JSON."""
    errors = []
    scope = data.get("call_scope")
    outcome = data.get("call_outcome")

    # --- Core fields ---
    for required in ["call_id", "call_scope", "call_outcome", "summary", "intent"]:
        if required not in data:
            errors.append(f"missing field: {required}")

    # call_scope enum
    if scope is not None and scope not in VALID_SCOPES:
        errors.append(f"call_scope invalid: '{scope}' (expected {VALID_SCOPES})")

    # call_outcome enum
    if outcome is not None and outcome not in VALID_OUTCOMES:
        errors.append(f"call_outcome invalid: '{outcome}' (expected {VALID_OUTCOMES})")

    # --- Conditional qualifiers ---

    # escalation_trigger: required when outcome=escalated, null otherwise
    et = data.get("escalation_trigger")
    if outcome == "escalated":
        if et is None:
            errors.append("escalation_trigger must be non-null when call_outcome=escalated")
        elif et not in VALID_ESCALATION_TRIGGERS:
            errors.append(f"escalation_trigger invalid: '{et}' (expected {VALID_ESCALATION_TRIGGERS})")
    else:
        if et is not None:
            errors.append(f"escalation_trigger must be null when call_outcome={outcome}, got '{et}'")

    # abandon_stage: required when outcome=abandoned, null otherwise
    ast = data.get("abandon_stage")
    if outcome == "abandoned":
        if ast is None:
            errors.append("abandon_stage must be non-null when call_outcome=abandoned")
        elif ast not in VALID_ABANDON_STAGES:
            errors.append(f"abandon_stage invalid: '{ast}' (expected {VALID_ABANDON_STAGES})")
    else:
        if ast is not None:
            errors.append(f"abandon_stage must be null when call_outcome={outcome}, got '{ast}'")

    # resolution_confirmed: required when outcome=completed, null otherwise
    rc = data.get("resolution_confirmed")
    if outcome == "completed":
        if rc is None:
            errors.append("resolution_confirmed must be bool when call_outcome=completed")
        elif not isinstance(rc, bool):
            errors.append(f"resolution_confirmed must be bool, got {type(rc).__name__}")
    else:
        if rc is not None:
            errors.append(f"resolution_confirmed must be null when call_outcome={outcome}, got {rc}")

    # --- Shared v4.5+ fields ---
    errors.extend(_validate_shared_fields(data))

    # --- Invalid combinations ---
    if scope == "no_request" and outcome == "completed":
        errors.append("invalid combination: no_request + completed")

    # --- Deprecated fields should not be present ---
    for deprecated in ("disposition", "escalation_initiator", "pre_intent_subtype",
                        "abandoned_path_viable", "escalation_requested",
                        "failure_recoverable", "failure_critical"):
        if deprecated in data:
            errors.append(f"deprecated field present: {deprecated}")

    return errors


def validate_v4_analysis(data: dict, filename: str) -> list[str]:
    """Validate a v4.5 analysis JSON (backward compat)."""
    errors = []
    disposition = data.get("disposition")

    # --- Field presence ---
    for required in ["call_id", "disposition", "summary", "intent"]:
        if required not in data:
            errors.append(f"missing field: {required}")

    if "resolution_confirmed" not in data:
        errors.append("missing field: resolution_confirmed")
    if "actions" not in data:
        errors.append("missing field: actions")
    if "transfer_queue_detected" not in data:
        errors.append("missing field: transfer_queue_detected")

    # --- Disposition enum ---
    if disposition is not None and disposition not in VALID_DISPOSITIONS_V4:
        errors.append(f"disposition invalid: '{disposition}'")

    # --- Conditional consistency ---
    rc = data.get("resolution_confirmed")
    if disposition in ("in_scope_success", "in_scope_partial"):
        if rc is None:
            errors.append(f"resolution_confirmed must be bool when disposition={disposition}")
    elif disposition in VALID_DISPOSITIONS_V4:
        if rc is not None:
            errors.append(f"resolution_confirmed must be null when disposition={disposition}, got {rc}")

    pis = data.get("pre_intent_subtype")
    if disposition == "pre_intent":
        if pis is None:
            errors.append("pre_intent_subtype must be non-null when disposition=pre_intent")
    elif pis is not None:
        errors.append(f"pre_intent_subtype must be null when disposition={disposition}")

    ei = data.get("escalation_initiator")
    if disposition == "escalated":
        if ei is None:
            errors.append("escalation_initiator must be non-null when disposition=escalated")
    elif ei is not None:
        errors.append(f"escalation_initiator must be null when disposition={disposition}")

    # --- Shared fields ---
    errors.extend(_validate_shared_fields(data))

    return errors


def _validate_shared_fields(data: dict) -> list[str]:
    """Validate fields common to v4.5 and v5.0."""
    errors = []

    # actions: array of objects
    actions = data.get("actions")
    if actions is not None:
        if not isinstance(actions, list):
            errors.append(f"actions must be a list, got {type(actions).__name__}")
        else:
            for i, action in enumerate(actions):
                if not isinstance(action, dict):
                    errors.append(f"actions[{i}] must be an object")
                    continue
                if "action" not in action:
                    errors.append(f"actions[{i}] missing 'action' field")
                elif action["action"] not in VALID_ACTION_TYPES:
                    errors.append(f"actions[{i}].action invalid: '{action['action']}'")
                if "outcome" not in action:
                    errors.append(f"actions[{i}] missing 'outcome' field")
                elif action["outcome"] not in VALID_ACTION_OUTCOMES:
                    errors.append(f"actions[{i}].outcome invalid: '{action['outcome']}'")
                if "detail" not in action:
                    errors.append(f"actions[{i}] missing 'detail' field")

    # transfer_destination
    td = data.get("transfer_destination")
    if td is not None:
        if not isinstance(td, str):
            errors.append(f"transfer_destination must be string or null, got {type(td).__name__}")
        elif td not in VALID_TRANSFER_DESTINATIONS:
            errors.append(f"transfer_destination unexpected: '{td}'")

    # transfer_queue_detected
    tqd = data.get("transfer_queue_detected")
    if tqd is not None and not isinstance(tqd, bool):
        errors.append(f"transfer_queue_detected must be bool, got {type(tqd).__name__}")

    return errors


def validate_analysis(data: dict, filename: str) -> list[str]:
    """Route to v5.0 or v4.x validator based on schema version."""
    version = data.get("schema_version", "")
    if version.startswith("v5"):
        return validate_v5_analysis(data, filename)
    else:
        return validate_v4_analysis(data, filename)


def validate_funnel(analyses: list[dict]) -> list[str]:
    """Check MECE funnel invariants across all analyses."""
    errors = []
    n = len(analyses)

    has_v5 = any(a.get("schema_version", "").startswith("v5") for a in analyses)

    if has_v5:
        # v5.0: scope-based funnel
        no_request = sum(1 for a in analyses if a.get("call_scope") == "no_request")
        request_made = sum(1 for a in analyses if a.get("call_scope") != "no_request")

        if no_request + request_made != n:
            errors.append(f"MECE: no_request({no_request}) + request_made({request_made}) != total({n})")

        # Outcome split for request_made calls
        completed = sum(1 for a in analyses if a.get("call_scope") != "no_request" and a.get("call_outcome") == "completed")
        escalated = sum(1 for a in analyses if a.get("call_scope") != "no_request" and a.get("call_outcome") == "escalated")
        abandoned_rm = sum(1 for a in analyses if a.get("call_scope") != "no_request" and a.get("call_outcome") == "abandoned")

        if completed + escalated + abandoned_rm != request_made:
            errors.append(
                f"MECE: completed({completed}) + escalated({escalated}) + "
                f"abandoned({abandoned_rm}) != request_made({request_made})"
            )

        # no_request should all be abandoned
        nr_non_abandoned = sum(1 for a in analyses
                                if a.get("call_scope") == "no_request"
                                and a.get("call_outcome") != "abandoned")
        if nr_non_abandoned > 0:
            nr_escalated = sum(1 for a in analyses
                                if a.get("call_scope") == "no_request"
                                and a.get("call_outcome") == "escalated")
            if nr_non_abandoned - nr_escalated > 0:
                errors.append(f"MECE: {nr_non_abandoned - nr_escalated} no_request calls with outcome != abandoned/escalated")
    else:
        # v4.x: disposition-based funnel
        pre_intent = sum(1 for a in analyses if a.get("disposition") == "pre_intent")
        intent_captured = sum(1 for a in analyses if a.get("disposition") != "pre_intent")

        if pre_intent + intent_captured != n:
            errors.append(f"MECE: pre_intent({pre_intent}) + intent_captured({intent_captured}) != total({n})")

        out_of_scope = sum(1 for a in analyses if a.get("disposition") in {"out_of_scope_handled", "out_of_scope_failed"})
        in_scope = sum(1 for a in analyses if a.get("disposition") in {"in_scope_success", "in_scope_partial", "in_scope_failed", "escalated"})

        if out_of_scope + in_scope != intent_captured:
            errors.append(f"MECE: oos({out_of_scope}) + is({in_scope}) != intent_captured({intent_captured})")

    return errors


def print_distribution(analyses: list[dict]) -> None:
    """Print field distribution summary for human review."""
    n = len(analyses)
    print(f"\n{'='*60}")
    print(f"DISTRIBUTION SUMMARY ({n} analyses)")
    print(f"{'='*60}")

    has_v5 = any(a.get("schema_version", "").startswith("v5") for a in analyses)

    if has_v5:
        # v5.0: scope × outcome
        scopes = Counter(a.get("call_scope") for a in analyses)
        print("\ncall_scope:")
        for s, c in scopes.most_common():
            print(f"  {s}: {c} ({c/n*100:.0f}%)")

        outcomes = Counter(a.get("call_outcome") for a in analyses)
        print("\ncall_outcome:")
        for o, c in outcomes.most_common():
            print(f"  {o}: {c} ({c/n*100:.0f}%)")

        # Cross-tab
        cross = Counter(f"{a.get('call_scope')}:{a.get('call_outcome')}" for a in analyses)
        print("\nscope:outcome cross-tab:")
        for combo, c in cross.most_common():
            print(f"  {combo}: {c} ({c/n*100:.0f}%)")

        # escalation_trigger
        et_vals = Counter(a.get("escalation_trigger") for a in analyses if a.get("escalation_trigger"))
        if et_vals:
            print("\nescalation_trigger:")
            for v, c in et_vals.most_common():
                print(f"  {v}: {c}")

        # abandon_stage
        as_vals = Counter(a.get("abandon_stage") for a in analyses if a.get("abandon_stage"))
        if as_vals:
            print("\nabandon_stage:")
            for v, c in as_vals.most_common():
                print(f"  {v}: {c}")
    else:
        # v4.x: disposition
        dispositions = Counter(a.get("disposition") for a in analyses)
        print("\ndisposition:")
        for d, c in dispositions.most_common():
            print(f"  {d}: {c} ({c/n*100:.0f}%)")

    # resolution_confirmed (shared)
    rc_vals = Counter()
    for a in analyses:
        rc = a.get("resolution_confirmed")
        if rc is True:
            rc_vals["true"] += 1
        elif rc is False:
            rc_vals["false"] += 1
        else:
            rc_vals["null"] += 1
    print("\nresolution_confirmed:")
    for v, c in rc_vals.most_common():
        print(f"  {v}: {c}")

    # actions (shared)
    action_counts = [len(a.get("actions", [])) for a in analyses]
    with_actions = sum(1 for c in action_counts if c > 0)
    print(f"\nactions: {with_actions}/{n} calls have actions")
    if action_counts:
        print(f"  avg actions/call: {sum(action_counts)/n:.1f}")
    action_types = Counter()
    action_outcomes = Counter()
    for a in analyses:
        for act in a.get("actions", []):
            action_types[act.get("action")] += 1
            action_outcomes[act.get("outcome")] += 1
    if action_types:
        print("  action types:")
        for t, c in action_types.most_common():
            print(f"    {t}: {c}")
    if action_outcomes:
        print("  outcomes:")
        for o, c in action_outcomes.most_common():
            print(f"    {o}: {c}")

    # transfer_destination (shared)
    td_vals = Counter(a.get("transfer_destination") for a in analyses if a.get("transfer_destination"))
    if td_vals:
        print("\ntransfer_destination:")
        for v, c in td_vals.most_common():
            print(f"  {v}: {c}")

    tqd_true = sum(1 for a in analyses if a.get("transfer_queue_detected") is True)
    print(f"\ntransfer_queue_detected: {tqd_true}/{n} true")


def main():
    parser = argparse.ArgumentParser(description="Validate v5.0/v4.5 schema fields in analysis JSONs")
    parser.add_argument("analyses_dir", type=Path, help="Path to analyses directory")
    parser.add_argument("--check-funnel", action="store_true", help="Also check MECE funnel invariants")
    args = parser.parse_args()

    if not args.analyses_dir.exists():
        print(f"Error: Directory not found: {args.analyses_dir}", file=sys.stderr)
        return 1

    # Load all analysis JSONs
    analyses = []
    for f in sorted(args.analyses_dir.glob("*.json")):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                if isinstance(data, dict) and "call_id" in data:
                    analyses.append((f.name, data))
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {f}: {e}", file=sys.stderr)

    if not analyses:
        print("Error: No valid analysis JSONs found", file=sys.stderr)
        return 1

    # Detect schema version
    versions = Counter(d.get("schema_version", "unknown") for _, d in analyses)
    print(f"Validating {len(analyses)} analyses from {args.analyses_dir}")
    print(f"Schema versions: {dict(versions)}")

    # Validate each analysis
    total_errors = 0
    passed = 0
    failed = 0

    for filename, data in analyses:
        errors = validate_analysis(data, filename)
        if errors:
            failed += 1
            total_errors += len(errors)
            print(f"\n  FAIL  {filename} ({len(errors)} errors):")
            for e in errors:
                print(f"        - {e}")
        else:
            passed += 1
            print(f"  PASS  {filename}")

    # Funnel invariants
    funnel_errors = []
    if args.check_funnel:
        all_data = [data for _, data in analyses]
        funnel_errors = validate_funnel(all_data)
        if funnel_errors:
            print(f"\n  FUNNEL ERRORS:")
            for e in funnel_errors:
                print(f"    - {e}")
        else:
            print(f"\n  FUNNEL: All MECE invariants hold")

    # Distribution summary
    all_data = [data for _, data in analyses]
    print_distribution(all_data)

    # Final summary
    print(f"\n{'='*60}")
    print(f"RESULT: {passed} passed, {failed} failed, {total_errors} total errors")
    if funnel_errors:
        print(f"FUNNEL: {len(funnel_errors)} invariant violations")
    print(f"{'='*60}")

    return 1 if (failed > 0 or funnel_errors) else 0


if __name__ == "__main__":
    sys.exit(main())
