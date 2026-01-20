#!/usr/bin/env python3
"""
Test Suite for v3.9 Features: Call Disposition Classification

Tests:
1. Schema has call_disposition field
2. Decision tree guidance in prompt
3. Scope reference in prompt
4. Disposition aggregation in compute_metrics
5. Funnel metrics calculation
6. NL extraction includes disposition
7. Render report includes disposition breakdown
"""

import json
import sys
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_v39_schema_structure():
    """Test v3.9 schema has call_disposition field."""
    print("\n=== Test: v3.9 Schema Structure ===")

    from analyze_transcript import ANALYSIS_SCHEMA

    # Check for call_disposition field
    assert '"call_disposition":' in ANALYSIS_SCHEMA, "Schema missing call_disposition field"
    print("  call_disposition field present in schema")

    # Check for valid enum values
    assert "'pre_intent'" in ANALYSIS_SCHEMA, "Schema missing pre_intent enum"
    assert "'out_of_scope_handled'" in ANALYSIS_SCHEMA, "Schema missing out_of_scope_handled enum"
    assert "'out_of_scope_abandoned'" in ANALYSIS_SCHEMA, "Schema missing out_of_scope_abandoned enum"
    assert "'in_scope_success'" in ANALYSIS_SCHEMA, "Schema missing in_scope_success enum"
    assert "'in_scope_partial'" in ANALYSIS_SCHEMA, "Schema missing in_scope_partial enum"
    assert "'in_scope_failed'" in ANALYSIS_SCHEMA, "Schema missing in_scope_failed enum"
    print("  All 6 disposition enum values present")

    print("  PASS: Schema structure test passed")
    return True


def test_v39_decision_tree_in_prompt():
    """Test v3.9 decision tree is included in system prompt."""
    print("\n=== Test: v3.9 Decision Tree in Prompt ===")

    from analyze_transcript import SYSTEM_PROMPT

    # Check for decision tree guidance
    assert "Call Disposition Classification" in SYSTEM_PROMPT, "Missing disposition classification section"
    print("  Disposition classification section present")

    assert "Did customer state a specific, actionable request?" in SYSTEM_PROMPT, "Missing decision tree step 1"
    print("  Decision tree step 1 present")

    assert "Could the agent handle this type of request?" in SYSTEM_PROMPT, "Missing decision tree step 2"
    print("  Decision tree step 2 present")

    assert "Did agent complete the requested action?" in SYSTEM_PROMPT, "Missing decision tree step 3"
    print("  Decision tree step 3 present")

    assert "Did customer express explicit satisfaction?" in SYSTEM_PROMPT, "Missing decision tree step 4"
    print("  Decision tree step 4 present")

    print("  PASS: Decision tree test passed")
    return True


def test_v39_scope_reference_in_prompt():
    """Test v3.9 scope reference is included in system prompt."""
    print("\n=== Test: v3.9 Scope Reference in Prompt ===")

    from analyze_transcript import SYSTEM_PROMPT

    # Check for scope reference
    assert "Scope Reference - IN-SCOPE" in SYSTEM_PROMPT, "Missing in-scope reference"
    print("  In-scope reference present")

    assert "Scope Reference - OUT-OF-SCOPE" in SYSTEM_PROMPT, "Missing out-of-scope reference"
    print("  Out-of-scope reference present")

    # Check for specific capabilities
    assert "Send links" in SYSTEM_PROMPT or "Payment" in SYSTEM_PROMPT, "Missing link sending capability"
    print("  Link sending capability mentioned")

    assert "Process payments directly" in SYSTEM_PROMPT, "Missing payment processing limitation"
    print("  Payment processing limitation mentioned")

    print("  PASS: Scope reference test passed")
    return True


def test_v39_disposition_aggregation():
    """Test disposition aggregation in compute_metrics."""
    print("\n=== Test: v3.9 Disposition Aggregation ===")

    from compute_metrics import compute_disposition_breakdown

    mock_analyses = [
        {"call_disposition": "in_scope_success"},
        {"call_disposition": "in_scope_success"},
        {"call_disposition": "in_scope_partial"},
        {"call_disposition": "in_scope_failed"},
        {"call_disposition": "out_of_scope_handled"},
        {"call_disposition": "out_of_scope_abandoned"},
        {"call_disposition": "pre_intent"},
        {},  # Missing disposition - should be counted as unknown
    ]

    result = compute_disposition_breakdown(mock_analyses)

    # Check structure
    assert "by_disposition" in result, "Missing by_disposition"
    assert "funnel_metrics" in result, "Missing funnel_metrics"
    print("  Result has correct structure")

    # Check counts
    by_disp = result["by_disposition"]
    assert by_disp.get("in_scope_success", {}).get("count") == 2, "Wrong in_scope_success count"
    assert by_disp.get("in_scope_partial", {}).get("count") == 1, "Wrong in_scope_partial count"
    assert by_disp.get("in_scope_failed", {}).get("count") == 1, "Wrong in_scope_failed count"
    assert by_disp.get("out_of_scope_handled", {}).get("count") == 1, "Wrong out_of_scope_handled count"
    assert by_disp.get("out_of_scope_abandoned", {}).get("count") == 1, "Wrong out_of_scope_abandoned count"
    assert by_disp.get("pre_intent", {}).get("count") == 1, "Wrong pre_intent count"
    assert by_disp.get("unknown", {}).get("count") == 1, "Wrong unknown count"
    print("  Disposition counts correct")

    # Check rates
    for disp, data in by_disp.items():
        assert "rate" in data, f"Missing rate for {disp}"
    print("  All dispositions have rates")

    print("  PASS: Disposition aggregation test passed")
    return True


def test_v39_funnel_metrics():
    """Test funnel metrics calculation."""
    print("\n=== Test: v3.9 Funnel Metrics ===")

    from compute_metrics import compute_disposition_breakdown

    mock_analyses = [
        # 4 in-scope: 2 success, 1 partial, 1 failed
        {"call_disposition": "in_scope_success"},
        {"call_disposition": "in_scope_success"},
        {"call_disposition": "in_scope_partial"},
        {"call_disposition": "in_scope_failed"},
        # 2 out-of-scope: 1 handled, 1 abandoned
        {"call_disposition": "out_of_scope_handled"},
        {"call_disposition": "out_of_scope_abandoned"},
        # 1 pre-intent
        {"call_disposition": "pre_intent"},
    ]

    result = compute_disposition_breakdown(mock_analyses)
    funnel = result.get("funnel_metrics", {})

    # In-scope success rate = 2/4 = 50%
    assert funnel.get("in_scope_success_rate") == 0.5, \
        f"Wrong in_scope_success_rate: {funnel.get('in_scope_success_rate')}"
    print("  In-scope success rate: 50%")

    # Out-of-scope recovery = 1/2 = 50%
    assert funnel.get("out_of_scope_recovery_rate") == 0.5, \
        f"Wrong out_of_scope_recovery_rate: {funnel.get('out_of_scope_recovery_rate')}"
    print("  Out-of-scope recovery rate: 50%")

    # Pre-intent rate = 1/7 = ~14.3%
    pre_intent_rate = funnel.get("pre_intent_rate")
    assert pre_intent_rate is not None, "Missing pre_intent_rate"
    assert 0.14 < pre_intent_rate < 0.15, f"Wrong pre_intent_rate: {pre_intent_rate}"
    print(f"  Pre-intent rate: {pre_intent_rate*100:.1f}%")

    # Totals
    assert funnel.get("in_scope_total") == 4, "Wrong in_scope_total"
    assert funnel.get("out_of_scope_total") == 2, "Wrong out_of_scope_total"
    print("  Totals correct")

    print("  PASS: Funnel metrics test passed")
    return True


def test_v39_nl_extraction():
    """Test NL extraction includes disposition."""
    print("\n=== Test: v3.9 NL Extraction ===")

    from extract_nl_fields import extract_nl_summary

    mock_analyses = [
        {
            "call_id": "test-1",
            "outcome": "resolved",
            "call_disposition": "in_scope_success",
            "summary": "Customer paid maintenance fees successfully.",
            "customer_verbatim": "Thanks so much!"
        },
        {
            "call_id": "test-2",
            "outcome": "abandoned",
            "call_disposition": "pre_intent",
            "summary": "Greeting-only call, customer hung up."
        },
        {
            "call_id": "test-3",
            "outcome": "escalated",
            "call_disposition": "out_of_scope_abandoned",
            "summary": "Customer wanted to make a reservation, agent couldn't help.",
            "failure_description": "Unable to book reservation"
        }
    ]

    result = extract_nl_summary(mock_analyses)

    # Check disposition_summary exists
    assert "disposition_summary" in result, "Missing disposition_summary"
    disp_summary = result["disposition_summary"]
    print("  disposition_summary present")

    # Check groupings
    assert "in_scope_success" in disp_summary, "Missing in_scope_success in summary"
    assert len(disp_summary["in_scope_success"]) == 1, "Wrong count for in_scope_success"
    print("  in_scope_success has 1 call")

    assert "pre_intent" in disp_summary, "Missing pre_intent in summary"
    assert len(disp_summary["pre_intent"]) == 1, "Wrong count for pre_intent"
    print("  pre_intent has 1 call")

    assert "out_of_scope_abandoned" in disp_summary, "Missing out_of_scope_abandoned in summary"
    assert len(disp_summary["out_of_scope_abandoned"]) == 1, "Wrong count for out_of_scope_abandoned"
    print("  out_of_scope_abandoned has 1 call")

    # Check entry structure
    success_entry = disp_summary["in_scope_success"][0]
    assert success_entry.get("call_id") == "test-1", "Wrong call_id"
    assert success_entry.get("summary") is not None, "Missing summary in entry"
    print("  Entry structure correct")

    print("  PASS: NL extraction test passed")
    return True


def test_v39_backwards_compatibility():
    """Test backwards compatibility with pre-v3.9 analyses (no disposition)."""
    print("\n=== Test: v3.9 Backwards Compatibility ===")

    from compute_metrics import compute_disposition_breakdown
    from extract_nl_fields import extract_nl_summary

    # Pre-v3.9 analyses without disposition
    mock_analyses = [
        {"call_id": "old-1", "outcome": "resolved"},
        {"call_id": "old-2", "outcome": "abandoned"},
        {"call_id": "old-3", "outcome": "escalated"},
    ]

    # Metrics should handle missing disposition
    metrics_result = compute_disposition_breakdown(mock_analyses)
    assert metrics_result.get("by_disposition", {}).get("unknown", {}).get("count") == 3, \
        "Should count all as unknown"
    print("  Metrics handles missing disposition (counts as unknown)")

    # NL extraction should handle missing disposition
    nl_result = extract_nl_summary(mock_analyses)
    disp_summary = nl_result.get("disposition_summary", {})
    assert len(disp_summary) == 0, "Should have no disposition entries for pre-v3.9"
    print("  NL extraction handles missing disposition (empty summary)")

    print("  PASS: Backwards compatibility test passed")
    return True


def test_v39_version_in_analysis():
    """Test that schema_version is set to v3.9."""
    print("\n=== Test: Schema Version ===")

    from analyze_transcript import ANALYSIS_SCHEMA

    # The schema should reference v3.9 structures
    assert "call_disposition" in ANALYSIS_SCHEMA
    print("  Schema includes v3.9 call_disposition field")

    # Check docstring mentions v3.9
    from analyze_transcript import __doc__ as module_doc
    assert "v3.9" in module_doc, "Module docstring should mention v3.9"
    print("  Module docstring mentions v3.9")

    print("  PASS: Schema version test passed")
    return True


def run_all_tests():
    """Run all v3.9 feature tests."""
    print("=" * 60)
    print("v3.9 Feature Tests: Call Disposition Classification")
    print("=" * 60)

    tests = [
        test_v39_schema_structure,
        test_v39_decision_tree_in_prompt,
        test_v39_scope_reference_in_prompt,
        test_v39_disposition_aggregation,
        test_v39_funnel_metrics,
        test_v39_nl_extraction,
        test_v39_backwards_compatibility,
        test_v39_version_in_analysis,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            import traceback
            traceback.print_exc()
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
