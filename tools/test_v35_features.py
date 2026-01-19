#!/usr/bin/env python3
"""
Unit tests for v3.5 features: Training Insights & Emergent Patterns

Tests cover:
- extract_nl_fields.py: training_details and all_additional_intents extraction
- render_report.py: Training narrative, cross-correlations, emergent patterns, secondary intents
"""

import json
import sys
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))

from extract_nl_fields import extract_nl_summary
from render_report import render_markdown


def create_mock_analyses():
    """Create mock v3 analysis data with training and intent fields."""
    return [
        {
            "schema_version": "v3",
            "call_id": "call_001",
            "outcome": "escalated",
            "failure_point": "auth_restriction",
            "failure_description": "Customer failed phone verification",
            "training_opportunity": "verification",
            "agent_miss_detail": "Did not offer contract ID fallback",
            "additional_intents": "Also wanted to check reservation status"
        },
        {
            "schema_version": "v3",
            "call_id": "call_002",
            "outcome": "abandoned",
            "failure_point": "nlu_miss",
            "failure_description": "Agent repeated same question",
            "training_opportunity": "clarification",
            "agent_miss_detail": "Did not rephrase after first NLU failure"
        },
        {
            "schema_version": "v3",
            "call_id": "call_003",
            "outcome": "escalated",
            "failure_point": "explicit_request",
            "failure_description": "Customer asked for representative",
            "training_opportunity": "escalation_handling",
            "agent_miss_detail": "Ignored explicit transfer request",
            "additional_intents": "Wanted to cancel membership"
        },
        {
            "schema_version": "v3",
            "call_id": "call_004",
            "outcome": "resolved",
            "failure_point": "none"
        },
        {
            "schema_version": "v3",
            "call_id": "call_005",
            "outcome": "escalated",
            "failure_point": "auth_restriction",
            "failure_description": "Phone lookup failed for valid customer",
            "training_opportunity": "verification",
            "agent_miss_detail": "No alternative verification offered"
        }
    ]


def create_mock_report():
    """Create mock combined report with v3.5 LLM insights."""
    return {
        "deterministic_metrics": {
            "metadata": {
                "total_calls_analyzed": 100,
                "report_generated": "2025-01-19T10:00:00"
            },
            "key_rates": {
                "success_rate": 0.45,
                "escalation_rate": 0.35,
                "failure_rate": 0.55
            },
            "quality_scores": {
                "customer_effort": {"mean": 3.2, "median": 3.0, "std": 0.8, "n": 100}
            },
            "training_priorities": {
                "verification": 666,
                "clarification": 489,
                "escalation_handling": 269
            },
            "failure_analysis": {
                "by_failure_point": {
                    "policy_gap": {"count": 500, "rate": 0.50},
                    "auth_restriction": {"count": 200, "rate": 0.20}
                }
            },
            "policy_gap_breakdown": {
                "by_category": {}
            }
        },
        "llm_insights": {
            "executive_summary": "Test summary.",
            "root_cause_analysis": {
                "primary_driver": "Verification failures",
                "contributing_factors": ["Rigid phone lookup", "No fallback"],
                "evidence": "666 verification training gaps"
            },
            "actionable_recommendations": [],
            "trend_narratives": {},
            "verbatim_highlights": {},
            "key_metrics_descriptions": {
                "success_rate": "Driven by verification failures blocking valid customers",
                "escalation_rate": "High due to auth restrictions with no fallback",
                "failure_rate": "Majority fail at verification or policy gaps",
                "customer_effort": "Multiple attempts needed due to rigid verification"
            },
            "failure_type_descriptions": {},
            "policy_gap_descriptions": {},
            "major_failure_breakdowns": {},
            "customer_ask_clusters": [],
            # v3.5 fields
            "training_analysis": {
                "narrative": "The agent shows systematic weakness in verification flows and escalation handling. These are interconnected: failed verifications often trigger escalation requests which are then mishandled.",
                "top_priorities": [
                    {
                        "skill": "verification",
                        "count": 666,
                        "why": "Phone lookup fails for valid customers with no alternative",
                        "action": "Add contract ID/address fallback to verification flow"
                    },
                    {
                        "skill": "clarification",
                        "count": 489,
                        "why": "Repeats same question vs. rephrasing after NLU miss",
                        "action": "Train on intent recognition and rephrase patterns"
                    },
                    {
                        "skill": "escalation_handling",
                        "count": 269,
                        "why": "Ignores explicit requests for human transfer",
                        "action": "Prioritize human transfer intent detection"
                    }
                ],
                "cross_correlations": [
                    {
                        "pattern": "verification + auth_restriction",
                        "count": 145,
                        "insight": "Customers who fail phone lookup are completely blocked with no recovery path"
                    },
                    {
                        "pattern": "clarification + nlu_miss",
                        "count": 89,
                        "insight": "Agent doesn't rephrase after NLU failure, leading to repeated misunderstanding"
                    }
                ]
            },
            "emergent_patterns": [
                {
                    "name": "Verify-Then-Dump",
                    "frequency": "~30% of abandoned calls",
                    "description": "Agent completes full verification, then terminates call due to closed department",
                    "significance": "Major driver of customer frustration and repeat calls",
                    "example_call_ids": ["9b6b3888", "a07abc2f", "33195b90"]
                }
            ],
            "secondary_intents_analysis": {
                "narrative": "Many customers have secondary needs beyond their primary intent that go unaddressed.",
                "clusters": [
                    {
                        "cluster": "Exit/Sell Timeshare",
                        "count": 50,
                        "implication": "High-value retention opportunity being missed"
                    },
                    {
                        "cluster": "Check Reservation Status",
                        "count": 35,
                        "implication": "Could be handled proactively during primary flow"
                    }
                ]
            }
        }
    }


def test_extract_training_details():
    """Training details should be extracted with context."""
    analyses = create_mock_analyses()
    nl_summary = extract_nl_summary(analyses)

    # Check training_details exists and has expected fields
    assert "training_details" in nl_summary, "training_details should be in nl_summary"
    training_details = nl_summary["training_details"]

    # Should have 4 entries (call_004 has no training_opportunity)
    assert len(training_details) == 4, f"Expected 4 training details, got {len(training_details)}"

    # Check first entry has all expected fields
    first = training_details[0]
    assert "call_id" in first, "training_detail should have call_id"
    assert "opportunity" in first, "training_detail should have opportunity"
    assert "outcome" in first, "training_detail should have outcome"
    assert "failure_point" in first, "training_detail should have failure_point"

    # Verify verification type appears twice
    verification_count = sum(1 for t in training_details if t.get("opportunity") == "verification")
    assert verification_count == 2, f"Expected 2 verification entries, got {verification_count}"

    print("  [PASS] test_extract_training_details")


def test_extract_additional_intents():
    """Additional intents should be extracted."""
    analyses = create_mock_analyses()
    nl_summary = extract_nl_summary(analyses)

    # Check all_additional_intents exists
    assert "all_additional_intents" in nl_summary, "all_additional_intents should be in nl_summary"
    intents = nl_summary["all_additional_intents"]

    # Should have 2 entries (call_001 and call_003)
    assert len(intents) == 2, f"Expected 2 additional intents, got {len(intents)}"

    # Check first entry has expected fields
    first = intents[0]
    assert "call_id" in first, "additional_intent should have call_id"
    assert "outcome" in first, "additional_intent should have outcome"
    assert "intent" in first, "additional_intent should have intent"

    print("  [PASS] test_extract_additional_intents")


def test_training_section_has_narrative():
    """Training section should have narrative, not just counts."""
    report = create_mock_report()
    markdown = render_markdown(report)

    # Should contain Training & Development header
    assert "## Training & Development" in markdown, "Should have Training & Development section"

    # Should contain narrative text
    assert "systematic weakness" in markdown, "Should contain training narrative"

    # Should contain Priority Skills table
    assert "### Priority Skills" in markdown, "Should have Priority Skills subsection"
    assert "Root Cause" in markdown, "Should have Root Cause column"
    assert "Recommended Action" in markdown, "Should have Recommended Action column"

    print("  [PASS] test_training_section_has_narrative")


def test_cross_correlations_rendered():
    """Cross-dimensional patterns should appear."""
    report = create_mock_report()
    markdown = render_markdown(report)

    # Should contain Cross-Dimensional Patterns header
    assert "### Cross-Dimensional Patterns" in markdown, "Should have Cross-Dimensional Patterns subsection"

    # Should contain specific correlation
    assert "verification + auth_restriction" in markdown, "Should render verification correlation"
    assert "145 calls" in markdown, "Should render correlation count"

    print("  [PASS] test_cross_correlations_rendered")


def test_emergent_patterns_rendered():
    """Emergent patterns section should appear when LLM provides them."""
    report = create_mock_report()
    markdown = render_markdown(report)

    # Should contain Emergent Patterns header
    assert "## Emergent Patterns" in markdown, "Should have Emergent Patterns section"

    # Should contain pattern name and details
    assert "Verify-Then-Dump" in markdown, "Should render pattern name"
    assert "~30% of abandoned calls" in markdown, "Should render frequency"
    assert "Major driver of customer frustration" in markdown, "Should render significance"

    # Should contain example call IDs
    assert "9b6b3888" in markdown, "Should render example call IDs"

    print("  [PASS] test_emergent_patterns_rendered")


def test_secondary_intents_rendered():
    """Secondary intents should be clustered and rendered."""
    report = create_mock_report()
    markdown = render_markdown(report)

    # Should contain Secondary Customer Needs header
    assert "## Secondary Customer Needs" in markdown, "Should have Secondary Customer Needs section"

    # Should contain narrative
    assert "secondary needs beyond their primary intent" in markdown, "Should render narrative"

    # Should contain cluster table
    assert "Exit/Sell Timeshare" in markdown, "Should render cluster name"
    assert "50" in markdown, "Should render cluster count"
    assert "retention opportunity" in markdown, "Should render implication"

    print("  [PASS] test_secondary_intents_rendered")


def test_empty_v35_fields_handled():
    """Empty v3.5 fields should not cause errors or render empty sections."""
    report = create_mock_report()
    # Remove v3.5 fields
    report["llm_insights"]["training_analysis"] = {}
    report["llm_insights"]["emergent_patterns"] = []
    report["llm_insights"]["secondary_intents_analysis"] = {}

    markdown = render_markdown(report)

    # Should still render without errors
    assert "## Training & Development" in markdown, "Training section should still appear (from metrics)"

    # Should not have empty emergent patterns section
    if "## Emergent Patterns" in markdown:
        # If section exists, it should have content
        assert "Unnamed" not in markdown, "Should not render empty patterns"

    print("  [PASS] test_empty_v35_fields_handled")


def run_all_tests():
    """Run all v3.5 feature tests."""
    print("\n" + "=" * 60)
    print("v3.5 FEATURE TESTS")
    print("=" * 60)

    tests = [
        test_extract_training_details,
        test_extract_additional_intents,
        test_training_section_has_narrative,
        test_cross_correlations_rendered,
        test_emergent_patterns_rendered,
        test_secondary_intents_rendered,
        test_empty_v35_fields_handled
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test_func.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  [ERROR] {test_func.__name__}: {e}")
            failed += 1

    print("\n" + "-" * 40)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 60)

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(run_all_tests())
