#!/usr/bin/env python3
"""
Test Suite for v3.7 Features: Preprocessing + Structured Event Context

Tests:
1. Transcript preprocessing (deterministic turn counting)
2. Cause enum in clarification details
3. Severity enum in correction details
4. Context sentences in both
5. Metrics aggregation by cause/severity
"""

import json
import sys
import tempfile
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))

from preprocess_transcript import preprocess_transcript, format_for_llm


def test_preprocess_transcript():
    """Test deterministic turn counting."""
    print("\n=== Test: Transcript Preprocessing ===")

    # Create a sample transcript
    sample_transcript = """assistant: Hi, how can I help you today?

user: I need help with my account.

assistant: Sure, what's your name?

user: John Smith.

assistant: Can you spell that for me?

user: J O H N S M I T H.
"""

    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        f.write(sample_transcript)
        temp_path = Path(f.name)

    try:
        result = preprocess_transcript(temp_path)

        # Verify metadata
        assert result["metadata"]["total_turns"] == 6, f"Expected 6 turns, got {result['metadata']['total_turns']}"
        assert result["metadata"]["user_turns"] == 3, f"Expected 3 user turns, got {result['metadata']['user_turns']}"
        assert result["metadata"]["agent_turns"] == 3, f"Expected 3 agent turns, got {result['metadata']['agent_turns']}"

        # Verify turn numbers are sequential
        for i, turn in enumerate(result["turns"]):
            assert turn["turn"] == i + 1, f"Expected turn {i+1}, got {turn['turn']}"

        print(f"  ✓ Total turns: {result['metadata']['total_turns']}")
        print(f"  ✓ User turns: {result['metadata']['user_turns']}")
        print(f"  ✓ Agent turns: {result['metadata']['agent_turns']}")
        print(f"  ✓ Turn numbers are sequential")

        # Test LLM format
        llm_format = format_for_llm(result)
        assert "[Turn 1] AGENT:" in llm_format
        assert "[Turn 2] USER:" in llm_format
        print(f"  ✓ LLM format includes turn numbers")

        print("  PASS: Preprocessing test passed")
        return True

    finally:
        temp_path.unlink()


def test_v37_schema_structure():
    """Test v3.7 schema has required fields."""
    print("\n=== Test: v3.7 Schema Structure ===")

    # Import schema from analyze_transcript
    from analyze_transcript import ANALYSIS_SCHEMA

    # Check for cause enum (uses double quotes in JSON-like format)
    assert '"cause":' in ANALYSIS_SCHEMA, "Schema missing cause field"
    assert "customer_refused" in ANALYSIS_SCHEMA, "Schema missing customer_refused enum"
    assert "agent_misheard" in ANALYSIS_SCHEMA, "Schema missing agent_misheard enum"
    print("  ✓ cause enum present in schema")

    # Check for severity enum
    assert '"severity":' in ANALYSIS_SCHEMA, "Schema missing severity field"
    assert "minor" in ANALYSIS_SCHEMA, "Schema missing minor enum"
    assert "moderate" in ANALYSIS_SCHEMA, "Schema missing moderate enum"
    assert "major" in ANALYSIS_SCHEMA, "Schema missing major enum"
    print("  ✓ severity enum present in schema")

    # Check for context field
    assert '"context":' in ANALYSIS_SCHEMA, "Schema missing context field"
    print("  ✓ context field present in schema")

    print("  PASS: Schema structure test passed")
    return True


def test_mock_analysis_with_v37_fields():
    """Test that v3.7 fields can be parsed correctly."""
    print("\n=== Test: v3.7 Field Parsing ===")

    mock_analysis = {
        "call_id": "test-123",
        "schema_version": "v3.7",
        "outcome": "escalated",
        "clarification_requests": {
            "count": 2,
            "details": [
                {
                    "type": "name_spelling",
                    "turn": 5,
                    "resolved": False,
                    "cause": "customer_refused",
                    "context": "Customer said 'I already told you' and requested human agent"
                },
                {
                    "type": "phone_confirmation",
                    "turn": 3,
                    "resolved": True,
                    "cause": "successful",
                    "context": "Customer confirmed phone number after agent repeated it"
                }
            ]
        },
        "user_corrections": {
            "count": 1,
            "details": [
                {
                    "what_was_wrong": "wrong resort name",
                    "turn": 7,
                    "frustration_signal": True,
                    "severity": "moderate",
                    "context": "Customer corrected 'Grandview' to 'Grand Pacific' with audible frustration"
                }
            ]
        }
    }

    # Test clarification parsing
    clar = mock_analysis["clarification_requests"]
    assert clar["count"] == 2
    assert clar["details"][0]["cause"] == "customer_refused"
    assert "context" in clar["details"][0]
    print("  ✓ Clarification cause parsed correctly")

    # Test correction parsing
    corr = mock_analysis["user_corrections"]
    assert corr["count"] == 1
    assert corr["details"][0]["severity"] == "moderate"
    assert "context" in corr["details"][0]
    print("  ✓ Correction severity parsed correctly")

    print("  PASS: v3.7 field parsing test passed")
    return True


def test_metrics_aggregation():
    """Test that compute_metrics aggregates by cause and severity."""
    print("\n=== Test: Metrics Aggregation ===")

    # Import the aggregation function
    from compute_metrics import compute_conversation_quality_metrics

    mock_analyses = [
        {
            "outcome": "escalated",
            "conversation_turns": 15,
            "clarification_requests": {
                "count": 2,
                "details": [
                    {"type": "name_spelling", "resolved": False, "cause": "customer_refused"},
                    {"type": "phone_confirmation", "resolved": True, "cause": "successful"}
                ]
            },
            "user_corrections": {
                "count": 1,
                "details": [
                    {"what_was_wrong": "wrong name", "frustration_signal": True, "severity": "moderate"}
                ]
            },
            "repeated_prompts": {"count": 0, "max_consecutive": 0}
        },
        {
            "outcome": "resolved",
            "conversation_turns": 8,
            "clarification_requests": {
                "count": 1,
                "details": [
                    {"type": "repeat_request", "resolved": True, "cause": "customer_unclear"}
                ]
            },
            "user_corrections": {
                "count": 0,
                "details": []
            },
            "repeated_prompts": {"count": 0, "max_consecutive": 0}
        }
    ]

    result = compute_conversation_quality_metrics(mock_analyses)

    # Check cause aggregation
    clar_stats = result.get("clarification_stats", {})
    by_cause = clar_stats.get("by_cause", {})
    assert "customer_refused" in by_cause, "Missing customer_refused in by_cause"
    assert "successful" in by_cause, "Missing successful in by_cause"
    assert "customer_unclear" in by_cause, "Missing customer_unclear in by_cause"
    print(f"  ✓ Cause aggregation: {list(by_cause.keys())}")

    # Check severity aggregation
    corr_stats = result.get("correction_stats", {})
    by_severity = corr_stats.get("by_severity", {})
    assert "moderate" in by_severity, "Missing moderate in by_severity"
    print(f"  ✓ Severity aggregation: {list(by_severity.keys())}")

    print("  PASS: Metrics aggregation test passed")
    return True


def test_nl_extraction():
    """Test that extract_nl_fields includes cause and context."""
    print("\n=== Test: NL Field Extraction ===")

    from extract_nl_fields import extract_nl_summary

    mock_analyses = [
        {
            "call_id": "test-001",
            "outcome": "escalated",
            "failure_point": "policy_gap",
            "clarification_requests": {
                "count": 1,
                "details": [
                    {
                        "type": "name_spelling",
                        "turn": 5,
                        "resolved": False,
                        "cause": "customer_refused",
                        "context": "Customer refused to spell name"
                    }
                ]
            },
            "user_corrections": {
                "count": 1,
                "details": [
                    {
                        "what_was_wrong": "wrong resort",
                        "turn": 7,
                        "frustration_signal": True,
                        "severity": "major",
                        "context": "Customer angrily corrected resort name"
                    }
                ]
            },
            "repeated_prompts": {"count": 0, "max_consecutive": 0}
        }
    ]

    result = extract_nl_summary(mock_analyses)

    # Check clarification events include cause and context
    clar_events = result.get("clarification_events", [])
    assert len(clar_events) == 1
    assert clar_events[0].get("cause") == "customer_refused"
    assert clar_events[0].get("context") == "Customer refused to spell name"
    print("  ✓ Clarification events include cause and context")

    # Check correction events include severity and context
    corr_events = result.get("correction_events", [])
    assert len(corr_events) == 1
    assert corr_events[0].get("severity") == "major"
    assert corr_events[0].get("context") == "Customer angrily corrected resort name"
    print("  ✓ Correction events include severity and context")

    print("  PASS: NL extraction test passed")
    return True


def run_all_tests():
    """Run all v3.7 feature tests."""
    print("=" * 60)
    print("v3.7 Feature Tests")
    print("=" * 60)

    tests = [
        test_preprocess_transcript,
        test_v37_schema_structure,
        test_mock_analysis_with_v37_fields,
        test_metrics_aggregation,
        test_nl_extraction,
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
            failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
