#!/usr/bin/env python3
"""
Test Suite for v3.8 Features: Agent Loops with Typed Detection

Tests:
1. Schema has agent_loops with type enum
2. Loop type parsing
3. Metrics aggregation by loop type
4. NL extraction includes loop type and context
5. Backwards compatibility with repeated_prompts
"""

import json
import sys
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_v38_schema_structure():
    """Test v3.8 schema has agent_loops with type enum."""
    print("\n=== Test: v3.8 Schema Structure ===")

    # Import schema from analyze_transcript
    from analyze_transcript import ANALYSIS_SCHEMA

    # Check for agent_loops field
    assert '"agent_loops":' in ANALYSIS_SCHEMA, "Schema missing agent_loops field"
    print("  agent_loops field present in schema")

    # Check for type enum values
    assert "info_retry" in ANALYSIS_SCHEMA, "Schema missing info_retry enum"
    assert "intent_retry" in ANALYSIS_SCHEMA, "Schema missing intent_retry enum"
    assert "deflection" in ANALYSIS_SCHEMA, "Schema missing deflection enum"
    assert "comprehension" in ANALYSIS_SCHEMA, "Schema missing comprehension enum"
    assert "action_retry" in ANALYSIS_SCHEMA, "Schema missing action_retry enum"
    print("  type enum values present (info_retry, intent_retry, deflection, comprehension, action_retry)")

    # Check for context field
    assert '"context":' in ANALYSIS_SCHEMA, "Schema missing context field in agent_loops"
    print("  context field present in schema")

    # Verify repeated_prompts is NOT in schema
    # (Actually we don't remove it from schema, just replace guidance)

    print("  PASS: Schema structure test passed")
    return True


def test_mock_analysis_with_agent_loops():
    """Test that agent_loops fields can be parsed correctly."""
    print("\n=== Test: Agent Loops Field Parsing ===")

    mock_analysis = {
        "call_id": "test-123",
        "schema_version": "v3.8",
        "outcome": "escalated",
        "conversation_turns": 20,
        "agent_loops": {
            "count": 3,
            "details": [
                {
                    "type": "intent_retry",
                    "context": "Asked 'how can I help' at turn 18 after customer stated intent at turn 2"
                },
                {
                    "type": "info_retry",
                    "context": "Asked for name spelling twice despite customer providing it"
                },
                {
                    "type": "deflection",
                    "context": "Asked 'anything else?' while primary issue unresolved"
                }
            ]
        }
    }

    # Test loop parsing
    loops = mock_analysis["agent_loops"]
    assert loops["count"] == 3
    assert len(loops["details"]) == 3
    print("  Loop count parsed correctly")

    # Test type enum
    types = [d["type"] for d in loops["details"]]
    assert "intent_retry" in types
    assert "info_retry" in types
    assert "deflection" in types
    print("  Loop types parsed correctly")

    # Test context
    assert "context" in loops["details"][0]
    assert len(loops["details"][0]["context"]) > 10
    print("  Loop context parsed correctly")

    print("  PASS: Agent loops field parsing test passed")
    return True


def test_metrics_aggregation_agent_loops():
    """Test that compute_metrics aggregates by loop type."""
    print("\n=== Test: Metrics Aggregation for Agent Loops ===")

    # Import the aggregation function
    from compute_metrics import compute_conversation_quality_metrics

    mock_analyses = [
        {
            "outcome": "escalated",
            "conversation_turns": 20,
            "clarification_requests": {"count": 0, "details": []},
            "user_corrections": {"count": 0, "details": []},
            "agent_loops": {
                "count": 2,
                "details": [
                    {"type": "intent_retry", "context": "Re-asked intent after verification"},
                    {"type": "deflection", "context": "Generic question while stuck"}
                ]
            }
        },
        {
            "outcome": "resolved",
            "conversation_turns": 10,
            "clarification_requests": {"count": 0, "details": []},
            "user_corrections": {"count": 0, "details": []},
            "agent_loops": {
                "count": 1,
                "details": [
                    {"type": "comprehension", "context": "Asked customer to repeat"}
                ]
            }
        },
        {
            "outcome": "abandoned",
            "conversation_turns": 15,
            "clarification_requests": {"count": 0, "details": []},
            "user_corrections": {"count": 0, "details": []},
            "agent_loops": {
                "count": 1,
                "details": [
                    {"type": "intent_retry", "context": "Asked intent again after hold"}
                ]
            }
        }
    ]

    result = compute_conversation_quality_metrics(mock_analyses)

    # Check loop stats
    loop_stats = result.get("loop_stats", {})
    assert loop_stats.get("calls_with_loops") == 3, f"Expected 3 calls with loops, got {loop_stats.get('calls_with_loops')}"
    assert loop_stats.get("total_loops") == 4, f"Expected 4 total loops, got {loop_stats.get('total_loops')}"
    print(f"  Calls with loops: {loop_stats.get('calls_with_loops')}")
    print(f"  Total loops: {loop_stats.get('total_loops')}")

    # Check type aggregation
    by_type = loop_stats.get("by_type", {})
    assert "intent_retry" in by_type, "Missing intent_retry in by_type"
    assert "deflection" in by_type, "Missing deflection in by_type"
    assert "comprehension" in by_type, "Missing comprehension in by_type"
    assert by_type["intent_retry"]["count"] == 2, f"Expected 2 intent_retry, got {by_type['intent_retry']['count']}"
    print(f"  Type aggregation: {list(by_type.keys())}")

    # Check loop density calculation
    if loop_stats.get("loop_density"):
        print(f"  Loop density: {loop_stats.get('loop_density')}")

    print("  PASS: Metrics aggregation test passed")
    return True


def test_nl_extraction_agent_loops():
    """Test that extract_nl_fields includes loop type and context."""
    print("\n=== Test: NL Field Extraction for Agent Loops ===")

    from extract_nl_fields import extract_nl_summary

    mock_analyses = [
        {
            "call_id": "test-001",
            "outcome": "escalated",
            "failure_point": "policy_gap",
            "clarification_requests": {"count": 0, "details": []},
            "user_corrections": {"count": 0, "details": []},
            "agent_loops": {
                "count": 2,
                "details": [
                    {
                        "type": "intent_retry",
                        "context": "Re-asked 'how can I help' after customer already stated they wanted to make a payment"
                    },
                    {
                        "type": "deflection",
                        "context": "Asked 'is there anything else' while payment issue unresolved"
                    }
                ]
            }
        }
    ]

    result = extract_nl_summary(mock_analyses)

    # Check loop events include type and context
    loop_events = result.get("loop_events", [])
    assert len(loop_events) == 2, f"Expected 2 loop events, got {len(loop_events)}"

    # Check first loop event
    assert loop_events[0].get("type") == "intent_retry"
    assert loop_events[0].get("context") is not None
    assert "how can I help" in loop_events[0].get("context", "")
    print("  Loop events include type and context")

    # Check outcome is preserved
    assert loop_events[0].get("outcome") == "escalated"
    print("  Loop events include call outcome")

    print("  PASS: NL extraction test passed")
    return True


def test_backwards_compatibility():
    """Test backwards compatibility with repeated_prompts (v3.7 format)."""
    print("\n=== Test: Backwards Compatibility ===")

    from compute_metrics import compute_conversation_quality_metrics
    from extract_nl_fields import extract_nl_summary

    # v3.7 style analysis with repeated_prompts
    mock_v37_analyses = [
        {
            "call_id": "legacy-001",
            "outcome": "abandoned",
            "conversation_turns": 12,
            "clarification_requests": {"count": 0, "details": []},
            "user_corrections": {"count": 0, "details": []},
            "repeated_prompts": {
                "count": 3,
                "max_consecutive": 2
            }
        }
    ]

    # Test metrics still works
    result = compute_conversation_quality_metrics(mock_v37_analyses)
    loop_stats = result.get("loop_stats", {})
    assert loop_stats.get("calls_with_loops") == 1, "Backwards compatibility failed for metrics"
    print("  Metrics aggregation works with repeated_prompts")

    # Test NL extraction still works
    nl_result = extract_nl_summary(mock_v37_analyses)
    loop_events = nl_result.get("loop_events", [])
    assert len(loop_events) == 1, "Backwards compatibility failed for NL extraction"
    # Legacy events should have count/max_consecutive, not type/context
    assert loop_events[0].get("count") == 3
    print("  NL extraction works with repeated_prompts")

    print("  PASS: Backwards compatibility test passed")
    return True


def test_loop_types_enum():
    """Test that all loop types are documented correctly."""
    print("\n=== Test: Loop Types Enum ===")

    expected_types = [
        "info_retry",      # Re-asked for info already provided
        "intent_retry",    # Re-asked for intent already stated
        "deflection",      # Generic questions masking inability
        "comprehension",   # Couldn't hear, ask to repeat
        "action_retry"     # System retry
    ]

    # Import schema
    from analyze_transcript import ANALYSIS_SCHEMA

    for loop_type in expected_types:
        assert loop_type in ANALYSIS_SCHEMA, f"Missing loop type: {loop_type}"
        print(f"  {loop_type} documented")

    print("  PASS: All loop types documented")
    return True


def run_all_tests():
    """Run all v3.8 feature tests."""
    print("=" * 60)
    print("v3.8 Feature Tests: Agent Loops with Typed Detection")
    print("=" * 60)

    tests = [
        test_v38_schema_structure,
        test_mock_analysis_with_agent_loops,
        test_metrics_aggregation_agent_loops,
        test_nl_extraction_agent_loops,
        test_backwards_compatibility,
        test_loop_types_enum,
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
