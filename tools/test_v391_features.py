#!/usr/bin/env python3
"""
Test suite for v3.9.1 features: Loop Subject Granularity

Tests the new `subject` field in friction loops that identifies WHAT is being looped on.

Run with:
    python3 tools/test_v391_features.py
"""

import json
import sys
from pathlib import Path

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from compute_metrics import parse_loops, compute_conversation_quality_metrics
from extract_nl_fields import extract_loop_events, extract_nl_summary


def test_parse_loops_with_subject():
    """Test that parse_loops correctly extracts subject field from v3.9.1 format."""
    print("\n=== Test: parse_loops with subject field ===")

    # v3.9.1 format analysis with subject
    analysis_v391 = {
        "friction": {
            "turns": 15,
            "loops": [
                {"t": [5, 8, 11], "type": "info_retry", "subject": "name", "ctx": "asked 3x despite providing"},
                {"t": [12, 14], "type": "intent_retry", "subject": "fee_info", "ctx": "re-asked after verification"},
            ]
        }
    }

    loops = parse_loops(analysis_v391)
    assert len(loops) == 2, f"Expected 2 loops, got {len(loops)}"

    # Check first loop
    turns1, type1, subject1, ctx1 = loops[0]
    assert turns1 == [5, 8, 11], f"Expected turns [5, 8, 11], got {turns1}"
    assert type1 == "info_retry", f"Expected type 'info_retry', got {type1}"
    assert subject1 == "name", f"Expected subject 'name', got {subject1}"
    assert "3x" in ctx1, f"Expected context with '3x', got {ctx1}"

    # Check second loop
    turns2, type2, subject2, ctx2 = loops[1]
    assert type2 == "intent_retry", f"Expected type 'intent_retry', got {type2}"
    assert subject2 == "fee_info", f"Expected subject 'fee_info', got {subject2}"

    print("  PASS: parse_loops correctly extracts subject from v3.9.1 format")
    return True


def test_parse_loops_backwards_compat():
    """Test that parse_loops handles v3.8.5 format without subject."""
    print("\n=== Test: parse_loops backwards compatibility (v3.8.5) ===")

    # v3.8.5 format without subject
    analysis_v385 = {
        "friction": {
            "turns": 10,
            "loops": [
                {"t": [3, 6], "type": "deflection", "ctx": "asked anything else 2x"},
            ]
        }
    }

    loops = parse_loops(analysis_v385)
    assert len(loops) == 1, f"Expected 1 loop, got {len(loops)}"

    turns, loop_type, subject, ctx = loops[0]
    assert loop_type == "deflection", f"Expected type 'deflection', got {loop_type}"
    assert subject is None, f"Expected subject None for v3.8.5 format, got {subject}"

    print("  PASS: parse_loops handles v3.8.5 format without subject")
    return True


def test_extract_loop_events_with_subject():
    """Test that extract_loop_events includes subject in output."""
    print("\n=== Test: extract_loop_events with subject ===")

    analysis = {
        "call_id": "test-123",
        "outcome": "abandoned",
        "friction": {
            "turns": 20,
            "loops": [
                {"t": [5, 8], "type": "info_retry", "subject": "phone", "ctx": "phone asked twice"},
                {"t": [15, 18], "type": "comprehension", "subject": "unclear_speech", "ctx": "couldn't hear"},
            ]
        }
    }

    events = extract_loop_events(analysis)
    assert len(events) == 2, f"Expected 2 events, got {len(events)}"

    # Check first event
    event1 = events[0]
    assert event1["call_id"] == "test-123"
    assert event1["type"] == "info_retry"
    assert event1["subject"] == "phone", f"Expected subject 'phone', got {event1['subject']}"
    assert event1["outcome"] == "abandoned"

    # Check second event
    event2 = events[1]
    assert event2["subject"] == "unclear_speech", f"Expected subject 'unclear_speech', got {event2['subject']}"

    print("  PASS: extract_loop_events includes subject field")
    return True


def test_compute_metrics_subject_stats():
    """Test that compute_conversation_quality_metrics aggregates subject stats."""
    print("\n=== Test: compute_conversation_quality_metrics subject aggregation ===")

    analyses = [
        {
            "outcome": "abandoned",
            "friction": {
                "turns": 15,
                "loops": [
                    {"t": [5, 8], "type": "info_retry", "subject": "name", "ctx": "name asked twice"},
                ]
            }
        },
        {
            "outcome": "resolved",
            "friction": {
                "turns": 20,
                "loops": [
                    {"t": [10, 12], "type": "info_retry", "subject": "name", "ctx": "name asked twice"},
                    {"t": [15, 18], "type": "info_retry", "subject": "phone", "ctx": "phone asked twice"},
                ]
            }
        },
        {
            "outcome": "escalated",
            "friction": {
                "turns": 25,
                "loops": [
                    {"t": [5, 8, 11], "type": "intent_retry", "subject": "fee_info", "ctx": "fee info re-asked"},
                ]
            }
        }
    ]

    metrics = compute_conversation_quality_metrics(analyses)
    loop_stats = metrics.get("loop_stats", {})

    # Check basic stats
    assert loop_stats.get("calls_with_loops") == 3
    assert loop_stats.get("total_loops") == 4

    # Check subject stats (v3.9.1)
    assert loop_stats.get("loops_with_subject") == 4, f"Expected 4 loops with subject, got {loop_stats.get('loops_with_subject')}"

    # Check by_subject breakdown
    by_subject = loop_stats.get("by_subject", {})
    assert "info_retry" in by_subject, "Expected 'info_retry' in by_subject"
    assert "intent_retry" in by_subject, "Expected 'intent_retry' in by_subject"

    # Check info_retry subjects
    info_retry_subjects = by_subject.get("info_retry", {})
    assert "name" in info_retry_subjects, "Expected 'name' in info_retry subjects"
    assert info_retry_subjects["name"]["count"] == 2, f"Expected 2 name loops, got {info_retry_subjects['name']['count']}"

    # Check top_subjects
    top_subjects = loop_stats.get("top_subjects", [])
    assert len(top_subjects) > 0, "Expected top_subjects list"
    assert top_subjects[0]["subject"] == "name", f"Expected top subject 'name', got {top_subjects[0]['subject']}"
    assert top_subjects[0]["count"] == 2, f"Expected count 2, got {top_subjects[0]['count']}"

    print("  PASS: compute_conversation_quality_metrics aggregates subject stats correctly")
    return True


def test_extract_nl_summary_loop_subject_pairs():
    """Test that extract_nl_summary extracts loop_subject_pairs."""
    print("\n=== Test: extract_nl_summary loop_subject_pairs ===")

    analyses = [
        {
            "call_id": "call-001",
            "outcome": "abandoned",
            "friction": {
                "turns": 15,
                "loops": [
                    {"t": [5, 8], "type": "info_retry", "subject": "name", "ctx": "name asked twice"},
                ]
            }
        },
        {
            "call_id": "call-002",
            "outcome": "resolved",
            "friction": {
                "turns": 20,
                "loops": [
                    {"t": [10, 12], "type": "intent_retry", "subject": "fee_info", "ctx": "fee info re-asked"},
                ]
            }
        }
    ]

    nl_summary = extract_nl_summary(analyses)

    # Check loop_subject_pairs
    loop_subject_pairs = nl_summary.get("loop_subject_pairs", [])
    assert len(loop_subject_pairs) == 2, f"Expected 2 pairs, got {len(loop_subject_pairs)}"

    # Check first pair
    pair1 = loop_subject_pairs[0]
    assert pair1["loop_type"] == "info_retry"
    assert pair1["subject"] == "name"
    assert pair1["outcome"] == "abandoned"

    # Check second pair
    pair2 = loop_subject_pairs[1]
    assert pair2["loop_type"] == "intent_retry"
    assert pair2["subject"] == "fee_info"
    assert pair2["outcome"] == "resolved"

    print("  PASS: extract_nl_summary extracts loop_subject_pairs correctly")
    return True


def test_subject_values_by_loop_type():
    """Test parsing with various subject values from the guided lists."""
    print("\n=== Test: Subject values by loop type ===")

    # Test various subject values from the spec
    test_cases = [
        # info_retry subjects
        ("info_retry", "name"),
        ("info_retry", "phone"),
        ("info_retry", "address"),
        ("info_retry", "zip"),
        ("info_retry", "state"),
        ("info_retry", "account"),
        ("info_retry", "email"),
        # intent_retry subjects
        ("intent_retry", "fee_info"),
        ("intent_retry", "balance"),
        ("intent_retry", "payment_link"),
        ("intent_retry", "autopay_link"),
        ("intent_retry", "transfer"),
        ("intent_retry", "callback"),
        # deflection subjects
        ("deflection", "anything_else"),
        ("deflection", "other_help"),
        ("deflection", "clarify_request"),
        # comprehension subjects
        ("comprehension", "unclear_speech"),
        ("comprehension", "background_noise"),
        ("comprehension", "connection"),
        # action_retry subjects
        ("action_retry", "verification"),
        ("action_retry", "link_send"),
        ("action_retry", "lookup"),
        ("action_retry", "transfer_attempt"),
    ]

    for loop_type, subject in test_cases:
        analysis = {
            "friction": {
                "turns": 10,
                "loops": [
                    {"t": [5, 8], "type": loop_type, "subject": subject, "ctx": "test"}
                ]
            }
        }
        loops = parse_loops(analysis)
        assert len(loops) == 1
        _, parsed_type, parsed_subject, _ = loops[0]
        assert parsed_type == loop_type
        assert parsed_subject == subject

    print(f"  PASS: All {len(test_cases)} subject values parsed correctly")
    return True


def test_freeform_subject():
    """Test that freeform/edge case subjects are preserved."""
    print("\n=== Test: Freeform subject values ===")

    # Test edge case subjects not in the guided lists
    analysis = {
        "friction": {
            "turns": 15,
            "loops": [
                {"t": [5, 8], "type": "info_retry", "subject": "custom_field", "ctx": "some edge case"},
                {"t": [10, 12], "type": "action_retry", "subject": "payment_processing", "ctx": "another edge case"},
            ]
        }
    }

    loops = parse_loops(analysis)
    assert len(loops) == 2

    _, _, subject1, _ = loops[0]
    _, _, subject2, _ = loops[1]

    assert subject1 == "custom_field", f"Expected freeform subject 'custom_field', got {subject1}"
    assert subject2 == "payment_processing", f"Expected freeform subject 'payment_processing', got {subject2}"

    print("  PASS: Freeform subject values preserved correctly")
    return True


def main():
    """Run all v3.9.1 feature tests."""
    print("=" * 60)
    print("v3.9.1 Feature Tests: Loop Subject Granularity")
    print("=" * 60)

    tests = [
        test_parse_loops_with_subject,
        test_parse_loops_backwards_compat,
        test_extract_loop_events_with_subject,
        test_compute_metrics_subject_stats,
        test_extract_nl_summary_loop_subject_pairs,
        test_subject_values_by_loop_type,
        test_freeform_subject,
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

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    exit(main())
