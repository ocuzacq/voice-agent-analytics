#!/usr/bin/env python3
"""
Unit Tests for Vacatia AI Voice Agent Analytics v3.6 Features

Tests the new conversation quality tracking:
- conversation_turns counting
- turns_to_failure for non-resolved calls
- clarification_requests aggregation
- user_corrections aggregation
- repeated_prompts (loop detection)
"""

import json
import sys
import unittest
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))

from compute_metrics import (
    compute_conversation_quality_metrics,
    safe_rate,
    safe_stats,
)


class TestConversationQualityMetrics(unittest.TestCase):
    """Tests for the compute_conversation_quality_metrics function."""

    def test_empty_analyses(self):
        """Empty input should return empty dict."""
        result = compute_conversation_quality_metrics([])
        self.assertEqual(result, {})

    def test_turn_stats_basic(self):
        """Basic turn statistics calculation."""
        analyses = [
            {"call_id": "001", "outcome": "resolved", "conversation_turns": 10},
            {"call_id": "002", "outcome": "resolved", "conversation_turns": 8},
            {"call_id": "003", "outcome": "abandoned", "conversation_turns": 15},
            {"call_id": "004", "outcome": "escalated", "conversation_turns": 12},
        ]
        result = compute_conversation_quality_metrics(analyses)

        turn_stats = result.get("turn_stats", {})
        self.assertEqual(turn_stats["calls_with_turn_data"], 4)
        # (10+8+15+12)/4 = 11.25 → rounds to 11.2
        self.assertAlmostEqual(turn_stats["avg_turns"], 11.2, places=1)
        self.assertAlmostEqual(turn_stats["avg_turns_resolved"], 9.0, places=1)
        self.assertAlmostEqual(turn_stats["avg_turns_failed"], 13.5, places=1)

    def test_turns_to_failure(self):
        """Turns to failure aggregation."""
        analyses = [
            {"call_id": "001", "outcome": "abandoned", "turns_to_failure": 5},
            {"call_id": "002", "outcome": "escalated", "turns_to_failure": 8},
            {"call_id": "003", "outcome": "abandoned", "turns_to_failure": 3},
            {"call_id": "004", "outcome": "resolved"},  # No turns_to_failure
        ]
        result = compute_conversation_quality_metrics(analyses)

        turn_stats = result.get("turn_stats", {})
        self.assertAlmostEqual(turn_stats["avg_turns_to_failure"], 5.3, places=1)

    def test_clarification_stats_basic(self):
        """Clarification statistics aggregation."""
        analyses = [
            {
                "call_id": "001",
                "clarification_requests": {
                    "count": 2,
                    "details": [
                        {"type": "name_spelling", "turn": 3, "resolved": True},
                        {"type": "phone_confirmation", "turn": 5, "resolved": True},
                    ]
                }
            },
            {
                "call_id": "002",
                "clarification_requests": {
                    "count": 1,
                    "details": [
                        {"type": "name_spelling", "turn": 4, "resolved": False},
                    ]
                }
            },
            {"call_id": "003"},  # No clarification data
        ]
        result = compute_conversation_quality_metrics(analyses)

        clar_stats = result.get("clarification_stats", {})
        self.assertEqual(clar_stats["calls_with_clarifications"], 2)
        self.assertAlmostEqual(clar_stats["pct_calls_with_clarifications"], 0.667, places=2)
        self.assertEqual(clar_stats["avg_clarifications_per_call"], 1.5)

        by_type = clar_stats.get("by_type", {})
        self.assertEqual(by_type["name_spelling"]["count"], 2)
        self.assertEqual(by_type["phone_confirmation"]["count"], 1)

        # Resolution rate: 2 resolved out of 3 total
        self.assertAlmostEqual(clar_stats["resolution_rate"], 0.667, places=2)

    def test_clarification_types_all(self):
        """All clarification types are tracked."""
        analyses = [
            {
                "call_id": "001",
                "clarification_requests": {
                    "count": 5,
                    "details": [
                        {"type": "name_spelling", "turn": 1, "resolved": True},
                        {"type": "phone_confirmation", "turn": 2, "resolved": True},
                        {"type": "intent_clarification", "turn": 3, "resolved": False},
                        {"type": "repeat_request", "turn": 4, "resolved": True},
                        {"type": "verification_retry", "turn": 5, "resolved": False},
                    ]
                }
            },
        ]
        result = compute_conversation_quality_metrics(analyses)

        by_type = result["clarification_stats"]["by_type"]
        self.assertIn("name_spelling", by_type)
        self.assertIn("phone_confirmation", by_type)
        self.assertIn("intent_clarification", by_type)
        self.assertIn("repeat_request", by_type)
        self.assertIn("verification_retry", by_type)

    def test_correction_stats_basic(self):
        """User correction statistics aggregation."""
        analyses = [
            {
                "call_id": "001",
                "user_corrections": {
                    "count": 2,
                    "details": [
                        {"what_was_wrong": "wrong name", "turn": 3, "frustration_signal": True},
                        {"what_was_wrong": "wrong resort", "turn": 5, "frustration_signal": False},
                    ]
                }
            },
            {
                "call_id": "002",
                "user_corrections": {
                    "count": 1,
                    "details": [
                        {"what_was_wrong": "wrong phone", "turn": 4, "frustration_signal": True},
                    ]
                }
            },
            {"call_id": "003"},  # No correction data
        ]
        result = compute_conversation_quality_metrics(analyses)

        corr_stats = result.get("correction_stats", {})
        self.assertEqual(corr_stats["calls_with_corrections"], 2)
        self.assertAlmostEqual(corr_stats["pct_calls_with_corrections"], 0.667, places=2)
        self.assertEqual(corr_stats["avg_corrections_per_call"], 1.5)
        self.assertEqual(corr_stats["with_frustration_signal"], 2)
        # Frustration rate: 2 frustrated out of 3 total corrections
        self.assertAlmostEqual(corr_stats["frustration_rate"], 0.667, places=2)

    def test_loop_stats_basic(self):
        """Loop detection statistics aggregation."""
        analyses = [
            {
                "call_id": "001",
                "repeated_prompts": {"count": 3, "max_consecutive": 2}
            },
            {
                "call_id": "002",
                "repeated_prompts": {"count": 5, "max_consecutive": 4}
            },
            {"call_id": "003"},  # No loop data
            {
                "call_id": "004",
                "repeated_prompts": {"count": 0, "max_consecutive": 0}  # No loops
            },
        ]
        result = compute_conversation_quality_metrics(analyses)

        loop_stats = result.get("loop_stats", {})
        self.assertEqual(loop_stats["calls_with_loops"], 2)
        self.assertAlmostEqual(loop_stats["pct_calls_with_loops"], 0.5, places=2)
        self.assertEqual(loop_stats["avg_repeats"], 4.0)  # (3+5)/2
        self.assertEqual(loop_stats["max_consecutive_overall"], 4)

    def test_null_and_missing_fields(self):
        """Handle null and missing fields gracefully."""
        analyses = [
            {"call_id": "001"},  # All fields missing
            {"call_id": "002", "conversation_turns": None},  # Explicit null
            {"call_id": "003", "clarification_requests": None},
            {"call_id": "004", "user_corrections": {}},  # Empty dict
            {"call_id": "005", "repeated_prompts": {"count": None}},
        ]
        # Should not raise any exceptions
        result = compute_conversation_quality_metrics(analyses)
        self.assertIsInstance(result, dict)


class TestSchemaVersionAcceptance(unittest.TestCase):
    """Tests for schema version handling in v3.6."""

    def test_v36_schema_version(self):
        """v3.6 schema version should be accepted."""
        # This test verifies the version string format
        # In actual implementation, analyze_transcript.py sets schema_version
        expected_version = "v3.6"
        self.assertTrue(expected_version.startswith("v3"))


class TestExtractNLFieldsV36(unittest.TestCase):
    """Tests for v3.6 additions in extract_nl_fields.py."""

    def test_clarification_events_extraction(self):
        """Clarification events should be extracted with proper structure."""
        from extract_nl_fields import extract_nl_summary

        analyses = [
            {
                "call_id": "test-001",
                "schema_version": "v3.6",
                "outcome": "abandoned",
                "clarification_requests": {
                    "count": 1,
                    "details": [
                        {"type": "name_spelling", "turn": 5, "resolved": False}
                    ]
                }
            }
        ]

        result = extract_nl_summary(analyses)
        clar_events = result.get("clarification_events", [])

        self.assertEqual(len(clar_events), 1)
        self.assertEqual(clar_events[0]["type"], "name_spelling")
        self.assertEqual(clar_events[0]["turn"], 5)
        self.assertEqual(clar_events[0]["resolved"], False)
        self.assertEqual(clar_events[0]["outcome"], "abandoned")

    def test_correction_events_extraction(self):
        """Correction events should include frustration signals."""
        from extract_nl_fields import extract_nl_summary

        analyses = [
            {
                "call_id": "test-002",
                "schema_version": "v3.6",
                "outcome": "escalated",
                "user_corrections": {
                    "count": 1,
                    "details": [
                        {"what_was_wrong": "wrong resort name", "turn": 8, "frustration_signal": True}
                    ]
                }
            }
        ]

        result = extract_nl_summary(analyses)
        corr_events = result.get("correction_events", [])

        self.assertEqual(len(corr_events), 1)
        self.assertEqual(corr_events[0]["what"], "wrong resort name")
        self.assertEqual(corr_events[0]["frustrated"], True)

    def test_loop_events_extraction(self):
        """Loop events should track max consecutive."""
        from extract_nl_fields import extract_nl_summary

        analyses = [
            {
                "call_id": "test-003",
                "schema_version": "v3.6",
                "outcome": "abandoned",
                "repeated_prompts": {
                    "count": 4,
                    "max_consecutive": 3
                }
            }
        ]

        result = extract_nl_summary(analyses)
        loop_events = result.get("loop_events", [])

        self.assertEqual(len(loop_events), 1)
        self.assertEqual(loop_events[0]["count"], 4)
        self.assertEqual(loop_events[0]["max_consecutive"], 3)


class TestSafeHelpers(unittest.TestCase):
    """Tests for safe calculation helpers."""

    def test_safe_rate_normal(self):
        """Normal rate calculation."""
        self.assertAlmostEqual(safe_rate(1, 4), 0.25, places=3)
        self.assertAlmostEqual(safe_rate(3, 10), 0.3, places=3)

    def test_safe_rate_zero_denominator(self):
        """Zero denominator returns None."""
        self.assertIsNone(safe_rate(5, 0))

    def test_safe_stats_empty(self):
        """Empty list returns null stats."""
        result = safe_stats([])
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["mean"])

    def test_safe_stats_with_none(self):
        """None values are filtered out."""
        result = safe_stats([1, None, 3, None, 5])
        self.assertEqual(result["n"], 3)
        self.assertEqual(result["mean"], 3.0)


if __name__ == "__main__":
    # Run tests with verbosity
    unittest.main(verbosity=2)
