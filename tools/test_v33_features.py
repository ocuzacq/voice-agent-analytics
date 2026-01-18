#!/usr/bin/env python3
"""
Unit Tests for v3.3 Features - Voice Agent Analytics

Tests the following v3.3 enhancements:
1. Validation bug detection (failure_point=none for non-resolved calls)
2. Call ID inclusion in insights prompts
3. Semantic clustering rendering
4. Explanatory qualifiers rendering
5. Call ID references in recommendations

Run with: python3 -m pytest tools/test_v33_features.py -v
"""

import json
import sys
from pathlib import Path

import pytest

# Add tools directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from compute_metrics import validate_failure_consistency, generate_report
from generate_insights import build_insights_prompt
from render_report import render_markdown


# =============================================================================
# Test 1: Validation Bug Detection
# =============================================================================

class TestValidateFailureConsistency:
    """Tests for validate_failure_consistency function."""

    def test_catches_invalid_failure_point_none(self):
        """failure_point=none should be flagged for non-resolved calls."""
        analyses = [
            {"call_id": "abc", "outcome": "abandoned", "failure_point": "none"},  # Invalid
            {"call_id": "def", "outcome": "resolved", "failure_point": "none"},   # Valid
            {"call_id": "ghi", "outcome": "escalated", "failure_point": "policy_gap"},  # Valid
        ]
        warnings = validate_failure_consistency(analyses)

        assert len(warnings["failure_point_inconsistencies"]) == 1
        assert warnings["failure_point_inconsistencies"][0]["call_id"] == "abc"
        assert warnings["failure_point_inconsistencies"][0]["outcome"] == "abandoned"

    def test_catches_multiple_invalid_entries(self):
        """Should catch all invalid entries, not just the first."""
        analyses = [
            {"call_id": "a1", "outcome": "abandoned", "failure_point": "none"},
            {"call_id": "a2", "outcome": "escalated", "failure_point": "none"},
            {"call_id": "a3", "outcome": "unclear", "failure_point": "none"},
            {"call_id": "a4", "outcome": "resolved", "failure_point": "none"},  # Valid
        ]
        warnings = validate_failure_consistency(analyses)

        assert len(warnings["failure_point_inconsistencies"]) == 3
        invalid_ids = [w["call_id"] for w in warnings["failure_point_inconsistencies"]]
        assert "a1" in invalid_ids
        assert "a2" in invalid_ids
        assert "a3" in invalid_ids
        assert "a4" not in invalid_ids

    def test_empty_analyses_returns_empty_warnings(self):
        """Empty analyses should return empty warnings."""
        warnings = validate_failure_consistency([])
        assert warnings["failure_point_inconsistencies"] == []

    def test_all_valid_returns_empty_warnings(self):
        """All valid entries should return empty warnings."""
        analyses = [
            {"call_id": "a", "outcome": "resolved", "failure_point": "none"},
            {"call_id": "b", "outcome": "abandoned", "failure_point": "policy_gap"},
            {"call_id": "c", "outcome": "escalated", "failure_point": "nlu_miss"},
        ]
        warnings = validate_failure_consistency(analyses)
        assert warnings["failure_point_inconsistencies"] == []

    def test_missing_fields_handled_gracefully(self):
        """Should handle missing outcome or failure_point fields."""
        analyses = [
            {"call_id": "a"},  # Missing both
            {"call_id": "b", "outcome": "abandoned"},  # Missing failure_point
            {"call_id": "c", "failure_point": "none"},  # Missing outcome
        ]
        # Should not crash
        warnings = validate_failure_consistency(analyses)
        assert "failure_point_inconsistencies" in warnings


# =============================================================================
# Test 2: Call ID Inclusion in Prompt
# =============================================================================

class TestBuildInsightsPromptCallIds:
    """Tests for call ID inclusion in build_insights_prompt."""

    def test_includes_call_ids_in_failure_entries(self):
        """NL data in prompt should include call_ids for traceability."""
        nl_summary = {
            "by_failure_type": {
                "policy_gap": [
                    {"call_id": "abc123", "outcome": "abandoned", "description": "test failure"}
                ]
            },
            "all_verbatims": [],
            "policy_gap_details": []
        }
        prompt = build_insights_prompt({}, nl_summary)
        assert "abc123" in prompt

    def test_includes_call_ids_in_verbatims(self):
        """Verbatims should include call_ids."""
        nl_summary = {
            "by_failure_type": {},
            "all_verbatims": [
                {"call_id": "def456", "outcome": "escalated", "quote": "help me please"}
            ],
            "policy_gap_details": []
        }
        prompt = build_insights_prompt({}, nl_summary)
        assert "def456" in prompt

    def test_includes_call_ids_in_policy_gaps(self):
        """Policy gap details should include call_ids."""
        nl_summary = {
            "by_failure_type": {},
            "all_verbatims": [],
            "policy_gap_details": [
                {"call_id": "ghi789", "category": "capability_limit", "gap": "test gap", "ask": "test ask", "blocker": "test blocker"}
            ]
        }
        prompt = build_insights_prompt({}, nl_summary)
        assert "ghi789" in prompt

    def test_handles_missing_call_ids_gracefully(self):
        """Gracefully handle missing call_ids."""
        nl_summary = {
            "by_failure_type": {},
            "all_verbatims": [
                {"outcome": "abandoned", "quote": "no call_id here"}
            ],
            "policy_gap_details": []
        }
        # Should not crash
        prompt = build_insights_prompt({}, nl_summary)
        # Should use placeholder or handle gracefully
        assert "abandoned" in prompt or "unknown" in prompt


# =============================================================================
# Test 3: Semantic Clustering Rendering
# =============================================================================

class TestRenderClusteredCustomerAsks:
    """Tests for rendering clustered customer asks."""

    def test_render_clustered_customer_asks(self):
        """Report should render clustered asks when available."""
        report = {
            "deterministic_metrics": {
                "metadata": {"total_calls_analyzed": 100},
                "policy_gap_breakdown": {}
            },
            "llm_insights": {
                "customer_ask_clusters": [
                    {
                        "canonical_label": "Request human agent",
                        "member_asks": ["speak to rep", "talk to agent", "live person"],
                        "total_count": 25,
                        "example_call_ids": ["a", "b", "c"]
                    }
                ]
            }
        }
        md = render_markdown(report)
        assert "Request human agent" in md
        assert "25" in md  # Count shown
        assert "speak to rep" in md or "Examples:" in md  # Member asks shown

    def test_render_multiple_clusters(self):
        """Should render multiple clusters properly."""
        report = {
            "deterministic_metrics": {
                "metadata": {"total_calls_analyzed": 100},
                "policy_gap_breakdown": {}
            },
            "llm_insights": {
                "customer_ask_clusters": [
                    {"canonical_label": "Transfer request", "member_asks": ["transfer me"], "total_count": 20, "example_call_ids": []},
                    {"canonical_label": "Billing question", "member_asks": ["check balance"], "total_count": 15, "example_call_ids": []},
                ]
            }
        }
        md = render_markdown(report)
        assert "Transfer request" in md
        assert "Billing question" in md

    def test_render_fallback_to_raw_asks(self):
        """Report should fallback to raw asks if no clusters."""
        report = {
            "deterministic_metrics": {
                "metadata": {"total_calls_analyzed": 100},
                "policy_gap_breakdown": {
                    "top_customer_asks": [{"ask": "pay bill", "count": 5}]
                }
            },
            "llm_insights": {}  # No clusters
        }
        md = render_markdown(report)
        assert "pay bill" in md
        assert "5" in md  # Count shown


# =============================================================================
# Test 4: Explanatory Qualifiers Rendering
# =============================================================================

class TestRenderFailureTypeExplanations:
    """Tests for rendering failure type explanations (updated for v3.4 format)."""

    def test_render_failure_type_explanations(self):
        """Failure point breakdown should include inline descriptions (v3.4 format)."""
        report = {
            "deterministic_metrics": {
                "metadata": {"total_calls_analyzed": 100},
                "failure_analysis": {
                    "by_failure_point": {"policy_gap": {"count": 62, "rate": 0.62}}
                }
            },
            "llm_insights": {
                # v3.4 format: simple key-value descriptions
                "failure_type_descriptions": {
                    "policy_gap": "System limitations prevented resolution"
                }
            }
        }
        md = render_markdown(report)
        assert "policy_gap" in md
        assert "System limitations prevented resolution" in md

    def test_render_multiple_failure_explanations(self):
        """Should render descriptions for multiple failure types (v3.4 format)."""
        report = {
            "deterministic_metrics": {
                "metadata": {"total_calls_analyzed": 100},
                "failure_analysis": {
                    "by_failure_point": {
                        "policy_gap": {"count": 40, "rate": 0.40},
                        "nlu_miss": {"count": 20, "rate": 0.20}
                    }
                }
            },
            "llm_insights": {
                # v3.4 format: simple key-value descriptions
                "failure_type_descriptions": {
                    "policy_gap": "Policy explanation here",
                    "nlu_miss": "NLU explanation here"
                }
            }
        }
        md = render_markdown(report)
        assert "Policy explanation here" in md
        assert "NLU explanation here" in md


class TestRenderPolicyGapExplanations:
    """Tests for rendering policy gap category explanations (updated for v3.4 format)."""

    def test_render_policy_gap_explanations(self):
        """Policy gap categories should include inline context (v3.4 format)."""
        report = {
            "deterministic_metrics": {
                "metadata": {"total_calls_analyzed": 100},
                "policy_gap_breakdown": {
                    "by_category": {"capability_limit": {"count": 40, "rate": 0.40}}
                }
            },
            "llm_insights": {
                # v3.4 format: simple key-value descriptions
                "policy_gap_descriptions": {
                    "capability_limit": "Transfer-to-human and payment processing blocked"
                }
            }
        }
        md = render_markdown(report)
        assert "capability_limit" in md
        assert "Transfer-to-human" in md or "payment" in md

    def test_render_no_explanations_gracefully(self):
        """Should render without errors when no explanations provided."""
        report = {
            "deterministic_metrics": {
                "metadata": {"total_calls_analyzed": 100},
                "policy_gap_breakdown": {
                    "by_category": {"capability_limit": {"count": 40, "rate": 0.40}}
                }
            },
            "llm_insights": {}  # No explanations
        }
        # Should not crash
        md = render_markdown(report)
        assert "capability_limit" in md


# =============================================================================
# Test 5: Call ID References in Recommendations
# =============================================================================

class TestRenderRecommendationCallIds:
    """Tests for rendering supporting call_ids in recommendations."""

    def test_render_recommendation_call_ids(self):
        """Recommendations should show supporting call_ids."""
        report = {
            "deterministic_metrics": {"metadata": {"total_calls_analyzed": 100}},
            "llm_insights": {
                "actionable_recommendations": [
                    {
                        "priority": "P0",
                        "category": "capability",
                        "recommendation": "Add live agent transfer",
                        "expected_impact": "Resolve 45% of escalations",
                        "evidence": "Top customer need",
                        "supporting_call_ids": ["abc123", "def456", "ghi789"]
                    }
                ]
            }
        }
        md = render_markdown(report)
        assert "Add live agent transfer" in md
        # Should show call IDs somewhere
        assert "abc123" in md or "Example calls" in md or "Supporting" in md

    def test_render_recommendations_without_call_ids(self):
        """Should render recommendations gracefully without call_ids."""
        report = {
            "deterministic_metrics": {"metadata": {"total_calls_analyzed": 100}},
            "llm_insights": {
                "actionable_recommendations": [
                    {
                        "priority": "P0",
                        "category": "capability",
                        "recommendation": "Improve training",
                        "expected_impact": "Better outcomes",
                        "evidence": "Data shows issues"
                        # No supporting_call_ids
                    }
                ]
            }
        }
        # Should not crash
        md = render_markdown(report)
        assert "Improve training" in md


# =============================================================================
# Test 6: Integration - Validation in generate_report
# =============================================================================

class TestGenerateReportWithValidation:
    """Tests that validation warnings are included in generate_report output."""

    def test_generate_report_includes_validation_warnings(self):
        """generate_report should include validation_warnings in output."""
        analyses = [
            {
                "call_id": "test1",
                "outcome": "abandoned",
                "failure_point": "none",  # Invalid - should be flagged
                "agent_effectiveness": 3,
                "conversation_quality": 3,
                "customer_effort": 3
            },
            {
                "call_id": "test2",
                "outcome": "resolved",
                "failure_point": "none",  # Valid
                "agent_effectiveness": 5,
                "conversation_quality": 5,
                "customer_effort": 2
            }
        ]
        report = generate_report(analyses)

        # Check that validation_warnings is present
        assert "validation_warnings" in report
        assert "failure_point_inconsistencies" in report["validation_warnings"]
        assert len(report["validation_warnings"]["failure_point_inconsistencies"]) == 1


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
