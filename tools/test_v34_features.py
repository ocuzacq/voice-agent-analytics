#!/usr/bin/env python3
"""
Unit tests for v3.4 features: inline descriptions and sub-breakdowns.

Tests:
1. Key Metrics table has 4 columns with Context
2. Failure Point table has 4 columns with inline descriptions
3. Policy Gap table has 4 columns with inline descriptions
4. Sub-breakdowns render for major failure types (≥5%)
"""

import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from render_report import render_markdown


def create_mock_report(
    with_key_metrics_desc: bool = True,
    with_failure_type_desc: bool = True,
    with_policy_gap_desc: bool = True,
    with_major_breakdowns: bool = True,
) -> dict:
    """Create a mock report with v3.4 structure."""
    return {
        "deterministic_metrics": {
            "metadata": {
                "total_calls_analyzed": 500,
                "report_generated": "2025-01-18T10:00:00"
            },
            "key_rates": {
                "success_rate": 0.392,
                "escalation_rate": 0.208,
                "failure_rate": 0.608
            },
            "quality_scores": {
                "customer_effort": {"mean": 2.76, "median": 3.0, "std": 0.8, "n": 400}
            },
            "failure_analysis": {
                "by_failure_point": {
                    "policy_gap": {"count": 175, "rate": 0.576},
                    "other": {"count": 52, "rate": 0.171},
                    "tech_issue": {"count": 47, "rate": 0.155},
                    "nlu_miss": {"count": 17, "rate": 0.056},
                    "wrong_action": {"count": 8, "rate": 0.026},
                    "customer_confusion": {"count": 5, "rate": 0.016}
                }
            },
            "policy_gap_breakdown": {
                "by_category": {
                    "capability_limit": {"count": 80, "rate": 0.457},
                    "business_rule": {"count": 45, "rate": 0.257},
                    "data_access": {"count": 30, "rate": 0.171},
                    "auth_restriction": {"count": 15, "rate": 0.086},
                    "integration_missing": {"count": 5, "rate": 0.029}
                },
                "top_customer_asks": []
            }
        },
        "llm_insights": {
            "executive_summary": "Test executive summary.",
            "root_cause_analysis": {
                "primary_driver": "Dead-end escalation logic",
                "contributing_factors": ["Missing callback capability"],
                "evidence": "57.6% of failures are policy gaps"
            },
            "actionable_recommendations": [],
            "trend_narratives": {},
            "verbatim_highlights": {},
            "key_metrics_descriptions": {
                "success_rate": "Driven by dead-end escalation; verified customers can't reach humans",
                "escalation_rate": "45% explicitly requested human; most blocked by capacity limits",
                "failure_rate": "58% are policy gaps; primarily missing callback/queue capability",
                "customer_effort": "Verification succeeds but leads nowhere (verify-then-dump)"
            } if with_key_metrics_desc else {},
            "failure_type_descriptions": {
                "policy_gap": "Dead-end escalation logic; no callback capability",
                "other": "Caller hangups and unclear outcomes",
                "tech_issue": "Latency, audio glitches, infinite loops",
                "nlu_miss": "Entity extraction failures on names/dates",
                "wrong_action": "Incorrect department transfers",
                "customer_confusion": "Caller misunderstanding of options"
            } if with_failure_type_desc else {},
            "policy_gap_descriptions": {
                "capability_limit": "No queue or callback; can't transfer to busy lines",
                "business_rule": "Hours/availability restrictions block requests",
                "data_access": "Can't look up external system info",
                "auth_restriction": "Verification required but no path forward",
                "integration_missing": "No connection to required backend systems"
            } if with_policy_gap_desc else {},
            "major_failure_breakdowns": {
                "other": {
                    "patterns": [
                        {"pattern": "caller_hangup", "count": 28, "description": "Customer disconnected mid-call before resolution"},
                        {"pattern": "unclear_outcome", "count": 14, "description": "Call ended without clear resolution status"},
                        {"pattern": "system_timeout", "count": 10, "description": "Session expired due to inactivity"}
                    ]
                },
                "tech_issue": {
                    "patterns": [
                        {"pattern": "latency", "count": 25, "description": "Response delays causing customer frustration"},
                        {"pattern": "audio_glitch", "count": 12, "description": "Audio quality issues disrupted conversation"},
                        {"pattern": "infinite_loop", "count": 10, "description": "Agent stuck repeating same response"}
                    ]
                }
            } if with_major_breakdowns else {},
            "customer_ask_clusters": []
        }
    }


def test_key_metrics_has_context_column():
    """Key Metrics table should have 4 columns including Context."""
    report = create_mock_report()
    md = render_markdown(report)

    # Check header
    assert "| Metric | Value | Assessment | Context |" in md, \
        "Key Metrics table should have Context column header"
    assert "|--------|-------|------------|---------|" in md, \
        "Key Metrics table should have proper separator"

    # Check that context appears inline with values
    assert "| Success Rate | 39.2% |" in md and "dead-end" in md.lower(), \
        "Success Rate row should have inline context about driver"


def test_failure_point_has_context_column():
    """Failure Point table should have 4 columns with inline descriptions."""
    report = create_mock_report()
    md = render_markdown(report)

    # Check header
    assert "| Failure Type | Count | % of Failures | Context |" in md, \
        "Failure Point table should have Context column header"
    assert "|--------------|-------|---------------|---------|" in md, \
        "Failure Point table should have proper separator"

    # Check that descriptions are inline, not after table
    lines = md.split('\n')
    for i, line in enumerate(lines):
        if "| policy_gap |" in line:
            assert "Dead-end" in line or "escalation" in line.lower(), \
                f"policy_gap row should have inline description: {line}"
            break


def test_policy_gap_has_context_column():
    """Policy Gap table should have 4 columns with inline descriptions."""
    report = create_mock_report()
    md = render_markdown(report)

    # Check header
    assert "| Category | Count | % of Gaps | Context |" in md, \
        "Policy Gap table should have Context column header"
    assert "|----------|-------|-----------|---------|" in md, \
        "Policy Gap table should have proper separator"

    # Check inline descriptions
    lines = md.split('\n')
    for line in lines:
        if "| capability_limit |" in line:
            assert "queue" in line.lower() or "callback" in line.lower(), \
                f"capability_limit row should have inline description: {line}"
            break


def test_major_failure_breakdowns_render():
    """Major failure types (≥5%) should have sub-breakdowns."""
    report = create_mock_report()
    md = render_markdown(report)

    # 'other' has 52 out of 304 total (17.1%) - should have breakdown
    assert "#### Other Breakdown" in md, \
        "Other failure type (17.1%) should have sub-breakdown"

    # tech_issue has 47 out of 304 (15.5%) - should have breakdown
    assert "#### Tech Issue Breakdown" in md, \
        "Tech Issue failure type (15.5%) should have sub-breakdown"

    # Check breakdown table structure
    assert "| Pattern | Count | Description |" in md, \
        "Sub-breakdown table should have proper headers"

    # Check specific patterns are rendered
    assert "caller_hangup" in md, \
        "caller_hangup pattern should appear in Other breakdown"
    assert "latency" in md, \
        "latency pattern should appear in Tech Issue breakdown"


def test_minor_failure_types_no_breakdown():
    """Failure types with <5% should not have sub-breakdowns."""
    report = create_mock_report()
    md = render_markdown(report)

    # nlu_miss has 17 out of 304 (5.6%) - should have breakdown
    # wrong_action has 8 out of 304 (2.6%) - should NOT have breakdown
    assert "#### Wrong Action Breakdown" not in md, \
        "wrong_action (2.6%) should not have sub-breakdown"

    # customer_confusion has 5 out of 304 (1.6%) - should NOT have breakdown
    assert "#### Customer Confusion Breakdown" not in md, \
        "customer_confusion (1.6%) should not have sub-breakdown"


def test_graceful_degradation_no_descriptions():
    """Tables should render correctly even without descriptions."""
    report = create_mock_report(
        with_key_metrics_desc=False,
        with_failure_type_desc=False,
        with_policy_gap_desc=False,
        with_major_breakdowns=False
    )
    md = render_markdown(report)

    # Tables should still have 4 columns, just empty Context
    assert "| Metric | Value | Assessment | Context |" in md
    assert "| Failure Type | Count | % of Failures | Context |" in md
    assert "| Category | Count | % of Gaps | Context |" in md

    # Should not have sub-breakdowns
    assert "#### Other Breakdown" not in md
    assert "#### Tech Issue Breakdown" not in md


def test_version_footer():
    """Report footer should show v3.4."""
    report = create_mock_report()
    md = render_markdown(report)

    assert "v3.4" in md, "Report should indicate v3.4 version"


def run_all_tests():
    """Run all v3.4 feature tests."""
    tests = [
        test_key_metrics_has_context_column,
        test_failure_point_has_context_column,
        test_policy_gap_has_context_column,
        test_major_failure_breakdowns_render,
        test_minor_failure_types_no_breakdown,
        test_graceful_degradation_no_descriptions,
        test_version_footer,
    ]

    passed = 0
    failed = 0
    errors = []

    for test in tests:
        try:
            test()
            print(f"  [PASS] {test.__name__}")
            passed += 1
        except AssertionError as e:
            print(f"  [FAIL] {test.__name__}: {e}")
            failed += 1
            errors.append((test.__name__, str(e)))
        except Exception as e:
            print(f"  [ERROR] {test.__name__}: {e}")
            failed += 1
            errors.append((test.__name__, str(e)))

    print(f"\n{'='*60}")
    print(f"v3.4 Feature Tests: {passed} passed, {failed} failed")
    print(f"{'='*60}")

    if errors:
        print("\nFailures:")
        for name, error in errors:
            print(f"  - {name}: {error}")

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
