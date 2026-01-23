#!/usr/bin/env python3
"""
Test Suite for v4.0 Features: Intent + Sentiment Analysis

Tests:
1. Schema has new intent fields (intent, intent_context, secondary_intent)
2. Schema has sentiment fields (sentiment_start, sentiment_end)
3. Schema has unified disposition field (replaces outcome + call_disposition)
4. Prompt has intent extraction guidelines
5. Prompt has sentiment tracking guidelines
6. compute_metrics handles v4.0 schema
7. compute_metrics backwards compatible with v3.x
8. extract_nl_fields extracts v4.0 data
9. generate_insights includes intent/sentiment analysis
10. render_report displays intent/sentiment sections
"""

import json
import sys
from pathlib import Path

# Add tools directory to path
sys.path.insert(0, str(Path(__file__).parent))


def test_v40_intent_fields_in_schema():
    """Test v4.0 schema has intent fields."""
    print("\n=== Test: v4.0 Intent Fields in Schema ===")

    from analyze_transcript import ANALYSIS_SCHEMA

    # Check for intent field
    assert '"intent":' in ANALYSIS_SCHEMA, "Schema missing intent field"
    print("  intent field present in schema")

    # Check for intent_context field
    assert '"intent_context":' in ANALYSIS_SCHEMA, "Schema missing intent_context field"
    print("  intent_context field present in schema")

    # Check for secondary_intent field
    assert '"secondary_intent":' in ANALYSIS_SCHEMA, "Schema missing secondary_intent field"
    print("  secondary_intent field present in schema")

    print("  PASS: Intent fields test passed")
    return True


def test_v40_sentiment_fields_in_schema():
    """Test v4.0 schema has sentiment fields."""
    print("\n=== Test: v4.0 Sentiment Fields in Schema ===")

    from analyze_transcript import ANALYSIS_SCHEMA

    # Check for sentiment_start field
    assert '"sentiment_start":' in ANALYSIS_SCHEMA, "Schema missing sentiment_start field"
    print("  sentiment_start field present in schema")

    # Check for sentiment_end field
    assert '"sentiment_end":' in ANALYSIS_SCHEMA, "Schema missing sentiment_end field"
    print("  sentiment_end field present in schema")

    # Check for valid enum values
    assert "'positive'" in ANALYSIS_SCHEMA, "Schema missing positive sentiment enum"
    assert "'neutral'" in ANALYSIS_SCHEMA, "Schema missing neutral sentiment enum"
    assert "'frustrated'" in ANALYSIS_SCHEMA, "Schema missing frustrated sentiment enum"
    print("  Sentiment enum values present")

    print("  PASS: Sentiment fields test passed")
    return True


def test_v40_disposition_field():
    """Test v4.0 uses unified disposition field."""
    print("\n=== Test: v4.0 Unified Disposition Field ===")

    from analyze_transcript import ANALYSIS_SCHEMA

    # Check for disposition field
    assert '"disposition":' in ANALYSIS_SCHEMA, "Schema missing disposition field"
    print("  disposition field present in schema")

    # Check for valid enum values
    assert "'pre_intent'" in ANALYSIS_SCHEMA, "Schema missing pre_intent enum"
    assert "'in_scope_success'" in ANALYSIS_SCHEMA, "Schema missing in_scope_success enum"
    assert "'in_scope_partial'" in ANALYSIS_SCHEMA, "Schema missing in_scope_partial enum"
    assert "'in_scope_failed'" in ANALYSIS_SCHEMA, "Schema missing in_scope_failed enum"
    assert "'out_of_scope_handled'" in ANALYSIS_SCHEMA, "Schema missing out_of_scope_handled enum"
    assert "'escalated'" in ANALYSIS_SCHEMA, "Schema missing escalated enum"
    print("  All disposition enum values present")

    print("  PASS: Disposition field test passed")
    return True


def test_v40_renamed_fields():
    """Test v4.0 schema uses renamed fields."""
    print("\n=== Test: v4.0 Renamed Fields ===")

    from analyze_transcript import ANALYSIS_SCHEMA

    # Check for renamed fields (v4.0 names)
    assert '"effectiveness":' in ANALYSIS_SCHEMA, "Schema missing effectiveness field"
    print("  effectiveness field present (was agent_effectiveness)")

    assert '"quality":' in ANALYSIS_SCHEMA, "Schema missing quality field"
    print("  quality field present (was conversation_quality)")

    assert '"effort":' in ANALYSIS_SCHEMA, "Schema missing effort field"
    print("  effort field present (was customer_effort)")

    assert '"verbatim":' in ANALYSIS_SCHEMA, "Schema missing verbatim field"
    print("  verbatim field present (was customer_verbatim)")

    assert '"coaching":' in ANALYSIS_SCHEMA, "Schema missing coaching field"
    print("  coaching field present (was agent_miss_detail)")

    assert '"resolution":' in ANALYSIS_SCHEMA, "Schema missing resolution field"
    print("  resolution field present (was resolution_type)")

    assert '"steps":' in ANALYSIS_SCHEMA, "Schema missing steps field"
    print("  steps field present (was resolution_steps)")

    print("  PASS: Renamed fields test passed")
    return True


def test_v40_flattened_friction():
    """Test v4.0 schema has flattened friction structure."""
    print("\n=== Test: v4.0 Flattened Friction Structure ===")

    from analyze_transcript import ANALYSIS_SCHEMA

    # Check for top-level turns (was friction.turns)
    assert '"turns":' in ANALYSIS_SCHEMA, "Schema missing turns field"
    print("  turns field at top level")

    # Check for top-level derailed_at (was friction.derailed_at)
    assert '"derailed_at":' in ANALYSIS_SCHEMA, "Schema missing derailed_at field"
    print("  derailed_at field at top level")

    # Check for top-level clarifications (was friction.clarifications)
    assert '"clarifications":' in ANALYSIS_SCHEMA, "Schema missing clarifications field"
    print("  clarifications field at top level")

    # Check for top-level loops (was friction.loops)
    assert '"loops":' in ANALYSIS_SCHEMA, "Schema missing loops field"
    print("  loops field at top level")

    print("  PASS: Flattened friction test passed")
    return True


def test_v40_intent_guidelines_in_prompt():
    """Test v4.0 prompt has intent extraction guidelines."""
    print("\n=== Test: v4.0 Intent Guidelines in Prompt ===")

    from analyze_transcript import SYSTEM_PROMPT

    # Check for intent extraction section
    assert "INTENT" in SYSTEM_PROMPT, "Missing intent section in prompt"
    print("  Intent section present in prompt")

    # Check for intent_context guidance
    assert "intent_context" in SYSTEM_PROMPT, "Missing intent_context guidance"
    print("  intent_context guidance present")

    # Check for normalization guidance
    assert "normalize" in SYSTEM_PROMPT.lower() or "normalized" in SYSTEM_PROMPT.lower(), \
        "Missing intent normalization guidance"
    print("  Intent normalization guidance present")

    print("  PASS: Intent guidelines test passed")
    return True


def test_v40_sentiment_guidelines_in_prompt():
    """Test v4.0 prompt has sentiment tracking guidelines."""
    print("\n=== Test: v4.0 Sentiment Guidelines in Prompt ===")

    from analyze_transcript import SYSTEM_PROMPT

    # Check for sentiment tracking section
    assert "sentiment" in SYSTEM_PROMPT.lower(), "Missing sentiment section in prompt"
    print("  Sentiment section present in prompt")

    # Check for sentiment_start guidance
    assert "sentiment_start" in SYSTEM_PROMPT, "Missing sentiment_start guidance"
    print("  sentiment_start guidance present")

    # Check for sentiment_end guidance
    assert "sentiment_end" in SYSTEM_PROMPT, "Missing sentiment_end guidance"
    print("  sentiment_end guidance present")

    print("  PASS: Sentiment guidelines test passed")
    return True


def test_v40_compute_metrics_compat():
    """Test compute_metrics handles both v4.0 and v3.x schemas."""
    print("\n=== Test: v4.0 compute_metrics Compatibility ===")

    from compute_metrics import (
        get_disposition,
        get_quality_scores,
        get_failure_info,
        compute_intent_stats,
        compute_sentiment_stats,
    )

    # Test v4.0 analysis
    v4_analysis = {
        "disposition": "in_scope_success",
        "effectiveness": 4,
        "quality": 5,
        "effort": 2,
        "failure_type": "none",
        "failure_detail": None,
        "failure_recoverable": None,
        "failure_critical": False,
        "intent": "Check balance",
        "intent_context": "Invoice not received",
        "secondary_intent": "Make payment",
        "sentiment_start": "neutral",
        "sentiment_end": "satisfied",
    }

    # Test v3.x analysis
    v3_analysis = {
        "call_disposition": "in_scope_success",
        "agent_effectiveness": 4,
        "conversation_quality": 5,
        "customer_effort": 2,
        "failure_point": "none",
        "failure_description": None,
        "was_recoverable": None,
        "critical_failure": False,
    }

    # Test get_disposition
    assert get_disposition(v4_analysis) == "in_scope_success", "v4.0 disposition failed"
    assert get_disposition(v3_analysis) == "in_scope_success", "v3.x disposition failed"
    print("  get_disposition works for both schemas")

    # Test get_quality_scores
    v4_scores = get_quality_scores(v4_analysis)
    v3_scores = get_quality_scores(v3_analysis)
    assert v4_scores == (4, 5, 2), f"v4.0 quality scores wrong: {v4_scores}"
    assert v3_scores == (4, 5, 2), f"v3.x quality scores wrong: {v3_scores}"
    print("  get_quality_scores works for both schemas")

    # Test get_failure_info
    v4_failure = get_failure_info(v4_analysis)
    v3_failure = get_failure_info(v3_analysis)
    assert v4_failure[0] == "none", f"v4.0 failure_type wrong: {v4_failure[0]}"
    assert v3_failure[0] == "none", f"v3.x failure_type wrong: {v3_failure[0]}"
    print("  get_failure_info works for both schemas")

    # Test intent stats
    intent_stats = compute_intent_stats([v4_analysis])
    assert intent_stats.get("total_with_intent") == 1, "Intent stats wrong"
    # Note: compute_intent_stats lowercases intents for grouping
    assert intent_stats.get("top_intents", [{}])[0].get("intent") == "check balance"
    print("  compute_intent_stats works")

    # Test sentiment stats
    sentiment_stats = compute_sentiment_stats([v4_analysis])
    assert sentiment_stats.get("total_with_sentiment") == 1, "Sentiment stats wrong"
    print("  compute_sentiment_stats works")

    print("  PASS: compute_metrics compatibility test passed")
    return True


def test_v40_extract_nl_fields():
    """Test extract_nl_fields extracts v4.0 data."""
    print("\n=== Test: v4.0 extract_nl_fields ===")

    from extract_nl_fields import extract_nl_summary

    mock_analyses = [
        {
            "call_id": "test-v4-1",
            "disposition": "in_scope_success",
            "intent": "Log into Clubhouse",
            "intent_context": "Registration link not arriving",
            "secondary_intent": "Pay maintenance fees",
            "sentiment_start": "neutral",
            "sentiment_end": "satisfied",
            "summary": "Customer logged into Clubhouse successfully",
            "verbatim": "Great, thank you!",
        },
        {
            "call_id": "test-v4-2",
            "disposition": "in_scope_failed",
            "intent": "Make payment",
            "intent_context": "Past due notice received",
            "sentiment_start": "frustrated",
            "sentiment_end": "angry",
            "summary": "Payment could not be processed",
            "verbatim": "This is ridiculous!",
            "coaching": "Should have offered callback option earlier",
        },
    ]

    result = extract_nl_summary(mock_analyses)

    # Check intent_data is extracted
    assert "intent_data" in result, "Missing intent_data"
    intent_data = result["intent_data"]
    assert len(intent_data) == 2, f"Wrong intent_data count: {len(intent_data)}"
    assert intent_data[0]["intent"] == "Log into Clubhouse", "Wrong intent"
    # Note: extract_nl_fields stores intent_context as "context" for brevity
    assert intent_data[0]["context"] == "Registration link not arriving", "Wrong intent context"
    print("  intent_data extracted correctly")

    # Check sentiment_data is extracted
    assert "sentiment_data" in result, "Missing sentiment_data"
    sentiment_data = result["sentiment_data"]
    assert len(sentiment_data) == 2, f"Wrong sentiment_data count: {len(sentiment_data)}"
    # Note: extract_nl_fields stores as "start" and "end" for brevity
    assert sentiment_data[0]["start"] == "neutral", "Wrong sentiment start"
    assert sentiment_data[0]["end"] == "satisfied", "Wrong sentiment end"
    print("  sentiment_data extracted correctly")

    # Check disposition_summary uses v4.0 disposition field
    disp_summary = result.get("disposition_summary", {})
    assert "in_scope_success" in disp_summary, "Missing in_scope_success in disposition_summary"
    assert "in_scope_failed" in disp_summary, "Missing in_scope_failed in disposition_summary"
    print("  disposition_summary uses v4.0 disposition field")

    print("  PASS: extract_nl_fields test passed")
    return True


def test_v40_generate_insights_sections():
    """Test generate_insights output schema includes v4.0 sections."""
    print("\n=== Test: v4.0 generate_insights Sections ===")

    from generate_insights import INSIGHTS_SYSTEM_PROMPT

    # Check for intent_analysis section
    assert '"intent_analysis"' in INSIGHTS_SYSTEM_PROMPT, "Missing intent_analysis in output schema"
    print("  intent_analysis section in output schema")

    # Check for sentiment_analysis section
    assert '"sentiment_analysis"' in INSIGHTS_SYSTEM_PROMPT, "Missing sentiment_analysis in output schema"
    print("  sentiment_analysis section in output schema")

    # Check for intent analysis guidelines (guideline 22)
    assert "Intent analysis" in INSIGHTS_SYSTEM_PROMPT or "intent analysis" in INSIGHTS_SYSTEM_PROMPT, \
        "Missing intent analysis guideline"
    print("  Intent analysis guideline present")

    # Check for sentiment analysis guidelines (guideline 23)
    assert "Sentiment analysis" in INSIGHTS_SYSTEM_PROMPT or "sentiment analysis" in INSIGHTS_SYSTEM_PROMPT, \
        "Missing sentiment analysis guideline"
    print("  Sentiment analysis guideline present")

    print("  PASS: generate_insights sections test passed")
    return True


def test_v40_render_report_sections():
    """Test render_report includes v4.0 sections."""
    print("\n=== Test: v4.0 render_report Sections ===")

    from render_report import render_markdown

    # Mock report with v4.0 insights
    mock_report = {
        "deterministic_metrics": {
            "metadata": {"total_calls_analyzed": 100, "report_generated": "2025-01-21T12:00:00"},
            "key_rates": {"success_rate": 0.75, "escalation_rate": 0.10, "failure_rate": 0.15},
            "quality_scores": {
                "agent_effectiveness": {"mean": 4.0, "median": 4, "std": 0.5, "n": 100},
                "conversation_quality": {"mean": 4.2, "median": 4, "std": 0.4, "n": 100},
                "customer_effort": {"mean": 2.5, "median": 2, "std": 0.8, "n": 100},
            },
        },
        "llm_insights": {
            "executive_summary": "Test summary",
            "intent_analysis": {
                "narrative": "Customers primarily call to check balances and make payments.",
                "top_clusters": [
                    {"intent_cluster": "Check balance", "count": 30, "pct": "30%", "success_rate": "90%", "insight": "High success rate"}
                ],
                "context_patterns": [
                    {"pattern": "Invoice not received", "frequency": "15%", "recommendation": "Improve email delivery"}
                ],
                "unmet_needs": [
                    {"need": "Direct payment processing", "frequency": "10%", "recommendation": "Add payment capability"}
                ]
            },
            "sentiment_analysis": {
                "narrative": "Most customers remain neutral or improve sentiment.",
                "journey_patterns": [
                    {"pattern": "neutral -> satisfied", "frequency": "40%", "drivers": "Quick resolution", "outcome_correlation": "High success"}
                ],
                "improvement_drivers": [
                    {"factor": "Quick resolution", "frequency": "35 calls", "recommendation": "Prioritize speed"}
                ],
                "degradation_drivers": [
                    {"factor": "Repeated questions", "frequency": "15 calls", "recommendation": "Improve ASR"}
                ],
                "health_metrics": {"assessment": "healthy", "improvement_rate": "40%", "degradation_rate": "10%"}
            }
        }
    }

    markdown = render_markdown(mock_report)

    # Check for Intent Analysis section
    assert "## Intent Analysis" in markdown, "Missing Intent Analysis section"
    print("  Intent Analysis section present in report")

    # Check for intent clusters
    assert "Top Intent Clusters" in markdown, "Missing Top Intent Clusters subsection"
    print("  Top Intent Clusters subsection present")

    # Check for Sentiment Analysis section
    assert "## Sentiment Analysis" in markdown, "Missing Sentiment Analysis section"
    print("  Sentiment Analysis section present in report")

    # Check for sentiment health
    assert "Sentiment Health" in markdown, "Missing Sentiment Health in report"
    print("  Sentiment Health metrics present")

    # Check for journey patterns
    assert "Emotional Journey Patterns" in markdown, "Missing Emotional Journey Patterns"
    print("  Emotional Journey Patterns present")

    print("  PASS: render_report sections test passed")
    return True


def test_v40_version_references():
    """Test that v4.0 is referenced in module docstrings."""
    print("\n=== Test: v4.0 Version References ===")

    # Check analyze_transcript
    from analyze_transcript import __doc__ as analyze_doc
    assert "v4.0" in analyze_doc, "analyze_transcript missing v4.0 reference"
    print("  analyze_transcript mentions v4.0")

    # Check compute_metrics
    from compute_metrics import __doc__ as compute_doc
    assert "v4.0" in compute_doc, "compute_metrics missing v4.0 reference"
    print("  compute_metrics mentions v4.0")

    # Check extract_nl_fields
    from extract_nl_fields import __doc__ as extract_doc
    assert "v4.0" in extract_doc, "extract_nl_fields missing v4.0 reference"
    print("  extract_nl_fields mentions v4.0")

    # Check generate_insights
    from generate_insights import __doc__ as insights_doc
    assert "v4.0" in insights_doc, "generate_insights missing v4.0 reference"
    print("  generate_insights mentions v4.0")

    # Check render_report
    from render_report import __doc__ as render_doc
    assert "v4.0" in render_doc, "render_report missing v4.0 reference"
    print("  render_report mentions v4.0")

    print("  PASS: Version references test passed")
    return True


def test_v40_ask_py_compat():
    """Test ask.py handles both v4.0 and v3.x schemas."""
    print("\n=== Test: v4.0 ask.py Compatibility ===")

    from ask import format_call_for_prompt, QA_SYSTEM_PROMPT

    # Check system prompt has v4.0 field documentation
    assert "disposition" in QA_SYSTEM_PROMPT, "Missing disposition in system prompt"
    assert "intent" in QA_SYSTEM_PROMPT, "Missing intent in system prompt"
    assert "sentiment_start" in QA_SYSTEM_PROMPT, "Missing sentiment_start in system prompt"
    print("  System prompt documents v4.0 fields")

    # Test v4.0 analysis
    v4_analysis = {
        "call_id": "test-v4-abcd",
        "disposition": "in_scope_success",
        "intent": "Check balance",
        "intent_context": "Invoice not received",
        "sentiment_start": "neutral",
        "sentiment_end": "satisfied",
        "summary": "Customer checked balance successfully",
        "verbatim": "Thank you so much!",
        "coaching": None,
        "turns": 8,
        "loops": [],
        "clarifications": [],
        "corrections": [],
    }

    # Test v3.x analysis
    v3_analysis = {
        "call_id": "test-v3-efgh",
        "outcome": "resolved",
        "call_disposition": "in_scope_success",
        "summary": "Customer checked balance successfully",
        "customer_verbatim": "Thank you so much!",
        "agent_miss_detail": None,
        "failure_point": "none",
        "failure_description": None,
        "friction": {
            "turns": 8,
            "loops": [],
            "clarifications": [],
            "corrections": [],
        }
    }

    # Format v4.0 analysis
    v4_result = format_call_for_prompt(v4_analysis)
    assert "test-v4" in v4_result, "v4.0 call_id not in output"
    assert "Disposition: in_scope_success" in v4_result, "v4.0 disposition not shown"
    assert "Intent: Check balance" in v4_result, "v4.0 intent not shown"
    assert "Invoice not received" in v4_result, "v4.0 intent_context not shown"
    assert "Sentiment: neutral → satisfied" in v4_result, "v4.0 sentiment not shown"
    assert "8 turns" in v4_result, "v4.0 turns not shown"
    print("  v4.0 analysis formatted correctly with intent/sentiment")

    # Format v3.x analysis
    v3_result = format_call_for_prompt(v3_analysis)
    assert "test-v3" in v3_result, "v3.x call_id not in output"
    assert "Outcome: resolved" in v3_result, "v3.x outcome not shown"
    assert "in_scope_success" in v3_result, "v3.x disposition not shown"
    assert "8 turns" in v3_result, "v3.x turns not shown"
    # v3.x doesn't have intent/sentiment, should not appear
    assert "Intent:" not in v3_result, "v3.x should not have intent"
    assert "Sentiment:" not in v3_result, "v3.x should not have sentiment"
    print("  v3.x analysis formatted correctly (no intent/sentiment)")

    print("  PASS: ask.py compatibility test passed")
    return True


def test_v40_backwards_compat_friction_parsing():
    """Test backwards compatibility for friction structure parsing."""
    print("\n=== Test: v4.0 Backwards Compat Friction Parsing ===")

    from compute_metrics import (
        parse_clarifications,
        parse_corrections,
        parse_loops,
        get_conversation_turns,
        get_turns_to_failure,
    )

    # v4.0 format (top-level arrays)
    v4_analysis = {
        "turns": 15,
        "derailed_at": 10,
        "clarifications": [{"turn": 3, "type": "name", "cause": "misheard", "note": "asked to spell"}],
        "corrections": [{"turn": 5, "severity": "minor", "note": "corrected name"}],
        "loops": [{"turns": [7, 9], "type": "info_retry", "subject": "phone", "note": "re-asked phone"}],
    }

    # v3.x format (nested under friction)
    v3_analysis = {
        "friction": {
            "turns": 15,
            "derailed_at": 10,
            "clarifications": [{"t": 3, "type": "name", "cause": "misheard", "ctx": "asked to spell"}],
            "corrections": [{"t": 5, "sev": "minor", "ctx": "corrected name"}],
            "loops": [{"t": [7, 9], "type": "info_retry", "subject": "phone", "ctx": "re-asked phone"}],
        }
    }

    # Test turns parsing
    assert get_conversation_turns(v4_analysis) == 15, "v4.0 turns parsing failed"
    assert get_conversation_turns(v3_analysis) == 15, "v3.x turns parsing failed"
    print("  get_conversation_turns works for both formats")

    # Test derailed_at parsing
    assert get_turns_to_failure(v4_analysis) == 10, "v4.0 derailed_at parsing failed"
    assert get_turns_to_failure(v3_analysis) == 10, "v3.x derailed_at parsing failed"
    print("  get_turns_to_failure works for both formats")

    # Test clarifications parsing
    # parse_clarifications returns list of (turn, type, cause, ctx) tuples
    v4_clars = parse_clarifications(v4_analysis)
    v3_clars = parse_clarifications(v3_analysis)
    assert len(v4_clars) == 1, "v4.0 clarifications parsing failed"
    assert len(v3_clars) == 1, "v3.x clarifications parsing failed"
    assert v4_clars[0][0] == 3, "v4.0 clarification turn wrong"  # tuple[0] = turn
    assert v3_clars[0][0] == 3, "v3.x clarification turn wrong"  # tuple[0] = turn
    print("  parse_clarifications works for both formats")

    # Test corrections parsing
    # parse_corrections returns list of (turn, severity, what, ctx) tuples
    v4_corrs = parse_corrections(v4_analysis)
    v3_corrs = parse_corrections(v3_analysis)
    assert len(v4_corrs) == 1, "v4.0 corrections parsing failed"
    assert len(v3_corrs) == 1, "v3.x corrections parsing failed"
    assert v4_corrs[0][1] == "minor", "v4.0 correction severity wrong"  # tuple[1] = severity
    assert v3_corrs[0][1] == "minor", "v3.x correction severity wrong"  # tuple[1] = severity
    print("  parse_corrections works for both formats")

    # Test loops parsing
    # parse_loops returns list of (turns_list, type, subject, ctx) tuples
    v4_loops = parse_loops(v4_analysis)
    v3_loops = parse_loops(v3_analysis)
    assert len(v4_loops) == 1, "v4.0 loops parsing failed"
    assert len(v3_loops) == 1, "v3.x loops parsing failed"
    assert v4_loops[0][0] == [7, 9], f"v4.0 loop turns wrong: {v4_loops[0][0]}"  # tuple[0] = turns_list
    assert v3_loops[0][0] == [7, 9], f"v3.x loop turns wrong: {v3_loops[0][0]}"  # tuple[0] = turns_list
    print("  parse_loops works for both formats")

    print("  PASS: Backwards compat friction parsing test passed")
    return True


def run_all_tests():
    """Run all v4.0 feature tests."""
    print("=" * 60)
    print("v4.0 Feature Tests: Intent + Sentiment Analysis")
    print("=" * 60)

    tests = [
        test_v40_intent_fields_in_schema,
        test_v40_sentiment_fields_in_schema,
        test_v40_disposition_field,
        test_v40_renamed_fields,
        test_v40_flattened_friction,
        test_v40_intent_guidelines_in_prompt,
        test_v40_sentiment_guidelines_in_prompt,
        test_v40_compute_metrics_compat,
        test_v40_extract_nl_fields,
        test_v40_generate_insights_sections,
        test_v40_render_report_sections,
        test_v40_version_references,
        test_v40_ask_py_compat,
        test_v40_backwards_compat_friction_parsing,
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
