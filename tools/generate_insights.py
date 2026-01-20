#!/usr/bin/env python3
"""
LLM Insights Generator for Vacatia AI Voice Agent Analytics (v3.9.1)

Generates Section B: LLM-powered insights by passing Section A metrics
and the condensed NL summary (from extract_nl_fields.py) to Gemini.

v3.9.1 additions:
- loop_subject_clusters: Semantic clustering of loop subjects by type
- Loop subject analysis narrative and insights
- Subject distribution breakdown per loop type
- Custom questions: --questions flag to answer user-provided analytical questions
- custom_analysis section in report with question/answer pairs

v3.9 additions:
- disposition_analysis: Insights on call disposition patterns
- Funnel analysis recommendations based on disposition breakdown
- Actionable insights by disposition type (in_scope_partial → confirmation prompts, etc.)

v3.8.5 additions:
- Backwards-compatible with both v3.8.5 (compact friction) and v3.8 formats
- Loop events now include turn numbers when available
- All friction data extracted via extract_nl_fields.py helper functions

v3.8 additions:
- loop_type_analysis: Distribution and impact of agent_loops by type
- Loop patterns: info_retry, intent_retry, deflection, comprehension, action_retry
- Intent retry rate: Specifically tracks re-asking for intent after identity check
- Deflection rate: Tracks generic questions masking inability to help

v3.7 additions:
- clarification_cause_analysis: Distribution and impact of cause types (customer_refused, agent_misheard, etc.)
- correction_severity_analysis: Distribution and impact of severity levels (minor, moderate, major)
- Context sentences for nuanced friction pattern discovery

v3.6 additions:
- conversation_quality_analysis: Friction hotspots, efficiency insights from turn/clarification/correction data
- Narrative on conversation friction patterns
- Recommendations for reducing clarification frequency

v3.5 additions:
- training_analysis: Narrative + priorities + cross-correlations for training gaps
- emergent_patterns: LLM-discovered patterns not fitting existing categories
- secondary_intents_analysis: Clustering of additional customer intents

v3.4 additions:
- Inline descriptions for ALL table rows (key metrics, failure types, policy gaps)
- key_metrics_descriptions: WHY-focused context for each metric (drivers/causes)
- failure_type_descriptions: Concise inline context for each failure type
- policy_gap_descriptions: Concise inline context for each policy gap category
- major_failure_breakdowns: Sub-breakdowns for failure types with ≥5% of failures

v3.3 additions:
- Call ID references throughout (failure entries, verbatims, policy gaps)
- supporting_call_ids in actionable_recommendations
- customer_ask_clusters: Semantic grouping of similar customer asks

Primary input: nl_summary_v3_{timestamp}.json (from extract_nl_fields.py)
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Load .env file if present
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types


INSIGHTS_SYSTEM_PROMPT = """You are a senior call center analytics consultant. Your job is to analyze voice agent performance metrics and natural language data to provide executive-ready insights and actionable recommendations.

You will receive:
1. Section A: Deterministic metrics (calculated via Python - factual, auditable)
2. Aggregated qualitative data: failure descriptions, customer quotes, agent misses (with call_ids for traceability)
3. Training opportunities grouped by type with failure context (v3.5)
4. Secondary customer intents for clustering (v3.5)
5. Conversation quality events: clarifications, corrections, turn outliers (v3.6)
6. Clarification cause distribution and context sentences (v3.7)
7. Correction severity distribution and context sentences (v3.7)
8. Agent loops with type, context, and turns for friction analysis (v3.8.5)

Your task: Generate Section B insights that synthesize the data into strategic recommendations.

## Output Format (v3.5)

Return ONLY valid JSON matching this structure:
{
  "executive_summary": "2-3 sentence high-level takeaway for leadership",

  "root_cause_analysis": {
    "primary_driver": "string (single biggest issue causing failures)",
    "contributing_factors": ["string", "string"],
    "evidence": "string (supporting data points from metrics)"
  },

  "actionable_recommendations": [
    {
      "priority": "P0 | P1 | P2",
      "category": "capability | training | prompt | process",
      "recommendation": "string (specific action to take)",
      "expected_impact": "string (e.g., 'Could resolve 18% of failures')",
      "evidence": "string (why this matters)",
      "supporting_call_ids": ["call_id1", "call_id2"]
    }
  ],

  "trend_narratives": {
    "failure_patterns": "string (narrative about what's failing and why)",
    "customer_experience": "string (narrative about customer friction points)",
    "agent_performance": "string (narrative about agent behavior patterns)"
  },

  "verbatim_highlights": {
    "most_frustrated": "string (worst customer quote showing pain)",
    "most_common_ask": "string (recurring unmet customer need)",
    "biggest_miss": "string (most impactful missed agent opportunity)"
  },

  "key_metrics_descriptions": {
    "success_rate": "WHY this rate - main driver/cause (max 15 words)",
    "escalation_rate": "WHY this rate - main driver/cause (max 15 words)",
    "failure_rate": "WHY this rate - main driver/cause (max 15 words)",
    "customer_effort": "WHY this score - main driver/cause (max 15 words)"
  },

  "failure_type_descriptions": {
    "<failure_type>": "concise run-specific description (max 15 words)"
  },

  "policy_gap_descriptions": {
    "<category>": "concise run-specific description (max 15 words)"
  },

  "major_failure_breakdowns": {
    "<failure_type>": {
      "patterns": [
        {"pattern": "pattern name", "count": 10, "description": "run-specific context (max 15 words)"}
      ]
    }
  },

  "customer_ask_clusters": [
    {
      "canonical_label": "Human-readable cluster name (e.g., 'Request human agent transfer')",
      "member_asks": ["speak to a representative", "talk to a live agent", "..."],
      "total_count": 10,
      "example_call_ids": ["call_id1", "call_id2", "call_id3"]
    }
  ],

  "training_analysis": {
    "narrative": "2-3 sentences synthesizing training gaps and root causes, explaining WHY these gaps exist and how they connect to failures",
    "top_priorities": [
      {
        "skill": "verification",
        "count": 666,
        "why": "WHY this skill gap exists (root cause analysis)",
        "action": "Specific training intervention or process change"
      }
    ],
    "cross_correlations": [
      {
        "pattern": "training_gap + failure_type (e.g., 'verification + auth_restriction')",
        "count": 145,
        "insight": "What this correlation reveals about systemic issues"
      }
    ]
  },

  "emergent_patterns": [
    {
      "name": "Descriptive pattern name",
      "frequency": "count or percentage string",
      "description": "What you observed in the data",
      "significance": "Why this matters for operations/CX",
      "example_call_ids": ["id1", "id2"]
    }
  ],

  "secondary_intents_analysis": {
    "narrative": "Summary of secondary customer needs beyond primary intent",
    "clusters": [
      {
        "cluster": "Cluster name (e.g., 'Exit/Sell Timeshare')",
        "count": 50,
        "implication": "What this reveals about customer needs or agent gaps"
      }
    ]
  },

  "conversation_quality_analysis": {
    "narrative": "2-3 sentences synthesizing conversation friction patterns: what causes most back-and-forth, which clarification types correlate with poor outcomes",

    "friction_hotspots": [
      {
        "pattern": "Pattern name (e.g., 'Name spelling requests')",
        "frequency": "18% of calls",
        "impact": "Effect on outcomes and customer experience",
        "recommendation": "Specific improvement suggestion"
      }
    ],

    "cause_analysis": {
      "narrative": "1-2 sentences explaining the cause distribution pattern",
      "insights": [
        {
          "cause": "customer_refused",
          "frequency": "15 events (30%)",
          "correlation": "How this cause type correlates with outcomes",
          "recommendation": "Specific action to reduce this cause type"
        }
      ]
    },

    "severity_analysis": {
      "narrative": "1-2 sentences explaining correction severity patterns",
      "insights": [
        {
          "severity": "major",
          "frequency": "8 events (20%)",
          "correlation": "How major corrections correlate with outcomes",
          "recommendation": "Specific action to reduce major corrections"
        }
      ]
    },

    "efficiency_insights": [
      "Insight about conversation length vs outcomes",
      "Insight about early friction predicting failure",
      "Insight about clarification patterns"
    ],

    "turn_analysis": {
      "long_call_patterns": "What's causing unusually long calls?",
      "short_call_patterns": "What characterizes quick resolutions or early abandonments?",
      "turns_to_failure_insight": "What do failed calls look like before they derail?"
    },

    "loop_type_analysis": {
      "narrative": "1-2 sentences explaining agent loop patterns and their impact",
      "insights": [
        {
          "type": "intent_retry",
          "frequency": "15 events (35%)",
          "pattern": "Common pattern observed with this loop type",
          "impact": "How this loop type affects call outcomes",
          "recommendation": "Specific action to reduce this loop type"
        }
      ],
      "intent_retry_rate": "% of calls with intent re-ask after identity check",
      "deflection_rate": "% of calls with deflection loops"
    }
  },

  "disposition_analysis": {
    "narrative": "2-3 sentences synthesizing the call disposition distribution and what it reveals about agent effectiveness",
    "actionable_insights": [
      {
        "disposition": "in_scope_partial | in_scope_failed | out_of_scope_abandoned | pre_intent",
        "frequency": "count and percentage string",
        "root_cause": "Why calls end up in this disposition",
        "recommendation": "Specific action to improve this metric"
      }
    ],
    "funnel_health": {
      "assessment": "healthy | needs_attention | critical",
      "explanation": "Brief explanation of overall funnel health",
      "priority_focus": "Which disposition category needs most attention and why"
    }
  },

  "loop_subject_clusters": {
    "narrative": "2-3 sentences synthesizing loop subject patterns and what they reveal about friction points",
    "by_loop_type": {
      "<loop_type>": {
        "top_subjects": [
          {
            "subject": "canonical_subject_name",
            "count": 10,
            "pct": "25%",
            "insight": "Why this subject is frequently looped (optional)"
          }
        ],
        "recommendation": "Specific action to reduce loops for this type"
      }
    },
    "high_impact_patterns": [
      {
        "loop_type": "info_retry | intent_retry | etc.",
        "subject": "name | phone | etc.",
        "frequency": "count or percentage",
        "impact": "How this pattern affects call outcomes",
        "recommendation": "Specific fix"
      }
    ]
  }
}

## Guidelines

1. **Be specific and actionable**: Avoid generic advice like "improve training". Instead: "Add verification bypass for returning customers to reduce 23% of auth failures"

2. **Ground in data**: Every recommendation should reference specific metrics. "Policy gaps account for 44% of failures, with capability_limit being the top category"

3. **Prioritize by impact**: P0 = critical/immediate (>20% of failures), P1 = high (10-20%), P2 = moderate (<10%)

4. **Use customer voice**: Include verbatim quotes that illustrate the problem - these are powerful for executive buy-in

5. **Connect dots**: Link failure patterns to customer experience to recommendations

6. **Be concise**: Executive summaries should be scannable in 30 seconds

7. **Include call_ids**: For each recommendation, include 2-5 supporting call_ids that exemplify the issue

8. **Semantic clustering - CRITICAL**: You will receive a list titled "ALL Customer Asks for Clustering" containing every customer_ask from policy gap failures. You MUST:
   - Group semantically similar asks (e.g., "speak to a representative", "Representative.", "talk to a live agent", "I need a real person" → all cluster as "Request human agent transfer")
   - Count EVERY item in the list - the total across all clusters should equal the number of asks provided
   - The `total_count` for each cluster should reflect actual occurrences, not pre-aggregated counts
   - Common clusters: "Request human agent", "Make payment by phone", "Make reservation", "RCI deposit/exchange", etc.

9. **CRITICAL - ALL descriptions required**: You MUST provide a description for EVERY row in failure_type_descriptions, policy_gap_descriptions, and key_metrics_descriptions. Do not skip any. If a failure type or category appears in the metrics, it MUST have a corresponding description.

10. **Key metrics descriptions - WHY, not WHAT**: For key_metrics_descriptions, focus on the PRIMARY DRIVER or ROOT CAUSE behind each metric value. Don't restate the number or compare to a threshold. Explain WHY.
   - BAD: "Below 50% target; majority fail"
   - GOOD: "Driven by dead-end escalation; verified customers can't reach humans"

11. **Sub-breakdowns for major failure types**: For each failure_type representing ≥5% of total failures (excluding policy_gap which already has a breakdown), analyze the failure descriptions and identify 2-5 common patterns. Group similar failures and provide run-specific context. Include in major_failure_breakdowns.

12. **Training analysis - CONNECT THE DOTS**: Analyze the "Training Opportunities" data to understand WHY training gaps exist:
    - What is the relationship between training gaps and failure types?
    - Which training gaps correlate with which policy gap categories?
    - What specific intervention (training, process, tooling) would address the root cause?
    - The narrative should explain the systemic issues, not just list counts.

13. **Cross-correlations**: Look for significant patterns spanning multiple dimensions:
    - training_opportunity + failure_point (e.g., "verification gaps correlate with auth_restriction failures")
    - training_opportunity + policy_gap category
    - Include the count and a meaningful insight for each correlation found

14. **Emergent patterns**: Actively identify patterns NOT covered by existing categories:
    - Unexpected correlations in the data
    - Themes in verbatims not captured by standard fields
    - Surprising observations worth flagging for investigation
    - Name each pattern clearly and explain its significance
    - Only include patterns with meaningful frequency (not one-offs)

15. **Secondary intents analysis**: Cluster the "Secondary Intents" data to understand:
    - What additional needs customers have beyond their primary intent
    - Which secondary needs are frequently unmet
    - What this reveals about customer journey gaps

16. **Conversation quality analysis (v3.6)**: Analyze the "Conversation Quality" data to understand:
    - **Friction hotspots**: Which clarification types cause most friction? Do name spelling requests correlate with escalation? Do phone confirmations lead to verification failures?
    - **Efficiency insights**: What distinguishes quick resolutions from drawn-out failures? Do calls with early corrections tend to fail?
    - **Turn analysis**: What patterns characterize long calls vs short calls? How many turns before a call typically derails?
    - **Correlation opportunities**: Connect clarification types to failure types. Do certain clarifications predict specific outcomes?
    - Generate 2-4 friction hotspots with specific improvement recommendations
    - Include 3-5 efficiency insights based on the turn and correction data

17. **Cause analysis (v3.7)**: Analyze clarification cause distribution:
    - Which causes (customer_refused, agent_misheard, customer_unclear, tech_issue) are most common?
    - Do certain causes correlate with call failure more than others?
    - What interventions could reduce the most impactful cause types?
    - Use the context sentences to identify patterns within each cause type

18. **Severity analysis (v3.7)**: Analyze correction severity distribution:
    - What's the breakdown between minor, moderate, and major corrections?
    - Do major corrections predict call failure?
    - What patterns exist in the context sentences for major corrections?
    - Recommend specific training or process changes to reduce major corrections

19. **Loop type analysis (v3.8)**: Analyze agent_loops by type:
    - **info_retry**: Agent re-asks for information already provided (e.g., name, phone number)
    - **intent_retry**: Agent re-asks for intent after customer already stated it (especially after identity check)
    - **deflection**: Generic questions while unable to help the primary request
    - **comprehension**: Agent couldn't hear/understand, asks to repeat
    - **action_retry**: Agent retries same action due to system failure
    - Calculate intent_retry_rate: % of calls where agent asks intent again after verification
    - Calculate deflection_rate: % of calls with deflection loops (signals capability gaps)
    - Focus on actionable recommendations: which loop types could be reduced by prompt improvements vs capability additions

20. **Disposition analysis (v3.9)**: Analyze call_disposition patterns to understand call funnel:
    - **pre_intent**: Calls ending before customer states request (possible IVR/routing issues)
    - **out_of_scope_handled**: Out-of-scope requests gracefully redirected (measure of deflection quality)
    - **out_of_scope_abandoned**: Out-of-scope where customer gave up (capability expansion opportunities)
    - **in_scope_success**: Request completed with explicit satisfaction (gold standard)
    - **in_scope_partial**: Completed but no confirmation (opportunity for confirmation prompts)
    - **in_scope_failed**: In-scope requests that couldn't be completed (training/capability gaps)
    - Provide actionable_insights for each disposition with high count
    - Assess overall funnel_health based on distribution
    - Focus on actionable recommendations: what would move in_scope_partial → in_scope_success?

21. **Loop subject clustering (v3.9.1)**: Analyze loop subject patterns for granular friction analysis:
    - Group subjects semantically within each loop type (e.g., "name", "customer_name", "spelling" → "name")
    - Calculate subject distribution per loop type (e.g., info_retry: name 45%, phone 30%, other 25%)
    - Identify high-impact patterns: which (loop_type, subject) combinations correlate with failed outcomes?
    - Provide actionable recommendations: e.g., "45% of info_retry loops involve name re-asking, suggesting ASR/name recognition as improvement area"
    - The narrative should explain WHAT subjects are causing friction and WHY

Return ONLY the JSON object, no markdown code blocks or additional text.
"""


def get_genai_client() -> genai.Client:
    """Get configured Google GenAI client."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
    return genai.Client(api_key=api_key)


# For backwards compatibility
def configure_genai():
    """Configure the Google Generative AI client (deprecated, use get_genai_client)."""
    get_genai_client()  # Just validate the key exists


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response."""
    text = text.strip()

    # Remove markdown code blocks if present
    if text.startswith('```'):
        first_newline = text.find('\n')
        if first_newline != -1:
            text = text[first_newline + 1:]
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Find JSON object in text
    brace_start = text.find('{')
    if brace_start != -1:
        depth = 0
        for i, char in enumerate(text[brace_start:], brace_start):
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[brace_start:i+1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Could not parse JSON from: {text[:500]}...")


def load_questions(questions_path: Path) -> list[str]:
    """Load custom questions from a file (one question per line).

    v3.9.1: Supports questions.txt or questions.json format.
    """
    if not questions_path.exists():
        return []

    content = questions_path.read_text(encoding='utf-8').strip()

    # Try JSON format first
    if questions_path.suffix == '.json':
        try:
            data = json.loads(content)
            if isinstance(data, list):
                return [q.strip() for q in data if q.strip()]
            elif isinstance(data, dict) and 'questions' in data:
                return [q.strip() for q in data['questions'] if q.strip()]
        except json.JSONDecodeError:
            pass

    # Default: one question per line
    questions = []
    for line in content.split('\n'):
        line = line.strip()
        # Skip empty lines and comments
        if line and not line.startswith('#'):
            questions.append(line)

    return questions


CUSTOM_QUESTIONS_SYSTEM_PROMPT = """You are a senior call center analytics consultant. You have access to comprehensive data about voice agent performance including metrics, failure patterns, customer verbatims, and conversation quality data.

Your task is to answer custom analytical questions based on the data provided. Each answer should:
1. Be specific and grounded in the data
2. Include relevant metrics or evidence when available
3. Be concise but complete (2-5 sentences per answer)
4. Acknowledge limitations if the data doesn't fully address the question

Return your answers as a JSON object with this structure:
{
  "custom_analysis": [
    {
      "question": "The original question",
      "answer": "Your data-driven answer",
      "evidence": ["supporting fact 1", "supporting fact 2"],
      "confidence": "high | medium | low"
    }
  ]
}

Return ONLY the JSON object, no markdown code blocks or additional text."""


def generate_custom_answers(
    questions: list[str],
    metrics: dict,
    nl_summary: dict,
    model_name: str = "gemini-3-pro-preview"
) -> list[dict]:
    """Generate answers to custom analytical questions using LLM.

    v3.9.1: Uses the same data context as generate_insights but answers
    specific user-provided questions.
    """
    if not questions:
        return []

    # Build context from metrics and NL summary (reuse build_insights_prompt structure)
    context_prompt = build_insights_prompt(metrics, nl_summary)

    # Format questions
    questions_text = "\n".join(f"{i+1}. {q}" for i, q in enumerate(questions))

    prompt = f"""Based on the following voice agent performance data, answer these custom analytical questions:

## QUESTIONS TO ANSWER

{questions_text}

---

{context_prompt}

---

Now answer each question based on the data above. Be specific and cite evidence from the data."""

    # Call LLM
    client = get_genai_client()

    config = types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=16000,
        system_instruction=CUSTOM_QUESTIONS_SYSTEM_PROMPT,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )

    result = extract_json_from_response(response.text)
    return result.get("custom_analysis", [])


def build_insights_prompt(metrics: dict, nl_summary: dict) -> str:
    """Build the prompt for LLM insight generation using v3.4 NL summary format with call_ids."""

    # Format Section A metrics
    metrics_json = json.dumps(metrics, indent=2)

    # Format natural language fields from nl_summary_v3 structure
    nl_sections = []

    # Failures grouped by type (v3.3: includes call_id for traceability)
    if nl_summary.get("by_failure_type"):
        nl_sections.append("## Failures by Type")
        for fp_type, entries in nl_summary["by_failure_type"].items():
            nl_sections.append(f"\n### {fp_type} ({len(entries)} calls)")
            for entry in entries[:10]:  # Limit per type
                call_id = entry.get('call_id', 'unknown')
                outcome = entry.get('outcome', 'unknown')
                parts = [f"[{call_id}]", f"[{outcome}]"]
                if entry.get("description"):
                    parts.append(entry["description"])
                if entry.get("verbatim"):
                    parts.append(f'Customer: "{entry["verbatim"]}"')
                if entry.get("miss"):
                    parts.append(f'Miss: {entry["miss"]}')
                nl_sections.append(f"- {' | '.join(parts)}")

    # Customer verbatims (all quotes) - v3.3: includes call_id
    if nl_summary.get("all_verbatims"):
        nl_sections.append("\n## Customer Verbatims (direct quotes)")
        for v in nl_summary["all_verbatims"][:15]:
            call_id = v.get("call_id", "unknown")
            outcome = v.get("outcome", "unknown")
            quote = v.get("quote", "")
            nl_sections.append(f'- [{call_id}] [{outcome}] "{quote}"')

    # Agent miss details
    if nl_summary.get("all_agent_misses"):
        nl_sections.append("\n## Agent Miss Details (coaching opportunities)")
        for m in nl_summary["all_agent_misses"][:15]:
            call_id = m.get("call_id", "unknown")
            recoverable = "recoverable" if m.get("was_recoverable") else "not recoverable"
            nl_sections.append(f"- [{call_id}] [{recoverable}] {m.get('miss', '')}")

    # Policy gap details (structured) - v3.3: includes call_id
    if nl_summary.get("policy_gap_details"):
        nl_sections.append("\n## Policy Gap Details (sample)")
        for g in nl_summary["policy_gap_details"][:15]:
            call_id = g.get("call_id", "unknown")
            nl_sections.append(
                f"- [{call_id}] [{g.get('category', 'unknown')}] Gap: {g.get('gap', '')} | "
                f"Ask: {g.get('ask', '')} | Blocker: {g.get('blocker', '')}"
            )

    # v3.4: ALL customer asks for semantic clustering (not limited to 15)
    if nl_summary.get("all_customer_asks"):
        all_asks = nl_summary["all_customer_asks"]
        nl_sections.append(f"\n## ALL Customer Asks for Clustering ({len(all_asks)} total)")
        nl_sections.append("Cluster these semantically similar asks and count them:")
        for ask in all_asks:
            nl_sections.append(f"- {ask}")

    # Failed call flows - v3.3: includes call_id
    if nl_summary.get("failed_call_flows"):
        nl_sections.append("\n## Sample Failed Call Flows")
        for call in nl_summary["failed_call_flows"][:5]:
            call_id = call.get("call_id", "unknown")
            steps = call.get("steps", [])
            steps_str = " → ".join(steps[:10])
            nl_sections.append(f"- [{call_id}] [{call.get('outcome', 'unknown')}] {steps_str}")

    # v3.5: Training opportunities grouped by type with failure context
    if nl_summary.get("training_details"):
        details = nl_summary["training_details"]
        nl_sections.append(f"\n## Training Opportunities ({len(details)} calls)")
        nl_sections.append("Analyze these training gaps to understand root causes and cross-correlations:")

        # Group by training type
        from collections import defaultdict
        by_type = defaultdict(list)
        for td in details:
            by_type[td.get("opportunity", "unknown")].append(td)

        for opp_type, entries in sorted(by_type.items(), key=lambda x: -len(x[1])):
            nl_sections.append(f"\n### {opp_type} ({len(entries)} calls)")
            for entry in entries[:8]:  # Sample per type to avoid overwhelming prompt
                parts = [f"[{entry.get('call_id', '?')[:8]}]"]
                parts.append(f"[{entry.get('failure_point', 'none')}]")
                if entry.get("failure_description"):
                    parts.append(entry["failure_description"][:100])
                if entry.get("agent_miss_detail"):
                    parts.append(f"Miss: {entry['agent_miss_detail'][:80]}")
                nl_sections.append(f"- {' | '.join(parts)}")

    # v3.5: Additional intents (secondary customer needs)
    if nl_summary.get("all_additional_intents"):
        intents = nl_summary["all_additional_intents"]
        nl_sections.append(f"\n## Secondary Intents ({len(intents)} calls)")
        nl_sections.append("Cluster these secondary customer needs:")
        for i in intents[:15]:
            call_id = i.get("call_id", "?")[:8]
            outcome = i.get("outcome", "unknown")
            intent = i.get("intent", "")
            nl_sections.append(f"- [{call_id}] [{outcome}] {intent}")

    # v3.6: Conversation Quality Events
    clar_events = nl_summary.get("clarification_events", [])
    corr_events = nl_summary.get("correction_events", [])
    loop_events = nl_summary.get("loop_events", [])
    turn_outliers = nl_summary.get("turn_outliers", [])

    if clar_events or corr_events or loop_events or turn_outliers:
        nl_sections.append("\n## Conversation Quality (v3.6)")

        # Clarification events
        if clar_events:
            nl_sections.append(f"\n### Clarification Events ({len(clar_events)} total)")
            nl_sections.append("Analyze which clarification types cause most friction:")

            # Group by type for analysis
            from collections import defaultdict
            by_type = defaultdict(list)
            by_cause = defaultdict(list)  # v3.7: Group by cause
            for e in clar_events:
                by_type[e.get("type", "unknown")].append(e)
                cause = e.get("cause")
                if cause:
                    by_cause[cause].append(e)

            for ctype, events in sorted(by_type.items(), key=lambda x: -len(x[1])):
                resolved_count = sum(1 for e in events if e.get("resolved"))
                failed_outcomes = sum(1 for e in events if e.get("outcome") != "resolved")
                nl_sections.append(f"- **{ctype}**: {len(events)} events, {resolved_count} resolved, {failed_outcomes} in failed calls")
                # Show a few examples with context (v3.7)
                for e in events[:3]:
                    context = e.get("context", "")
                    context_str = f" - {context}" if context else ""
                    nl_sections.append(f"  - [{e.get('call_id', '?')[:8]}] turn {e.get('turn', '?')} [{e.get('cause', '?')}] → {e.get('outcome', '?')}{context_str}")

            # v3.7: Show cause distribution
            if by_cause:
                nl_sections.append("\n### Clarification Causes (v3.7)")
                nl_sections.append("Analyze root causes of clarification friction:")
                for cause, events in sorted(by_cause.items(), key=lambda x: -len(x[1])):
                    failed_outcomes = sum(1 for e in events if e.get("outcome") != "resolved")
                    nl_sections.append(f"- **{cause}**: {len(events)} events, {failed_outcomes} in failed calls")
                    # Show context sentences for pattern discovery
                    contexts = [e.get("context") for e in events if e.get("context")][:3]
                    for ctx in contexts:
                        nl_sections.append(f"  - \"{ctx}\"")

        # Correction events
        if corr_events:
            nl_sections.append(f"\n### User Corrections ({len(corr_events)} total)")
            frustrated = sum(1 for e in corr_events if e.get("frustrated"))
            nl_sections.append(f"Corrections with frustration signal: {frustrated}")

            # v3.7: Group by severity
            from collections import defaultdict
            by_severity = defaultdict(list)
            for e in corr_events:
                severity = e.get("severity") or "unknown"
                by_severity[severity].append(e)

            for severity, events in sorted(by_severity.items(), key=lambda x: ({"major": 0, "moderate": 1, "minor": 2}.get(x[0], 3))):
                failed_outcomes = sum(1 for e in events if e.get("outcome") != "resolved")
                severity_label = (severity or "unknown").upper()
                nl_sections.append(f"\n**{severity_label}** ({len(events)} events, {failed_outcomes} in failed calls):")
                for e in events[:5]:
                    frust = "😤" if e.get("frustrated") else ""
                    context = e.get("context", "")
                    context_str = f" - {context}" if context else ""
                    nl_sections.append(f"  - [{e.get('call_id', '?')[:8]}] turn {e.get('turn', '?')}: {e.get('what', 'unknown')} [{e.get('outcome', '?')}] {frust}{context_str}")

        # Loop events (v3.9.1: agent_loops with type + subject + context)
        if loop_events:
            # Check if using v3.8+ format (has type field) or legacy format
            v38_events = [e for e in loop_events if e.get("type")]
            if v38_events:
                nl_sections.append(f"\n### Agent Loops ({len(v38_events)} friction loops)")
                nl_sections.append("Analyze loop types and their patterns:")

                # Group by type for analysis
                from collections import defaultdict
                by_type = defaultdict(list)
                for e in v38_events:
                    by_type[e.get("type", "unknown")].append(e)

                for loop_type, events in sorted(by_type.items(), key=lambda x: -len(x[1])):
                    failed_outcomes = sum(1 for e in events if e.get("outcome") != "resolved")
                    nl_sections.append(f"\n**{loop_type}** ({len(events)} events, {failed_outcomes} in failed calls):")
                    for e in events[:5]:
                        context = e.get("context", "")
                        subject = e.get("subject", "")
                        subject_str = f" [{subject}]" if subject else ""
                        context_str = f" - {context}" if context else ""
                        nl_sections.append(f"  - [{e.get('call_id', '?')[:8]}]{subject_str} → {e.get('outcome', '?')}{context_str}")
            else:
                # Legacy format (repeated_prompts)
                nl_sections.append(f"\n### Loop Events ({len(loop_events)} calls with repeated prompts)")
                sorted_loops = sorted(loop_events, key=lambda x: -(x.get("max_consecutive") or 0))
                for e in sorted_loops[:5]:
                    nl_sections.append(f"- [{e.get('call_id', '?')[:8]}] {e.get('count', 0)} repeats, max {e.get('max_consecutive', 0)} consecutive → {e.get('outcome', '?')}")

        # v3.9.1: Loop Subject Pairs for clustering
        loop_subject_pairs = nl_summary.get("loop_subject_pairs", [])
        if loop_subject_pairs:
            nl_sections.append(f"\n### Loop Subject Data for Clustering ({len(loop_subject_pairs)} pairs)")
            nl_sections.append("Cluster these (loop_type, subject) pairs and identify high-impact patterns:")

            from collections import defaultdict, Counter
            # Group by loop_type
            by_type = defaultdict(list)
            for p in loop_subject_pairs:
                by_type[p.get("loop_type", "unknown")].append(p)

            for loop_type, pairs in sorted(by_type.items(), key=lambda x: -len(x[1])):
                nl_sections.append(f"\n**{loop_type}** ({len(pairs)} events):")
                # Show subject distribution
                subjects = Counter(p.get("subject") for p in pairs if p.get("subject"))
                failed_by_subject = Counter(
                    p.get("subject") for p in pairs
                    if p.get("subject") and p.get("outcome") != "resolved"
                )
                for subject, count in subjects.most_common(5):
                    failed = failed_by_subject.get(subject, 0)
                    nl_sections.append(f"  - {subject}: {count} events ({failed} in failed calls)")

        # Turn outliers
        if turn_outliers:
            long_calls = [e for e in turn_outliers if e.get("type") == "long"]
            short_calls = [e for e in turn_outliers if e.get("type") == "short"]

            if long_calls:
                nl_sections.append(f"\n### Unusually Long Calls ({len(long_calls)} calls)")
                for e in long_calls[:5]:
                    nl_sections.append(f"- [{e.get('call_id', '?')[:8]}] {e.get('turns', '?')} turns → {e.get('outcome', '?')}")

            if short_calls:
                nl_sections.append(f"\n### Unusually Short Calls ({len(short_calls)} calls)")
                for e in short_calls[:5]:
                    nl_sections.append(f"- [{e.get('call_id', '?')[:8]}] {e.get('turns', '?')} turns → {e.get('outcome', '?')}")

    # v3.9: Disposition Summary
    disp_summary = nl_summary.get("disposition_summary", {})
    if disp_summary:
        nl_sections.append("\n## Call Disposition (v3.9)")
        nl_sections.append("Analyze disposition patterns for funnel insights:")

        for disposition, entries in sorted(disp_summary.items(), key=lambda x: -len(x[1])):
            nl_sections.append(f"\n### {disposition} ({len(entries)} calls)")
            for entry in entries[:8]:  # Sample per disposition
                parts = [f"[{entry.get('call_id', '?')[:8]}]"]
                if entry.get("summary"):
                    parts.append(entry["summary"][:100])
                elif entry.get("description"):
                    parts.append(entry["description"][:100])
                if entry.get("verbatim"):
                    parts.append(f'"{entry["verbatim"][:60]}..."' if len(entry.get("verbatim", "")) > 60 else f'"{entry["verbatim"]}"')
                nl_sections.append(f"- {' | '.join(parts)}")

    nl_text = "\n".join(nl_sections)

    return f"""Analyze this voice agent performance data and generate Section B insights.

## SECTION A: DETERMINISTIC METRICS

```json
{metrics_json}
```

## AGGREGATED QUALITATIVE DATA

{nl_text}

---

Based on the above data, generate the Section B insights JSON following the schema in your instructions.
Focus on:
1. What's the #1 thing causing failures?
2. What are the top 3 actionable fixes?
3. What does the customer voice tell us?

Return ONLY the JSON object."""


def generate_insights(
    metrics_path: Path,
    nl_summary_path: Path | None,
    model_name: str = "gemini-3-pro-preview"
) -> dict:
    """Generate Section B insights using LLM.

    Args:
        metrics_path: Path to Section A metrics JSON
        nl_summary_path: Path to NL summary from extract_nl_fields.py
        model_name: Gemini model to use
    """

    # Load Section A metrics
    with open(metrics_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)

    metrics = report_data.get("deterministic_metrics", report_data)

    # Load NL summary (required for v3.1)
    nl_summary = {}
    if nl_summary_path and nl_summary_path.exists():
        with open(nl_summary_path, 'r', encoding='utf-8') as f:
            nl_summary = json.load(f)
        print(f"Loaded NL summary: {nl_summary.get('metadata', {}).get('calls_with_nl_data', 0)} calls with NL data", file=sys.stderr)
    else:
        print("Warning: No NL summary provided. Insights will be based on metrics only.", file=sys.stderr)

    # Build prompt
    prompt = build_insights_prompt(metrics, nl_summary)

    # Call LLM with new SDK
    client = get_genai_client()

    config = types.GenerateContentConfig(
        temperature=0.3,
        max_output_tokens=32000,  # v3.3: High limit for expanded schema with large datasets
        # No thinking_config = uses model default (HIGH for Pro)
        system_instruction=INSIGHTS_SYSTEM_PROMPT,
    )

    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=config,
    )

    insights = extract_json_from_response(response.text)

    return insights


def combine_report(
    metrics_path: Path,
    insights: dict,
    output_path: Path,
    custom_analysis: list[dict] = None
) -> dict:
    """Combine Section A metrics, Section B insights, and custom analysis into final report.

    v3.9.1: Added custom_analysis parameter for user-provided questions.
    """

    with open(metrics_path, 'r', encoding='utf-8') as f:
        report_data = json.load(f)

    metrics = report_data.get("deterministic_metrics", report_data)

    full_report = {
        "deterministic_metrics": metrics,
        "llm_insights": insights
    }

    # v3.9.1: Include custom analysis if provided
    if custom_analysis:
        full_report["custom_analysis"] = custom_analysis

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(full_report, f, indent=2)

    return full_report


def print_insights_summary(insights: dict) -> None:
    """Print human-readable summary of insights."""
    print("\n" + "=" * 60)
    print("VACATIA AI VOICE AGENT - INSIGHTS (v3.9.1 - Section B)")
    print("=" * 60)

    # Executive Summary
    print("\n" + "-" * 40)
    print("EXECUTIVE SUMMARY")
    print("-" * 40)
    print(f"  {insights.get('executive_summary', 'N/A')}")

    # Root Cause Analysis
    print("\n" + "-" * 40)
    print("ROOT CAUSE ANALYSIS")
    print("-" * 40)
    rca = insights.get("root_cause_analysis", {})
    print(f"  Primary Driver: {rca.get('primary_driver', 'N/A')}")
    factors = rca.get("contributing_factors", [])
    if factors:
        print("  Contributing Factors:")
        for f in factors:
            print(f"    - {f}")
    print(f"  Evidence: {rca.get('evidence', 'N/A')}")

    # Recommendations
    print("\n" + "-" * 40)
    print("RECOMMENDATIONS")
    print("-" * 40)
    recs = insights.get("actionable_recommendations", [])
    for rec in recs:
        priority = rec.get("priority", "?")
        category = rec.get("category", "?")
        print(f"\n  [{priority}] {rec.get('recommendation', 'N/A')}")
        print(f"       Category: {category}")
        print(f"       Impact: {rec.get('expected_impact', 'N/A')}")

    # Trend Narratives
    print("\n" + "-" * 40)
    print("TREND NARRATIVES")
    print("-" * 40)
    narratives = insights.get("trend_narratives", {})
    for name, narrative in narratives.items():
        label = name.replace("_", " ").title()
        print(f"\n  {label}:")
        print(f"    {narrative}")

    # Verbatim Highlights
    print("\n" + "-" * 40)
    print("CUSTOMER VOICE HIGHLIGHTS")
    print("-" * 40)
    verbatims = insights.get("verbatim_highlights", {})
    if verbatims.get("most_frustrated"):
        print(f"\n  Most Frustrated:")
        print(f'    "{verbatims["most_frustrated"]}"')
    if verbatims.get("most_common_ask"):
        print(f"\n  Most Common Ask:")
        print(f'    "{verbatims["most_common_ask"]}"')
    if verbatims.get("biggest_miss"):
        print(f"\n  Biggest Agent Miss:")
        print(f'    "{verbatims["biggest_miss"]}"')

    # v3.5: Training Analysis
    training = insights.get("training_analysis", {})
    if training:
        print("\n" + "-" * 40)
        print("TRAINING ANALYSIS (v3.5)")
        print("-" * 40)
        if training.get("narrative"):
            print(f"  {training['narrative']}")
        priorities = training.get("top_priorities", [])
        if priorities:
            print("\n  Top Priorities:")
            for p in priorities[:3]:
                print(f"    - {p.get('skill', 'N/A')} ({p.get('count', 0)}): {p.get('action', 'N/A')}")
        correlations = training.get("cross_correlations", [])
        if correlations:
            print("\n  Cross-Correlations:")
            for c in correlations:
                print(f"    - {c.get('pattern', 'N/A')} ({c.get('count', 0)}): {c.get('insight', 'N/A')}")

    # v3.5: Emergent Patterns
    emergent = insights.get("emergent_patterns", [])
    if emergent:
        print("\n" + "-" * 40)
        print("EMERGENT PATTERNS (v3.5)")
        print("-" * 40)
        for p in emergent:
            print(f"\n  {p.get('name', 'Unnamed')} ({p.get('frequency', 'unknown')})")
            print(f"    {p.get('significance', '')}")

    # v3.5: Secondary Intents
    secondary = insights.get("secondary_intents_analysis", {})
    if secondary.get("clusters"):
        print("\n" + "-" * 40)
        print("SECONDARY INTENTS (v3.5)")
        print("-" * 40)
        if secondary.get("narrative"):
            print(f"  {secondary['narrative']}")
        for c in secondary.get("clusters", [])[:3]:
            print(f"    - {c.get('cluster', 'N/A')} ({c.get('count', 0)})")

    # v3.7: Conversation Quality Analysis
    cq_analysis = insights.get("conversation_quality_analysis", {})
    if cq_analysis:
        print("\n" + "-" * 40)
        print("CONVERSATION QUALITY ANALYSIS (v3.7)")
        print("-" * 40)
        if cq_analysis.get("narrative"):
            print(f"  {cq_analysis['narrative']}")

        friction_hotspots = cq_analysis.get("friction_hotspots", [])
        if friction_hotspots:
            print("\n  Friction Hotspots:")
            for fh in friction_hotspots[:3]:
                print(f"    - {fh.get('pattern', 'N/A')} ({fh.get('frequency', 'N/A')})")
                print(f"      Impact: {fh.get('impact', 'N/A')}")

        # v3.7: Cause analysis
        cause_analysis = cq_analysis.get("cause_analysis", {})
        if cause_analysis.get("insights"):
            print("\n  Clarification Cause Analysis (v3.7):")
            if cause_analysis.get("narrative"):
                print(f"    {cause_analysis['narrative']}")
            for ci in cause_analysis.get("insights", [])[:3]:
                print(f"    - {ci.get('cause', 'N/A')} ({ci.get('frequency', 'N/A')}): {ci.get('recommendation', 'N/A')}")

        # v3.7: Severity analysis
        severity_analysis = cq_analysis.get("severity_analysis", {})
        if severity_analysis.get("insights"):
            print("\n  Correction Severity Analysis (v3.7):")
            if severity_analysis.get("narrative"):
                print(f"    {severity_analysis['narrative']}")
            for si in severity_analysis.get("insights", [])[:3]:
                print(f"    - {si.get('severity', 'N/A')} ({si.get('frequency', 'N/A')}): {si.get('recommendation', 'N/A')}")

        efficiency_insights = cq_analysis.get("efficiency_insights", [])
        if efficiency_insights:
            print("\n  Efficiency Insights:")
            for ei in efficiency_insights[:3]:
                print(f"    - {ei}")

        turn_analysis = cq_analysis.get("turn_analysis", {})
        if turn_analysis:
            print("\n  Turn Analysis:")
            if turn_analysis.get("long_call_patterns"):
                print(f"    Long calls: {turn_analysis['long_call_patterns']}")
            if turn_analysis.get("turns_to_failure_insight"):
                print(f"    Failure pattern: {turn_analysis['turns_to_failure_insight']}")

        # v3.8: Loop type analysis
        loop_analysis = cq_analysis.get("loop_type_analysis", {})
        if loop_analysis.get("insights"):
            print("\n  Loop Type Analysis (v3.8):")
            if loop_analysis.get("narrative"):
                print(f"    {loop_analysis['narrative']}")
            for li in loop_analysis.get("insights", [])[:3]:
                print(f"    - {li.get('type', 'N/A')} ({li.get('frequency', 'N/A')}): {li.get('recommendation', 'N/A')}")
            if loop_analysis.get("intent_retry_rate"):
                print(f"    Intent Retry Rate: {loop_analysis['intent_retry_rate']}")
            if loop_analysis.get("deflection_rate"):
                print(f"    Deflection Rate: {loop_analysis['deflection_rate']}")

    # v3.9: Disposition Analysis
    disp_analysis = insights.get("disposition_analysis", {})
    if disp_analysis:
        print("\n" + "-" * 40)
        print("DISPOSITION ANALYSIS (v3.9)")
        print("-" * 40)
        if disp_analysis.get("narrative"):
            print(f"  {disp_analysis['narrative']}")

        actionable_insights = disp_analysis.get("actionable_insights", [])
        if actionable_insights:
            print("\n  Actionable Insights:")
            for ai in actionable_insights[:4]:
                print(f"    - {ai.get('disposition', 'N/A')} ({ai.get('frequency', 'N/A')})")
                print(f"      Root cause: {ai.get('root_cause', 'N/A')}")
                print(f"      Recommendation: {ai.get('recommendation', 'N/A')}")

        funnel_health = disp_analysis.get("funnel_health", {})
        if funnel_health:
            assessment = funnel_health.get('assessment') or 'N/A'
            print(f"\n  Funnel Health: {assessment.upper()}")
            if funnel_health.get("priority_focus"):
                print(f"    Priority Focus: {funnel_health['priority_focus']}")

    # v3.9.1: Loop Subject Clusters
    loop_subject = insights.get("loop_subject_clusters", {})
    if loop_subject:
        print("\n" + "-" * 40)
        print("LOOP SUBJECT ANALYSIS (v3.9.1)")
        print("-" * 40)
        if loop_subject.get("narrative"):
            print(f"  {loop_subject['narrative']}")

        by_loop_type = loop_subject.get("by_loop_type", {})
        if by_loop_type:
            print("\n  By Loop Type:")
            for loop_type, data in by_loop_type.items():
                top_subjects = data.get("top_subjects", [])
                if top_subjects:
                    subjects_str = ", ".join(
                        f"{s.get('subject', 'N/A')} ({s.get('pct', 'N/A')})"
                        for s in top_subjects[:3]
                    )
                    print(f"    {loop_type}: {subjects_str}")

        high_impact = loop_subject.get("high_impact_patterns", [])
        if high_impact:
            print("\n  High-Impact Patterns:")
            for p in high_impact[:3]:
                print(f"    - {p.get('loop_type', 'N/A')}/{p.get('subject', 'N/A')}: {p.get('recommendation', 'N/A')}")

    print("\n" + "=" * 60)


def print_custom_analysis(custom_analysis: list[dict]) -> None:
    """Print custom analysis section.

    v3.9.1: Displays answers to user-provided questions.
    """
    if not custom_analysis:
        return

    print("\n" + "=" * 60)
    print("CUSTOM ANALYSIS (v3.9.1)")
    print("=" * 60)

    for i, qa in enumerate(custom_analysis, 1):
        question = qa.get("question", "N/A")
        answer = qa.get("answer", "N/A")
        confidence = qa.get("confidence", "medium")
        evidence = qa.get("evidence", [])

        print(f"\n{i}. {question}")
        print(f"   [{confidence.upper()}]")
        print(f"   {answer}")

        if evidence:
            print("\n   Evidence:")
            for e in evidence[:3]:
                print(f"     - {e}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Generate v3.9.1 Section B: LLM Insights (loop subject granularity)")
    parser.add_argument("-m", "--metrics", type=Path,
                        help="Path to Section A metrics JSON file")
    parser.add_argument("-n", "--nl-summary", type=Path,
                        help="Path to NL summary JSON from extract_nl_fields.py")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "reports",
                        help="Output directory for combined report")
    parser.add_argument("--model", type=str, default="gemini-3-pro-preview",
                        help="Gemini model to use (default: gemini-3-pro-preview)")
    parser.add_argument("--json-only", action="store_true",
                        help="Output only JSON")
    parser.add_argument("--questions", type=Path,
                        help="Path to questions file (one question per line) for custom analysis")

    args = parser.parse_args()

    reports_dir = args.output_dir

    # Find latest metrics file if not specified
    if not args.metrics:
        if reports_dir.exists():
            metrics_files = sorted(reports_dir.glob("metrics_v3_*.json"), reverse=True)
            if metrics_files:
                args.metrics = metrics_files[0]
                print(f"Using latest metrics: {args.metrics}", file=sys.stderr)

    if not args.metrics or not args.metrics.exists():
        print("Error: No metrics file found. Run compute_metrics.py first.", file=sys.stderr)
        return 1

    # Find latest NL summary if not specified
    if not args.nl_summary:
        if reports_dir.exists():
            nl_files = sorted(reports_dir.glob("nl_summary_v3_*.json"), reverse=True)
            if nl_files:
                args.nl_summary = nl_files[0]
                print(f"Using latest NL summary: {args.nl_summary}", file=sys.stderr)

    if not args.nl_summary or not args.nl_summary.exists():
        print("Warning: No NL summary found. Run extract_nl_fields.py first for richer insights.", file=sys.stderr)

    try:
        configure_genai()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Generating insights from: {args.metrics}", file=sys.stderr)

    try:
        insights = generate_insights(
            args.metrics,
            args.nl_summary,
            args.model
        )
    except Exception as e:
        print(f"Error generating insights: {e}", file=sys.stderr)
        return 1

    # v3.9.1: Process custom questions if provided
    custom_analysis = None
    if args.questions:
        questions = load_questions(args.questions)
        if questions:
            print(f"Processing {len(questions)} custom questions...", file=sys.stderr)

            # Load metrics and NL summary for custom questions
            with open(args.metrics, 'r', encoding='utf-8') as f:
                metrics_data = json.load(f)
            metrics = metrics_data.get("deterministic_metrics", metrics_data)

            nl_summary = {}
            if args.nl_summary and args.nl_summary.exists():
                with open(args.nl_summary, 'r', encoding='utf-8') as f:
                    nl_summary = json.load(f)

            try:
                custom_analysis = generate_custom_answers(
                    questions, metrics, nl_summary, args.model
                )
                print(f"Generated {len(custom_analysis)} custom answers", file=sys.stderr)
            except Exception as e:
                print(f"Warning: Failed to generate custom answers: {e}", file=sys.stderr)
        else:
            print(f"Warning: No questions found in {args.questions}", file=sys.stderr)

    # Create output path
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args.output_dir / f"report_v3_{timestamp}.json"

    # Combine and save (v3.9.1: includes custom_analysis)
    full_report = combine_report(args.metrics, insights, output_path, custom_analysis)
    print(f"Full report saved: {output_path}", file=sys.stderr)

    if args.json_only:
        print(json.dumps(full_report, indent=2))
    else:
        print_insights_summary(insights)
        # v3.9.1: Print custom analysis if available
        if custom_analysis:
            print_custom_analysis(custom_analysis)
        print(f"\nFull report: {output_path}")

    return 0


if __name__ == "__main__":
    exit(main())
