#!/usr/bin/env python3
"""
Single Transcript Analyzer for Vacatia AI Voice Agent Analytics (v3.6)

Hybrid schema with 23 per-call fields enabling both programmatic analysis
and executive-ready insights through a two-part report architecture.

v3.6 additions (5 new fields):
- conversation_turns: Total exchange pairs (proxy for call duration)
- turns_to_failure: When the call started derailing (non-resolved only)
- clarification_requests: Agent asks customer to repeat/spell/confirm
- user_corrections: Customer corrects agent's understanding
- repeated_prompts: Agent says substantially similar things multiple times
"""

import argparse
import json
import os
import sys
from pathlib import Path

import google.generativeai as genai


# v3.6 Schema - Hybrid metrics + insights (23 fields)
ANALYSIS_SCHEMA = """
{
  "call_id": "string (UUID from filename)",

  "outcome": "enum: 'resolved' | 'escalated' | 'abandoned' | 'unclear'",
  "resolution_type": "string (what was accomplished: 'payment processed', 'callback scheduled', 'info provided', 'none', etc.)",

  "agent_effectiveness": "integer 1-5 (did agent understand and respond appropriately?)",
  "conversation_quality": "integer 1-5 (flow, tone, clarity combined)",
  "customer_effort": "integer 1-5 (how hard did customer work? 1=effortless, 5=painful)",

  "failure_point": "enum: 'none' | 'nlu_miss' | 'wrong_action' | 'policy_gap' | 'customer_confusion' | 'tech_issue' | 'other'",
  "failure_description": "string or null (one sentence: what went wrong, if anything)",
  "was_recoverable": "boolean or null (could agent have saved this call?)",
  "critical_failure": "boolean (wrong info given, impossible promise made, compliance issue)",

  "escalation_requested": "boolean (did customer ask for a human?)",
  "repeat_caller_signals": "boolean (customer mentioned prior calls, ongoing frustration)",
  "training_opportunity": "string or null (specific skill gap if any: 'payment_flow', 'empathy', 'clarification', 'verification', etc.)",

  "additional_intents": "string or null (if customer had secondary requests beyond main intent, describe briefly)",

  "summary": "string (1-2 sentence summary focused on outcome and key driver)",

  "policy_gap_detail": "object or null - REQUIRED when failure_point='policy_gap'. Structure: { 'category': enum 'capability_limit' | 'data_access' | 'auth_restriction' | 'business_rule' | 'integration_missing', 'specific_gap': string (what exactly couldn't be done), 'customer_ask': string (what the customer wanted), 'blocker': string (why it couldn't be fulfilled) }",

  "customer_verbatim": "string or null (key quote from customer capturing their core frustration, need, or emotional state. Direct transcript excerpt, 1-2 sentences max. Example: 'I've called three times about this and no one can help me')",

  "agent_miss_detail": "string or null (what the agent should have said or done differently. Actionable coaching insight. Example: 'Should have offered callback scheduling when customer expressed time constraints instead of continuing verification loop')",

  "resolution_steps": "array of strings or null (sequence of actions taken or attempted during the call. Example: ['greeted customer', 'attempted verification via phone', 'verification failed', 'offered alternative ID method', 'customer declined', 'offered human escalation', 'customer accepted'])",

  "conversation_turns": "integer (total user+assistant exchange pairs; count how many times user spoke)",

  "turns_to_failure": "integer or null (for non-resolved calls: turn number where call started derailing; null for resolved calls)",

  "clarification_requests": {
    "count": "integer (total clarification requests)",
    "details": [
      {
        "type": "enum: 'name_spelling' | 'phone_confirmation' | 'intent_clarification' | 'repeat_request' | 'verification_retry'",
        "turn": "integer (which turn)",
        "resolved": "boolean (did clarification succeed?)"
      }
    ]
  },

  "user_corrections": {
    "count": "integer (total customer corrections)",
    "details": [
      {
        "what_was_wrong": "string (brief: what agent got wrong)",
        "turn": "integer (which turn)",
        "frustration_signal": "boolean (did customer show frustration during correction?)"
      }
    ]
  },

  "repeated_prompts": {
    "count": "integer (total repeated prompts)",
    "max_consecutive": "integer (longest streak of similar prompts)"
  }
}
"""

SYSTEM_PROMPT = """You are an expert call center quality analyst. Analyze this Vacatia AI voice agent call transcript and produce a focused JSON analysis.

## Context
Vacatia is a timeshare management company. Their AI handles: RCI membership, timeshare deposits, reservations, payments, account verification.

## Transcript Format
- `assistant:` = AI voice agent
- `user:` = Customer

## Outcome Classification Rules
- **resolved**: Customer's primary need was addressed (even if partially)
- **escalated**: Call transferred to human agent
- **abandoned**: Call ended without resolution - includes:
  - Greeting-only calls (no customer engagement)
  - Customer hung up
  - Call disconnected mid-conversation
- **unclear**: ONLY use when transcript is corrupted/unreadable

Prefer a definitive outcome over "unclear". If customer got ANY useful info, lean toward "resolved".

## Scoring Guidelines (1-5 scale)

**agent_effectiveness** - Did the agent understand and respond appropriately?
- 5: Perfect understanding, ideal responses
- 3: Adequate, some minor misses
- 1: Frequent misunderstandings, poor responses

**conversation_quality** - Flow, tone, clarity combined
- 5: Natural, professional, clear
- 3: Functional but robotic or slightly awkward
- 1: Stilted, confusing, or inappropriate tone

**customer_effort** - How hard did the customer work? (1=best)
- 1: Effortless, smooth experience
- 3: Some repetition or confusion to work through
- 5: Frustrating, excessive effort required

## Failure Point Categories
- **none**: Call succeeded (use ONLY when outcome="resolved")
- **nlu_miss**: Agent misunderstood what customer said/wanted
- **wrong_action**: Agent understood but took wrong action
- **policy_gap**: Agent couldn't help due to business rules/limitations
- **customer_confusion**: Customer unclear or provided wrong info
- **tech_issue**: System errors, call quality problems
- **other**: Anything else

## Policy Gap Categories (for policy_gap_detail.category)
- **capability_limit**: Feature not built yet (e.g., can't process refunds)
- **data_access**: Agent can't access needed information
- **auth_restriction**: Customer couldn't be verified to proceed
- **business_rule**: Policy prevents action (e.g., can't waive fees)
- **integration_missing**: System integration not available

## Critical Failure Flag
Set `critical_failure: true` ONLY if:
- Agent provided incorrect information
- Agent promised something impossible
- Compliance/security issue occurred
- Customer could be harmed by following agent's guidance

## VALIDATION RULES (CRITICAL)

1. **failure_point consistency**:
   - If outcome is "abandoned", "escalated", or "unclear" → failure_point MUST NOT be "none"
   - If outcome is "resolved" → failure_point SHOULD be "none" (exceptions allowed for partial resolution)

2. **policy_gap_detail required**:
   - If failure_point is "policy_gap" → policy_gap_detail MUST be a complete object with all 4 fields

3. **agent_miss_detail conditional**:
   - If was_recoverable is true → agent_miss_detail SHOULD be populated with actionable coaching

## New v3 Fields Guidelines

**customer_verbatim**: Extract a direct quote (1-2 sentences) that best captures the customer's emotional state or core need. Look for expressions of frustration, urgency, or unmet expectations. If no notable quote, use null.

**agent_miss_detail**: Describe what the agent should have done differently. Be specific and actionable (e.g., "Should have offered callback scheduling" not "Could have been better"). If agent performed well or no clear miss, use null.

**resolution_steps**: List the key actions in chronological order. Keep each step brief (2-5 words). Include both successful and failed attempts. This creates a call flow timeline.

## Conversation Quality Tracking (v3.6)

**conversation_turns**: Count total exchanges. A "turn" = one user message followed by agent response. Count how many times the user spoke. Example: 15-turn call has 15 user messages.

**turns_to_failure**: For non-resolved calls ONLY, identify which turn the call started going wrong. This helps measure "time to failure" using turns as a proxy for duration (since timestamps aren't available). For resolved calls, use null.

**clarification_requests**: Track EVERY time the agent asks customer to clarify something:
- **name_spelling**: Agent asks to spell name ("Can you spell your name?")
- **phone_confirmation**: Agent confirms phone number ("So that's 315-276-0534?")
- **intent_clarification**: Agent asks what customer needs after intent was given ("What can I help you with?" when customer already stated need)
- **repeat_request**: Agent asks to repeat ("Can you say that again?", "Sorry, I didn't catch that")
- **verification_retry**: Agent asks for different verification info ("Can you try a different phone number?")

For each, note: turn number, whether the clarification successfully resolved the confusion.

**user_corrections**: Track when CUSTOMER corrects the agent's understanding:
- "No, that's not right"
- "I said X, not Y"
- "Let me spell that for you" (proactive correction)
- Direct disagreement with agent's interpretation

Note: turn number, what was wrong, whether customer showed frustration (exasperation, repeating themselves forcefully, expressing annoyance).

**repeated_prompts**: Count when agent says substantially the same thing multiple times. This detects loop behavior.
- Track total count of repeated prompts
- Track max consecutive repeats (indicates stuck-in-loop)
- "Substantially similar" = same intent/request, not exact text match
- Example: "Can you spell your name?" followed by "Please spell your first name" = 2 consecutive repeats

## Output Requirements
- Return ONLY valid JSON
- No markdown code blocks
- Be concise and actionable
- Focus on what matters for improving the agent
- Ensure all validation rules are followed
"""

USER_PROMPT_TEMPLATE = """Analyze this call transcript and return JSON matching this schema:

{schema}

## Transcript
Call ID: {call_id}

---
{transcript}
---

Return ONLY the JSON object."""


def configure_genai():
    """Configure the Google Generative AI client."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)


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

    raise ValueError(f"Could not parse JSON from: {text[:300]}...")


def analyze_transcript(transcript_path: Path, model_name: str = "gemini-3-flash-preview") -> dict:
    """Analyze a single transcript using the LLM."""
    content = transcript_path.read_text(encoding='utf-8')
    call_id = transcript_path.stem

    user_prompt = USER_PROMPT_TEMPLATE.format(
        schema=ANALYSIS_SCHEMA,
        call_id=call_id,
        transcript=content
    )

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT
    )

    response = model.generate_content(
        user_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=4096,
        )
    )

    analysis = extract_json_from_response(response.text)
    analysis["call_id"] = call_id
    analysis["schema_version"] = "v3.6"

    return analysis


def save_analysis(analysis: dict, output_dir: Path) -> Path:
    """Save analysis JSON to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{analysis['call_id']}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Analyze call transcript (v3.6 - hybrid metrics + insights + conversation quality schema)")
    parser.add_argument("transcript", type=Path, help="Path to transcript file")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "analyses",
                        help="Output directory for analysis JSON")
    parser.add_argument("--model", type=str, default="gemini-3-flash-preview",
                        help="Gemini model to use (default: gemini-3-flash-preview)")
    parser.add_argument("--stdout", action="store_true",
                        help="Print to stdout instead of saving")

    args = parser.parse_args()

    if not args.transcript.exists():
        print(f"Error: File not found: {args.transcript}", file=sys.stderr)
        return 1

    try:
        configure_genai()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Analyzing: {args.transcript}", file=sys.stderr)

    try:
        analysis = analyze_transcript(args.transcript, args.model)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if args.stdout:
        print(json.dumps(analysis, indent=2))
    else:
        output_path = save_analysis(analysis, args.output_dir)
        print(f"Saved: {output_path}", file=sys.stderr)
        print(json.dumps(analysis, indent=2))

    return 0


if __name__ == "__main__":
    exit(main())
