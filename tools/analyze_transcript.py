#!/usr/bin/env python3
"""
Single Transcript Analyzer for Vacatia AI Voice Agent Analytics (v3.8)

Hybrid schema with 23 per-call fields enabling both programmatic analysis
and executive-ready insights through a two-part report architecture.

v3.8 additions:
- agent_loops: Replaces repeated_prompts with clearer terminology and type enum
- Loop types: info_retry, intent_retry, deflection, comprehension, action_retry
- Only tracks friction loops (benign repetition excluded)
- Each loop has type (enum for aggregation) + context (description for understanding)

v3.7 additions:
- Transcript preprocessing: Deterministic turn counting before LLM analysis
- clarification_requests.details[].cause: Enum explaining why clarification was needed
- clarification_requests.details[].context: Concise description (~10-20 words)
- user_corrections.details[].severity: Enum for correction severity
- user_corrections.details[].context: Concise description (~10-20 words)

v3.6 additions (5 new fields):
- conversation_turns: Total exchange pairs (proxy for call duration)
- turns_to_failure: When the call started derailing (non-resolved only)
- clarification_requests: Agent asks customer to repeat/spell/confirm
- user_corrections: Customer corrects agent's understanding
- (replaced by agent_loops in v3.8)
"""

import argparse
import json
import os
import sys
from pathlib import Path

import google.generativeai as genai

# Import preprocessing function
from preprocess_transcript import preprocess_transcript, format_for_llm


# v3.8 Schema - Hybrid metrics + insights + agent loops (23 fields)
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
        "turn": "integer (which turn, use the turn number from the preprocessed transcript)",
        "resolved": "boolean (did clarification succeed?)",
        "cause": "enum: 'customer_refused' | 'customer_unclear' | 'agent_misheard' | 'tech_issue' | 'successful'",
        "context": "string (10-20 words describing what happened)"
      }
    ]
  },

  "user_corrections": {
    "count": "integer (total customer corrections)",
    "details": [
      {
        "what_was_wrong": "string (brief: what agent got wrong)",
        "turn": "integer (which turn, use the turn number from the preprocessed transcript)",
        "frustration_signal": "boolean (did customer show frustration during correction?)",
        "severity": "enum: 'minor' | 'moderate' | 'major'",
        "context": "string (10-20 words describing what happened)"
      }
    ]
  },

  "agent_loops": {
    "count": "integer (total friction loops detected)",
    "details": [
      {
        "type": "enum: 'info_retry' | 'intent_retry' | 'deflection' | 'comprehension' | 'action_retry'",
        "context": "string (10-20 words describing what happened and when)"
      }
    ]
  }
}
"""

SYSTEM_PROMPT = """You are an expert call center quality analyst. Analyze this Vacatia AI voice agent call transcript and produce a focused JSON analysis.

## Context
Vacatia is a timeshare management company. Their AI handles: RCI membership, timeshare deposits, reservations, payments, account verification.

## Transcript Format
The transcript has been preprocessed with turn numbers for your reference:
- [Turn N] AGENT: = AI voice agent
- [Turn N] USER: = Customer

Use these turn numbers EXACTLY when referencing specific turns in clarification_requests and user_corrections.

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

## Agent Loops (v3.8)

Detect patterns where the agent repeats similar requests. Track ONLY friction loops (problematic), not benign repetition.

### Loop Types (all are friction)

- **info_retry**: Agent re-asks for information already provided
  - "Spell your name" asked twice
  - "What's your phone number?" after customer gave it

- **intent_retry**: Agent re-asks for intent already stated
  - "How can I help you?" AFTER customer already stated their need
  - Common pattern: Customer states intent → agent does identity check → agent asks intent again

- **deflection**: Generic questions masking inability to help
  - "Is there anything else?" while primary request is unresolved
  - "How else can I help?" when customer's issue wasn't addressed

- **comprehension**: Agent couldn't hear/understand, asks to repeat
  - "Sorry, can you say that again?"
  - "One more time please?"

- **action_retry**: Agent retries same action due to system/process failure
  - "Let me try that again"
  - "One moment, the system is slow"

### What to EXCLUDE (benign, don't track)

- Greeting after returning from hold
- Re-engagement after silence/topic change
- Required compliance disclosures
- Confirmation before taking action

### Output Format

For each friction loop:
- `type`: One of [info_retry, intent_retry, deflection, comprehension, action_retry]
- `context`: 10-20 word description of what happened and when (include turn numbers if helpful)

## Clarification & Correction Details (v3.7)

For each clarification request, provide TWO additional fields:

1. **cause** (enum): Why the clarification was needed
   - `customer_refused`: Customer declined to provide info ("No", "I already told you", demands human)
   - `customer_unclear`: Customer provided info but unclear (mumbled, partial, ambiguous)
   - `agent_misheard`: Agent failed to parse what customer said clearly
   - `tech_issue`: Audio cut off, connection problem, silence
   - `successful`: Clarification succeeded (use when resolved=true)

2. **context** (string, 10-20 words): Concise description of what happened
   - Good: "Customer refused to spell name, said 'I already told you twice'"
   - Good: "Agent heard 'Butchering' instead of 'Butcherine', customer corrected"
   - Bad: "The clarification was needed" (too vague)
   - Bad: Long paragraph describing the whole exchange (too verbose)

For each user correction, provide TWO additional fields:

1. **severity** (enum): How significant was the correction?
   - `minor`: Simple correction, no frustration, quickly resolved
   - `moderate`: Correction with mild frustration or took multiple attempts
   - `major`: Explicit anger, multiple corrections needed, or critical mistake

2. **context** (string, 10-20 words): Concise description of what happened
   - Good: "Customer corrected 'Grandview' to 'Grand Pacific' with audible frustration"
   - Good: "Customer spelled name again after agent confirmed wrong spelling"
   - Bad: "There was a correction" (too vague)

## Output Requirements
- Return ONLY valid JSON
- No markdown code blocks
- Be concise and actionable
- Focus on what matters for improving the agent
- Ensure all validation rules are followed
- Use the EXACT turn numbers from the preprocessed transcript
"""

USER_PROMPT_TEMPLATE = """Analyze this call transcript and return JSON matching this schema:

{schema}

## Preprocessed Transcript with Turn Numbers

{transcript}

Return ONLY the JSON object."""


def configure_genai():
    """Configure the Google Generative AI client."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable.")
    genai.configure(api_key=api_key)


def repair_truncated_json(text: str) -> str:
    """Attempt to repair truncated JSON by closing open structures."""
    # Find the last complete value
    # Remove any trailing incomplete string
    if text.count('"') % 2 == 1:
        # Odd number of quotes - find last complete string
        last_quote = text.rfind('"')
        # Check if this is an opening quote of an incomplete string
        prev_colon = text.rfind(':', 0, last_quote)
        prev_comma = text.rfind(',', 0, last_quote)
        if prev_colon > prev_comma:
            # Likely an incomplete value after a key - truncate to before the key
            key_start = text.rfind('"', 0, prev_colon)
            if key_start > 0:
                text = text[:key_start].rstrip().rstrip(',')

    # Count open braces and brackets
    open_braces = text.count('{') - text.count('}')
    open_brackets = text.count('[') - text.count(']')

    # Remove trailing incomplete content
    text = text.rstrip()
    while text and text[-1] not in '{}[],"0123456789truefalsn':
        text = text[:-1]

    # Handle trailing comma
    text = text.rstrip(',')

    # Close open structures
    text += ']' * open_brackets
    text += '}' * open_braces

    return text


def extract_json_from_response(text: str, allow_repair: bool = True) -> dict:
    """Extract JSON from LLM response, with optional repair for truncated responses."""
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

        # JSON appears truncated - attempt repair if enabled
        if allow_repair and depth > 0:
            json_text = text[brace_start:]
            repaired = repair_truncated_json(json_text)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not parse JSON from: {text[:300]}...")


def analyze_transcript(transcript_path: Path, model_name: str = "gemini-2.5-flash") -> dict:
    """Analyze a single transcript using the LLM.

    v3.7: Uses preprocessing to provide structured input with turn numbers.
    """
    # Preprocess transcript for deterministic turn numbers
    preprocessed = preprocess_transcript(transcript_path)
    call_id = preprocessed["call_id"]

    # Format for LLM with turn numbers
    formatted_transcript = format_for_llm(preprocessed)

    user_prompt = USER_PROMPT_TEMPLATE.format(
        schema=ANALYSIS_SCHEMA,
        transcript=formatted_transcript
    )

    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT
    )

    response = model.generate_content(
        user_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=16384,  # Increased for v3.8 agent_loops with detailed contexts
        )
    )

    # Check finish reason for debugging truncation issues
    finish_reason = None
    if response.candidates:
        finish_reason = response.candidates[0].finish_reason
        # STOP=1 is normal, MAX_TOKENS=2 indicates truncation
        if finish_reason == 2:  # MAX_TOKENS
            import sys
            print(f"  ⚠ Response truncated (MAX_TOKENS) for {call_id}", file=sys.stderr)

    analysis = extract_json_from_response(response.text)
    analysis["call_id"] = call_id
    analysis["schema_version"] = "v3.8"

    # Add preprocessing metadata for reference
    analysis["_preprocessing"] = {
        "total_turns": preprocessed["metadata"]["total_turns"],
        "user_turns": preprocessed["metadata"]["user_turns"],
        "agent_turns": preprocessed["metadata"]["agent_turns"]
    }

    return analysis


def save_analysis(analysis: dict, output_dir: Path) -> Path:
    """Save analysis JSON to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{analysis['call_id']}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Analyze call transcript (v3.8 - agent_loops replacing repeated_prompts)")
    parser.add_argument("transcript", type=Path, help="Path to transcript file")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path(__file__).parent.parent / "analyses",
                        help="Output directory for analysis JSON")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        help="Gemini model to use (default: gemini-2.5-flash)")
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
