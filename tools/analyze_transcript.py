#!/usr/bin/env python3
"""
Single Transcript Analyzer for Vacatia AI Voice Agent Analytics (v4.0)

Clean schema redesign with:
- Primary intent capture (WHAT the customer wants + WHY)
- Sentiment tracking (start → end emotional journey)
- Flattened friction structure (no more nested friction object)
- Unified disposition field (replaces outcome + call_disposition)
- Cleaner naming (shorter, more consistent field names)

v4.0 changes - Schema Redesign:
- NEW: intent, intent_context, secondary_intent (primary caller intent)
- NEW: sentiment_start, sentiment_end (emotional journey tracking)
- RENAMED: outcome → disposition (unified with call_disposition)
- RENAMED: resolution_type → resolution
- RENAMED: ended_reason → ended_by (simplified: "agent" | "customer" | "error" | "unknown")
- RENAMED: agent_effectiveness → effectiveness
- RENAMED: conversation_quality → quality
- RENAMED: customer_effort → effort
- RENAMED: failure_point → failure_type
- RENAMED: failure_description → failure_detail
- RENAMED: was_recoverable → failure_recoverable
- RENAMED: critical_failure → failure_critical
- RENAMED: repeat_caller_signals → repeat_caller
- RENAMED: additional_intents → secondary_intent
- RENAMED: customer_verbatim → verbatim
- RENAMED: agent_miss_detail → coaching
- RENAMED: resolution_steps → steps
- RENAMED: policy_gap_detail → policy_gap
- REMOVED: outcome (redundant with disposition)
- REMOVED: call_disposition (merged into disposition)
- REMOVED: training_opportunity (merged into coaching)
- PROMOTED: friction.turns → turns (top-level)
- PROMOTED: friction.derailed_at → derailed_at (top-level)
- PROMOTED: friction.clarifications → clarifications (top-level)
- PROMOTED: friction.corrections → corrections (top-level)
- PROMOTED: friction.loops → loops (top-level)
- RENAMED in arrays: t → turn/turns, ctx → note, sev → severity

Previous versions archived in tools/v3/ directory.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Load .env file if present
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

# Import preprocessing function
from preprocess_transcript import preprocess_transcript, format_for_llm


# v4.0 Schema - Clean redesign with intent, sentiment, flattened friction
ANALYSIS_SCHEMA = """
{
  "call_id": "string (UUID from filename)",
  "schema_version": "v4.0",

  // === METADATA ===
  "turns": "integer (total conversation turns - count user messages)",
  "ended_by": "enum: 'agent' | 'customer' | 'error' | 'unknown'",

  // === INTENT (NEW - captures WHAT + WHY) ===
  "intent": "string (customer's primary request in 3-8 words, normalized phrase starting with action verb)",
  "intent_context": "string or null (WHY they need it - underlying situation/reason)",
  "secondary_intent": "string or null (additional request beyond main intent)",

  // === RESOLUTION ===
  "disposition": "enum: 'pre_intent' | 'out_of_scope_handled' | 'out_of_scope_failed' | 'in_scope_success' | 'in_scope_partial' | 'in_scope_failed' | 'escalated'",
  "resolution": "string (what was accomplished: 'payment processed', 'callback scheduled', 'info provided', 'none', etc.)",
  "steps": "array of strings or null (sequence of actions taken/attempted)",

  // === QUALITY ===
  "effectiveness": "integer 1-5 (did agent understand and respond appropriately?)",
  "quality": "integer 1-5 (flow, tone, clarity combined)",
  "effort": "integer 1-5 (how hard did customer work? 1=effortless, 5=painful)",
  "sentiment_start": "enum: 'positive' | 'neutral' | 'frustrated' | 'angry'",
  "sentiment_end": "enum: 'satisfied' | 'neutral' | 'frustrated' | 'angry'",

  // === FAILURE ===
  "failure_type": "enum or null: 'nlu_miss' | 'wrong_action' | 'policy_gap' | 'customer_confusion' | 'tech_issue' | 'other'",
  "failure_detail": "string or null (one sentence: what went wrong)",
  "failure_recoverable": "boolean or null (could agent have saved this call?)",
  "failure_critical": "boolean (wrong info given, impossible promise made, compliance issue)",
  "policy_gap": "object or null - REQUIRED when failure_type='policy_gap'. Structure: { 'category': enum 'capability_limit' | 'data_access' | 'auth_restriction' | 'business_rule' | 'integration_missing', 'specific_gap': string, 'customer_ask': string, 'blocker': string }",

  // === FRICTION (flattened from friction object) ===
  "derailed_at": "integer or null (turn where call started failing; null if successful)",
  "clarifications": [
    {
      "turn": "integer (turn number)",
      "type": "enum: 'name' | 'phone' | 'intent' | 'repeat' | 'verify'",
      "cause": "enum: 'misheard' | 'unclear' | 'refused' | 'tech' | 'ok'",
      "note": "string (5-8 words, telegraph style)"
    }
  ],
  "corrections": [
    {
      "turn": "integer (turn number)",
      "severity": "enum: 'minor' | 'moderate' | 'major'",
      "note": "string (5-8 words, telegraph style)"
    }
  ],
  "loops": [
    {
      "turns": "array of integers (turn numbers involved in this loop)",
      "type": "enum: 'info_retry' | 'intent_retry' | 'deflection' | 'comprehension' | 'action_retry'",
      "subject": "string (what is being looped on)",
      "note": "string (5-8 words, telegraph style)"
    }
  ],

  // === INSIGHTS ===
  "summary": "string (1-2 sentence summary: what customer wanted, what happened, outcome)",
  "verbatim": "string or null (key quote capturing customer's core need/frustration)",
  "coaching": "string or null (what the agent should have done differently, actionable)",

  // === FLAGS ===
  "escalation_requested": "boolean (did customer ask for a human?)",
  "repeat_caller": "boolean (customer mentioned prior calls, ongoing frustration)"
}
"""

SYSTEM_PROMPT = """You are an expert call center quality analyst. Analyze this Vacatia AI voice agent call transcript and produce a focused JSON analysis using the v4.0 schema.

## Context
Vacatia is a timeshare management company. Their AI handles: RCI membership, timeshare deposits, reservations, payments, account verification.

## Transcript Format
The transcript has been preprocessed with turn numbers for your reference:
- [Turn N] AGENT: = AI voice agent
- [Turn N] USER: = Customer

Use these turn numbers EXACTLY when referencing specific turns.

## INTENT (CRITICAL - Answer "Why did customer call?")

**intent**: The customer's primary request in a normalized phrase (3-8 words).
- Start with an action verb when possible
- Be specific but concise
- Use consistent terminology (see examples)

**intent_context**: The underlying reason or situation behind the intent (or null).
- Captures WHY the customer needs this
- Includes relevant background that explains the request
- Helps understand root cause, not just surface request

### Intent Examples (WHAT + WHY)

| intent | intent_context |
|--------|----------------|
| "Log into Clubhouse portal" | "Forgot which email address is registered" |
| "Log into Clubhouse portal" | "Registration link never arrived" |
| "Check maintenance fee balance" | "Has not received invoice by mail as usual" |
| "Check maintenance fee balance" | "Wants to verify amount before paying" |
| "Make a payment" | "Received past-due notice" |
| "Make a payment" | "Setting up recurring payments for first time" |
| "Get reservation dates" | "Booking confirmation email was lost" |
| "Get reservation dates" | "Planning travel and needs exact check-in time" |
| "Speak to a representative" | "Agent could not resolve booking code issue" |
| "Speak to a representative" | "Frustrated after multiple failed verification attempts" |
| "Update contact information" | "Recently moved to new address" |
| "Cancel reservation" | null |

### Normalization Guidelines

**DO**: Use consistent verb phrases
- "Check balance" not "find out how much I owe"
- "Make a payment" not "pay my bill"
- "Log into portal" not "access my account online"
- "Get reservation dates" not "find out when my trip is"

**DON'T**: Put the WHY in the intent field
- BAD: "Check balance because invoice didn't arrive"
- GOOD: intent="Check maintenance fee balance", intent_context="Invoice didn't arrive"

## DISPOSITION Classification (replaces outcome + call_disposition)

**disposition**: Unified call outcome using this decision tree:

1. Did customer state a specific, actionable request?
   - NO → `pre_intent` (greeting-only, ≤2 turns, hung up before stating need)

2. Was call transferred to human agent?
   - YES → `escalated`

3. Could the agent handle this type of request? (See scope reference below)
   - NO + customer redirected/informed → `out_of_scope_handled`
   - NO + customer abandoned/gave up → `out_of_scope_failed`

4. Did agent complete the requested action?
   - NO → `in_scope_failed` (verification failed, system error, couldn't help)

5. Did customer express explicit satisfaction? ("thank you", "that helps", "perfect", "great")
   - YES → `in_scope_success`
   - NO → `in_scope_partial` (completed but customer just said "okay" or hung up)

**Scope Reference - IN-SCOPE (agent CAN handle):**
- Info lookup: maintenance fees, balances, due dates, property info, payment history
- Send links: Payment, Auto Pay, Account History, Rental Agreement, Clubhouse, RCI
- Account verification via phone/name/state
- Transfer to concierge (on-hours) or route to IVR callback (off-hours)

**Scope Reference - OUT-OF-SCOPE (agent CANNOT handle):**
- Process payments directly (only sends links)
- Update contact info (email, phone, address)
- Book/modify/cancel reservations
- RCI exchanges, week banking/rollover
- Live transfers when unavailable
- Complex issues: deceased owner, disputes, tax breakdown, international customers

## SENTIMENT Tracking

**sentiment_start**: Customer's mood at conversation start
**sentiment_end**: Customer's mood at conversation end

Values:
- `positive`: Happy, enthusiastic, grateful
- `neutral`: Calm, matter-of-fact, businesslike
- `frustrated`: Annoyed, impatient, exasperated
- `angry`: Explicitly upset, hostile, demanding

End sentiment for successful calls:
- `satisfied`: Explicitly pleased with outcome ("thank you", "perfect", "great")
- Use `neutral` if customer just acknowledged without emotion

## ended_by (simplified from ended_reason)

Map the metadata `ended_reason` to these simplified values:
- `assistant-ended-call` → `agent`
- `user-ended-call` → `customer`
- `error` or technical issues → `error`
- Otherwise → `unknown`

## Scoring Guidelines (1-5 scale)

**effectiveness** - Did the agent understand and respond appropriately?
- 5: Perfect understanding, ideal responses
- 3: Adequate, some minor misses
- 1: Frequent misunderstandings, poor responses

**quality** - Flow, tone, clarity combined
- 5: Natural, professional, clear
- 3: Functional but robotic or slightly awkward
- 1: Stilted, confusing, or inappropriate tone

**effort** - How hard did the customer work? (1=best)
- 1: Effortless, smooth experience
- 3: Some repetition or confusion to work through
- 5: Frustrating, excessive effort required

## FAILURE Classification

**failure_type** categories (null for successful calls):
- `nlu_miss`: Agent misunderstood what customer said/wanted
- `wrong_action`: Agent understood but took wrong action
- `policy_gap`: Agent couldn't help due to business rules/limitations
- `customer_confusion`: Customer unclear or provided wrong info
- `tech_issue`: System errors, call quality problems
- `other`: Anything else

**failure_critical**: Set to true ONLY if:
- Agent provided incorrect information
- Agent promised something impossible
- Compliance/security issue occurred
- Customer could be harmed by following agent's guidance

**policy_gap** (REQUIRED when failure_type='policy_gap'):
- `category`: capability_limit | data_access | auth_restriction | business_rule | integration_missing
- `specific_gap`: What exactly couldn't be done
- `customer_ask`: What the customer wanted
- `blocker`: Why it couldn't be fulfilled

## FRICTION Tracking (flattened - no more nested object)

### turns
Count total conversation turns (count user messages).

### derailed_at
Turn where call started failing. Null for successful calls.

### clarifications
Track EVERY time agent asks customer to clarify. Format: `{turn, type, cause, note}`

**Types**: name | phone | intent | repeat | verify
**Causes**: misheard | unclear | refused | tech | ok

### corrections
Track when CUSTOMER corrects agent's understanding. Format: `{turn, severity, note}`

**Severities**: minor | moderate | major

### loops
Detect agent friction loops (NOT benign repetition). Format: `{turns, type, subject, note}`

**Types**: info_retry | intent_retry | deflection | comprehension | action_retry

**subject values by type**:
- info_retry: name, phone, address, zip, state, account, email
- intent_retry: fee_info, balance, payment_link, autopay_link, history_link, rental_link, clubhouse_link, rci_link, transfer, callback
- deflection: anything_else, other_help, clarify_request
- comprehension: unclear_speech, background_noise, connection
- action_retry: verification, link_send, lookup, transfer_attempt

### Note Style (CRITICAL)
Use **telegraph style** (5-8 words max):
- GOOD: "re-spelled after mishearing"
- GOOD: "asked 3x despite refusal"
- BAD: "The agent asked the customer to spell their name again"

## INSIGHTS Fields

**summary**: 1-2 sentences covering: what customer wanted, what happened, outcome.

**verbatim**: Direct quote (1-2 sentences) capturing customer's emotional state or core need.

**coaching**: What the agent should have done differently. Be specific and actionable. Null if agent performed well.

## VALIDATION RULES (CRITICAL)

1. **failure_type consistency**:
   - If disposition is in_scope_failed, out_of_scope_failed, or escalated → failure_type SHOULD be populated
   - If disposition is in_scope_success → failure_type MUST be null

2. **policy_gap required**:
   - If failure_type is "policy_gap" → policy_gap MUST be a complete object with all 4 fields

3. **coaching conditional**:
   - If failure_recoverable is true → coaching SHOULD be populated with actionable guidance

4. **sentiment_end alignment**:
   - If disposition is in_scope_success → sentiment_end should typically be "satisfied" or "neutral"
   - If disposition is in_scope_failed → sentiment_end should typically be "frustrated" or "angry"

## Output Requirements
- Return ONLY valid JSON (no markdown code blocks)
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


def map_ended_reason_to_ended_by(ended_reason: str | None) -> str:
    """Map v3.9 ended_reason values to v4.0 simplified ended_by values."""
    if not ended_reason:
        return "unknown"
    ended_reason = ended_reason.lower()
    if "assistant" in ended_reason or "agent" in ended_reason:
        return "agent"
    if "user" in ended_reason or "customer" in ended_reason:
        return "customer"
    if "error" in ended_reason:
        return "error"
    return "unknown"


def analyze_transcript(transcript_path: Path, model_name: str = "gemini-3-flash-preview", client: genai.Client = None) -> dict:
    """Analyze a single transcript using the LLM.

    v4.0: Complete schema redesign with intent, sentiment, flattened friction.
    - New fields: intent, intent_context, secondary_intent, sentiment_start, sentiment_end
    - Renamed fields: outcome→disposition, resolution_type→resolution, etc.
    - Flattened: friction.* → top-level turns, derailed_at, clarifications, corrections, loops
    - Simplified: ended_reason → ended_by (agent/customer/error/unknown)
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

    # Get client if not provided
    if client is None:
        client = get_genai_client()

    # Build generation config - thinking only supported by Gemini 3 models
    config_kwargs = {
        "temperature": 0.2,
        "max_output_tokens": 16384,
        "system_instruction": SYSTEM_PROMPT,
    }

    # Only Gemini 3 models support thinking config
    if "gemini-3" in model_name:
        config_kwargs["thinking_config"] = types.ThinkingConfig(
            thinking_level="LOW"  # Fast for per-call analysis
        )

    config = types.GenerateContentConfig(**config_kwargs)

    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=config,
    )

    # Check finish reason for debugging truncation issues
    finish_reason = None
    if response.candidates:
        finish_reason = response.candidates[0].finish_reason
        # Check for truncation
        if finish_reason and "MAX_TOKENS" in str(finish_reason):
            print(f"  ⚠ Response truncated (MAX_TOKENS) for {call_id}", file=sys.stderr)

    analysis = extract_json_from_response(response.text)
    analysis["call_id"] = call_id
    analysis["schema_version"] = "v4.0"

    # v4.0: Map ended_reason from preprocessing to simplified ended_by
    ended_reason = preprocessed.get("metadata", {}).get("ended_reason")
    analysis["ended_by"] = map_ended_reason_to_ended_by(ended_reason)

    # v4.0: Ensure arrays exist even if empty (for consistent parsing downstream)
    if "clarifications" not in analysis:
        analysis["clarifications"] = []
    if "corrections" not in analysis:
        analysis["corrections"] = []
    if "loops" not in analysis:
        analysis["loops"] = []
    if "steps" not in analysis:
        analysis["steps"] = []

    return analysis


def save_analysis(analysis: dict, output_dir: Path) -> Path:
    """Save analysis JSON to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{analysis['call_id']}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Analyze call transcript (v4.0 - Schema redesign with intent & sentiment)")
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
        client = get_genai_client()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"Analyzing: {args.transcript}", file=sys.stderr)

    try:
        analysis = analyze_transcript(args.transcript, args.model, client)
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
