#!/usr/bin/env python3
"""
Single Transcript Analyzer for Vacatia AI Voice Agent Analytics (v5.0)

v5.0 changes - Disposition Model Redesign:
- REPLACED: single `disposition` enum → `call_scope` + `call_outcome` (orthogonal dimensions)
- REPLACED: `escalation_initiator` → `escalation_trigger` (why, not who)
- REPLACED: `pre_intent_subtype` → `abandon_stage` (generalized to all abandons)
- REMOVED: `abandoned_path_viable` (covered by call_scope + abandon_stage)
- REMOVED: `escalation_requested` (covered by escalation_trigger = customer_requested)
- REMOVED: `failure_recoverable` (covered by failure_type + outcome)
- REMOVED: `failure_critical` (rare, covered by coaching)
- KEPT: resolution_confirmed (bool, only when call_outcome = completed)
- KEPT: actions[], transfer_destination, transfer_queue_detected

Previous schema versions: v4.5 (dashboard fields), v4.4 (handle time), v4.0 (schema redesign).
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


# v5.0 Schema - Orthogonal disposition model: call_scope × call_outcome
ANALYSIS_SCHEMA = """
{
  "call_id": "string (UUID from filename)",
  "schema_version": "v5.0",

  "turns": "integer (total conversation turns - count user messages)",
  "ended_by": "enum: 'agent' | 'customer' | 'error' | 'unknown'",

  "intent": "string (customer's primary request in 3-8 words, normalized phrase starting with action verb)",
  "intent_context": "string or null (WHY they need it - underlying situation/reason)",
  "secondary_intent": "string or null (additional request beyond main intent)",

  "call_scope": "enum: 'in_scope' | 'out_of_scope' | 'mixed' | 'no_request'",
  "call_outcome": "enum: 'completed' | 'escalated' | 'abandoned'",
  "resolution": "string (what was accomplished: 'info provided', 'link sent', 'transferred to concierge', 'none', etc.)",
  "steps": "array of strings or null (sequence of actions taken/attempted)",

  "escalation_trigger": "enum or null: 'customer_requested' | 'scope_limit' | 'task_failure' | 'policy_routing' (required when call_outcome = escalated, null otherwise)",
  "abandon_stage": "enum or null: 'pre_greeting' | 'pre_intent' | 'mid_task' | 'post_delivery' (required when call_outcome = abandoned, null otherwise)",
  "resolution_confirmed": "boolean or null (true=customer explicitly confirmed, false=action taken but unconfirmed, null when call_outcome != completed)",

  "effectiveness": "integer 1-5 (did agent understand and respond appropriately?)",
  "quality": "integer 1-5 (flow, tone, clarity combined)",
  "effort": "integer 1-5 (how hard did customer work? 1=effortless, 5=painful)",
  "sentiment_start": "enum: 'positive' | 'neutral' | 'frustrated' | 'angry'",
  "sentiment_end": "enum: 'satisfied' | 'neutral' | 'frustrated' | 'angry'",

  "failure_type": "enum or null: 'nlu_miss' | 'wrong_action' | 'policy_gap' | 'customer_confusion' | 'tech_issue' | 'other'",
  "failure_detail": "string or null (one sentence: what went wrong)",
  "policy_gap": "object or null - REQUIRED when failure_type='policy_gap'. Structure: { 'category': enum, 'specific_gap': string, 'customer_ask': string, 'blocker': string }",

  "derailed_at": "integer or null (turn where call started failing; null if successful)",
  "clarifications": [
    {
      "turn": "integer",
      "type": "enum: 'name' | 'phone' | 'intent' | 'repeat' | 'verify'",
      "cause": "enum: 'misheard' | 'unclear' | 'refused' | 'tech' | 'ok'",
      "note": "string (5-8 words, telegraph style)"
    }
  ],
  "corrections": [
    {
      "turn": "integer",
      "severity": "enum: 'minor' | 'moderate' | 'major'",
      "note": "string (5-8 words, telegraph style)"
    }
  ],
  "loops": [
    {
      "turns": "array of integers",
      "type": "enum: 'info_retry' | 'intent_retry' | 'deflection' | 'comprehension' | 'action_retry'",
      "subject": "string",
      "note": "string (5-8 words, telegraph style)"
    }
  ],

  "summary": "string (1-2 sentence summary: what customer wanted, what happened, outcome)",
  "verbatim": "string or null (key quote capturing customer's core need/frustration)",
  "coaching": "string or null (what the agent should have done differently, actionable)",

  "repeat_caller": "boolean (customer mentioned prior calls, ongoing frustration)",

  "actions": [
    {
      "action": "enum: 'account_lookup' | 'verification' | 'send_payment_link' | 'send_portal_link' | 'send_autopay_link' | 'send_rental_link' | 'send_clubhouse_link' | 'send_rci_link' | 'transfer' | 'other'",
      "outcome": "enum: 'success' | 'failed' | 'retry' | 'unknown'",
      "detail": "string (5-10 words, telegraph style)"
    }
  ],
  "transfer_destination": "string or null ('concierge' | 'specific_department' | 'ivr' | 'unknown' | null if no transfer)",
  "transfer_queue_detected": "boolean (true if transcript shows post-transfer queue content)"
}
"""

SYSTEM_PROMPT = """You are an expert call center quality analyst. Analyze this Vacatia AI voice agent call transcript and produce a focused JSON analysis using the v5.0 schema.

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

**intent_context**: The underlying reason or situation behind the intent (or null).

**secondary_intent**: Additional request beyond main intent (or null).

### Intent Examples

| intent | intent_context |
|--------|----------------|
| "Check maintenance fee balance" | "Has not received invoice by mail as usual" |
| "Make a payment" | "Received past-due notice" |
| "Log into Clubhouse portal" | "Registration link never arrived" |
| "Speak to a representative" | "Agent could not resolve booking code issue" |
| "Cancel reservation" | null |

**DO**: "Check balance" not "find out how much I owe". Put the WHY in intent_context, not intent.

## CALL SCOPE — Could the AI handle this request?

Evaluate the customer's request(s) against the agent's capabilities. This is independent of what actually happened on the call.

**call_scope values:**

- `in_scope` — ALL of the customer's requests fall within AI capabilities
- `out_of_scope` — ALL requests require human judgment or systems the AI lacks
- `mixed` — Multiple intents where at least one is in-scope and at least one is out-of-scope
- `no_request` — Customer never articulated a specific request (hung up, silence, greeting-only)

**Scope Reference — IN-SCOPE (agent CAN handle):**
- Info lookup: maintenance fees, balances, due dates, property info, payment history
- Send links: Payment, Auto Pay, Account History, Rental Agreement, Clubhouse, RCI
- Account verification via phone/name/state
- Transfer to concierge (on-hours) or route to IVR callback (off-hours)

**Scope Reference — OUT-OF-SCOPE (agent CANNOT handle):**
- Process payments directly (only sends links)
- Update contact info (email, phone, address)
- Book/modify/cancel reservations
- RCI exchanges, week banking/rollover
- Sell/transfer timeshare ownership
- Complex issues: deceased owner, disputes, tax breakdown, international customers

## CALL OUTCOME — What actually happened?

**call_outcome values:**

- `completed` — AI agent finished its task flow (looked up info, sent link, verified account, etc.)
- `escalated` — Call was transferred to a human agent
- `abandoned` — Customer disconnected or call ended without completion or transfer

### escalation_trigger (REQUIRED when call_outcome = escalated, null otherwise)
- `customer_requested` — Customer explicitly asked for a human/live agent/manager
- `scope_limit` — AI recognized the request as beyond its capabilities and initiated transfer
- `task_failure` — AI was attempting an in-scope task but hit an unrecoverable error and escalated
- `policy_routing` — Business rules require human handling for this request type

### abandon_stage (REQUIRED when call_outcome = abandoned, null otherwise)
- `pre_greeting` — Disconnected before or during AI greeting
- `pre_intent` — Disconnected after greeting but before stating a request
- `mid_task` — Disconnected while AI was working on request
- `post_delivery` — Disconnected after AI delivered info/action but before explicit confirmation

### resolution_confirmed (only when call_outcome = completed, null otherwise)
- true: Customer EXPLICITLY acknowledged completion ("got it", "I see the link", "thanks that worked")
- false: Agent took action but customer did NOT explicitly confirm receipt/completion

## SENTIMENT Tracking

**sentiment_start**: Customer's mood at conversation start
**sentiment_end**: Customer's mood at conversation end

Values: `positive` | `neutral` | `frustrated` | `angry`
End-only value: `satisfied` (explicitly pleased with outcome)

## ended_by

Map the metadata `ended_reason` to: `agent` | `customer` | `error` | `unknown`

## Scoring Guidelines (1-5 scale)

**effectiveness** — Did the agent understand and respond appropriately? (5=perfect, 1=poor)
**quality** — Flow, tone, clarity combined (5=natural, 1=stilted)
**effort** — How hard did the customer work? (1=effortless, 5=painful)

## FAILURE Classification

**failure_type** (null when call_outcome = completed with no issues):
- `nlu_miss`: Agent misunderstood what customer said/wanted
- `wrong_action`: Agent understood but took wrong action
- `policy_gap`: Agent couldn't help due to business rules/limitations
- `customer_confusion`: Customer unclear or provided wrong info
- `tech_issue`: System errors, call quality problems
- `other`: Anything else

**policy_gap** (REQUIRED when failure_type='policy_gap'):
- `category`: capability_limit | data_access | auth_restriction | business_rule | integration_missing
- `specific_gap`: What exactly couldn't be done
- `customer_ask`: What the customer wanted
- `blocker`: Why it couldn't be fulfilled

## FRICTION Tracking

**derailed_at**: Turn where call started failing. Null if successful.

**clarifications**: Every time agent asks customer to clarify. `{turn, type, cause, note}`
Types: name | phone | intent | repeat | verify
Causes: misheard | unclear | refused | tech | ok

**corrections**: When customer corrects agent. `{turn, severity, note}`
Severities: minor | moderate | major

**loops**: Agent friction loops (NOT benign repetition). `{turns, type, subject, note}`
Types: info_retry | intent_retry | deflection | comprehension | action_retry

**Note style**: Telegraph, 5-8 words max. "re-spelled after mishearing", NOT "The agent asked the customer to spell their name again"

## INSIGHTS

**summary**: 1-2 sentences: what customer wanted, what happened, outcome.
**verbatim**: Direct quote capturing customer's emotional state or core need.
**coaching**: What agent should have done differently. Null if agent performed well.

## ACTIONS

Track EVERY distinct tool/action the agent attempted:
- account_lookup, verification, send_payment_link, send_portal_link, send_autopay_link, send_rental_link, send_clubhouse_link, send_rci_link, transfer, other
- Outcome: success | failed | retry | unknown
- Detail: 5-10 words telegraph style
- List each attempt separately if same action tried multiple times.

**transfer_destination**: concierge | specific_department | ivr | unknown | null
**transfer_queue_detected**: true if post-transfer queue evidence in transcript

## VALIDATION RULES (CRITICAL)

1. **call_scope + call_outcome independence**: Evaluate scope BEFORE looking at outcome. A call can be in_scope + escalated, or out_of_scope + abandoned.

2. **Conditional field consistency**:
   - call_outcome = escalated → escalation_trigger MUST be non-null
   - call_outcome = abandoned → abandon_stage MUST be non-null
   - call_outcome = completed → resolution_confirmed MUST be true or false (not null)
   - call_outcome != escalated → escalation_trigger MUST be null
   - call_outcome != abandoned → abandon_stage MUST be null
   - call_outcome != completed → resolution_confirmed MUST be null

3. **Invalid combinations**:
   - no_request + completed is INVALID (can't complete what was never requested)
   - failure_type must be null when call_outcome = completed with no issues

4. **failure_type consistency**:
   - When call_outcome = escalated with escalation_trigger = task_failure → failure_type SHOULD be populated
   - When call_outcome = completed successfully → failure_type MUST be null

5. **policy_gap required**: If failure_type = policy_gap → policy_gap object MUST have all 4 fields

6. **sentiment alignment**:
   - call_outcome = completed + resolution_confirmed = true → sentiment_end typically "satisfied" or "neutral"

## Output Requirements
- Return ONLY valid JSON (no markdown code blocks)
- Be concise and actionable
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

    v5.0: Orthogonal disposition model (call_scope × call_outcome).
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
    analysis["schema_version"] = "v5.0"

    # Inject duration_seconds from raw transcript (deterministic, not LLM-extracted)
    raw_duration = preprocessed.get("metadata", {}).get("duration")
    analysis["duration_seconds"] = round(raw_duration, 1) if raw_duration is not None else None

    # Map ended_reason from preprocessing to simplified ended_by
    ended_reason = preprocessed.get("metadata", {}).get("ended_reason")
    analysis["ended_by"] = map_ended_reason_to_ended_by(ended_reason)

    # Ensure arrays exist even if empty
    for arr_field in ("clarifications", "corrections", "loops", "steps", "actions"):
        if arr_field not in analysis:
            analysis[arr_field] = []
    if "transfer_queue_detected" not in analysis:
        analysis["transfer_queue_detected"] = False

    # v5.0: Enforce conditional field consistency based on call_outcome
    outcome = analysis.get("call_outcome")
    if outcome != "escalated":
        analysis["escalation_trigger"] = None
    if outcome != "abandoned":
        analysis["abandon_stage"] = None
    if outcome != "completed":
        analysis["resolution_confirmed"] = None

    # Clean up any deprecated fields the LLM might still produce
    for old_field in ("disposition", "escalation_initiator", "pre_intent_subtype",
                      "abandoned_path_viable", "escalation_requested",
                      "failure_recoverable", "failure_critical"):
        analysis.pop(old_field, None)

    return analysis


def save_analysis(analysis: dict, output_dir: Path) -> Path:
    """Save analysis JSON to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{analysis['call_id']}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Analyze call transcript (v5.0 - Orthogonal scope × outcome model)")
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
