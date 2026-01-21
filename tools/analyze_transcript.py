#!/usr/bin/env python3
"""
Single Transcript Analyzer for Vacatia AI Voice Agent Analytics (v3.9.2)

Hybrid schema with streamlined friction tracking enabling both programmatic analysis
and executive-ready insights through a two-part report architecture.

v3.9.2 changes - JSON Transcript Support:
- Support for new JSON transcript format with timestamps and metadata
- Pass through `ended_reason` from preprocessing to analysis output
- Include `ended_reason` context in LLM prompt for better outcome classification
- Values: assistant-ended-call, user-ended-call, error, etc.

v3.9.1 changes - Loop Subject Granularity:
- New `subject` field in friction.loops identifying WHAT is being looped on
- Guided values by loop type (info_retry: name, phone, etc.; intent_retry: fee_info, etc.)
- Freeform fallback for edge cases not in the shortlist
- Enables granular analysis of friction patterns

v3.9 changes - Call Disposition Classification:
- New call_disposition field for funnel analysis
- Values: pre_intent, out_of_scope_handled, out_of_scope_abandoned, in_scope_success, in_scope_partial, in_scope_failed
- Decision tree for classification based on customer intent, agent scope, and completion status
- Scope reference for in-scope vs out-of-scope request classification

v3.8.5 changes - Streamlined Friction Tracking:
- Consolidated friction fields into single compact `friction` object
- Shorter enum values (name vs name_spelling, phone vs phone_confirmation, etc.)
- Compact keys: t (turn), ctx (context), sev (severity)
- Telegraph-style context strings (5-8 words max)
- Loops now include turn numbers in `t` array
- Removed: _preprocessing from output, resolved boolean, what_was_wrong
- ~31% output size reduction (2,906 → ~2,000 bytes/call)

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

# Load .env file if present
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

# Import preprocessing function
from preprocess_transcript import preprocess_transcript, format_for_llm


# v3.9.2 Schema - Hybrid metrics + streamlined friction tracking + ended_reason
ANALYSIS_SCHEMA = """
{
  "call_id": "string (UUID from filename)",

  "outcome": "enum: 'resolved' | 'escalated' | 'abandoned' | 'unclear'",
  "resolution_type": "string (what was accomplished: 'payment processed', 'callback scheduled', 'info provided', 'none', etc.)",
  "ended_reason": "string or null (from transcript metadata: 'assistant-ended-call', 'user-ended-call', 'error', etc.)",

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

  "call_disposition": "enum: 'pre_intent' | 'out_of_scope_handled' | 'out_of_scope_abandoned' | 'in_scope_success' | 'in_scope_partial' | 'in_scope_failed'",

  "policy_gap_detail": "object or null - REQUIRED when failure_point='policy_gap'. Structure: { 'category': enum 'capability_limit' | 'data_access' | 'auth_restriction' | 'business_rule' | 'integration_missing', 'specific_gap': string (what exactly couldn't be done), 'customer_ask': string (what the customer wanted), 'blocker': string (why it couldn't be fulfilled) }",

  "customer_verbatim": "string or null (key quote from customer capturing their core frustration, need, or emotional state. Direct transcript excerpt, 1-2 sentences max. Example: 'I've called three times about this and no one can help me')",

  "agent_miss_detail": "string or null (what the agent should have said or done differently. Actionable coaching insight. Example: 'Should have offered callback scheduling when customer expressed time constraints instead of continuing verification loop')",

  "resolution_steps": "array of strings or null (sequence of actions taken or attempted during the call. Example: ['greeted customer', 'attempted verification via phone', 'verification failed', 'offered alternative ID method', 'customer declined', 'offered human escalation', 'customer accepted'])",

  "friction": {
    "turns": "integer (total conversation turns - count user messages)",
    "derailed_at": "integer or null (turn where call started failing; null if resolved)",

    "clarifications": [
      {
        "t": "integer (turn number)",
        "type": "enum: 'name' | 'phone' | 'intent' | 'repeat' | 'verify'",
        "cause": "enum: 'misheard' | 'unclear' | 'refused' | 'tech' | 'ok'",
        "ctx": "string (5-8 words, telegraph style)"
      }
    ],

    "corrections": [
      {
        "t": "integer (turn number)",
        "sev": "enum: 'minor' | 'moderate' | 'major'",
        "ctx": "string (5-8 words, telegraph style)"
      }
    ],

    "loops": [
      {
        "t": "array of integers (turn numbers involved in this loop)",
        "type": "enum: 'info_retry' | 'intent_retry' | 'deflection' | 'comprehension' | 'action_retry'",
        "subject": "string (what is being looped on, see subject values by loop type)",
        "ctx": "string (5-8 words, telegraph style)"
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

## Using ended_reason Context (v3.9.2)
If `Ended Reason:` is provided in the transcript header, use it to inform your outcome classification:
- **assistant-ended-call**: Agent initiated call end - often indicates resolved or escalated
- **user-ended-call**: Customer hung up - could be abandoned, resolved (satisfied), or unclear
- **error**: Technical issue - likely abandoned or unclear
- Don't rely solely on ended_reason - analyze the conversation to determine actual outcome

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

## Call Disposition Classification (v3.9)

**call_disposition**: Classify the overall call outcome using this decision tree:

1. Did customer state a specific, actionable request?
   - NO → `pre_intent` (greeting-only, ≤2 turns, hung up before stating need)

2. Could the agent handle this type of request? (See scope reference below)
   - NO + customer redirected/informed → `out_of_scope_handled`
   - NO + customer abandoned/escalated → `out_of_scope_abandoned`

3. Did agent complete the requested action?
   - NO → `in_scope_failed` (verification failed, system error, couldn't help)

4. Did customer express explicit satisfaction? ("thank you", "that helps", "perfect", "great")
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

## Friction Tracking (v3.8.5)

Track ALL conversation friction in the compact `friction` object.

### friction.turns
Count total conversation turns (count user messages). Example: 15-turn call = 15 user messages.

### friction.derailed_at
For NON-RESOLVED calls only: turn where call started failing. Null for resolved calls.

### friction.clarifications
Track EVERY time agent asks customer to clarify. Compact format: `{t, type, cause, ctx}`

**Types** (SHORT enum values):
- `name`: Agent asks to spell name
- `phone`: Agent confirms phone number
- `intent`: Agent asks what customer needs
- `repeat`: Agent asks to repeat
- `verify`: Agent asks for different verification info

**Causes** (SHORT enum values):
- `misheard`: Agent failed to parse what customer said
- `unclear`: Customer provided but unclear
- `refused`: Customer declined to provide info
- `tech`: Audio/connection problem
- `ok`: Clarification succeeded

### friction.corrections
Track when CUSTOMER corrects agent's understanding. Compact format: `{t, sev, ctx}`

**Severities**:
- `minor`: Simple correction, no frustration
- `moderate`: Correction with mild frustration
- `major`: Explicit anger or critical mistake

### friction.loops (v3.9.1)
Detect agent friction loops (NOT benign repetition). Compact format: `{t, type, subject, ctx}`

**t field**: Array of turn numbers involved in this loop pattern.

**Types**:
- `info_retry`: Re-asks for info already provided (name, phone asked twice)
- `intent_retry`: Re-asks for intent already stated (common after identity check)
- `deflection`: Generic questions while unable to help primary request
- `comprehension`: Agent couldn't hear, asks to repeat
- `action_retry`: System retry ("let me try that again")

**subject field (v3.9.1)**: Identifies WHAT is being looped on. Use guided values when they fit:

**info_retry** (verification data):
- `name`, `phone`, `address`, `zip`, `state`, `account`, `email`

**intent_retry** (customer intents):
- `fee_info`, `balance`, `payment_link`, `autopay_link`, `history_link`
- `rental_link`, `clubhouse_link`, `rci_link`, `transfer`, `callback`

**deflection** (stalling questions):
- `anything_else`, `other_help`, `clarify_request`

**comprehension** (audio issues):
- `unclear_speech`, `background_noise`, `connection`

**action_retry** (system retries):
- `verification`, `link_send`, `lookup`, `transfer_attempt`

For edge cases not in these lists, use descriptive lowercase_with_underscores.

**EXCLUDE** (benign, don't track):
- Greeting after hold
- Re-engagement after silence
- Compliance disclosures
- Confirmation before action

### Context Style Guide (CRITICAL)

Use **telegraph style** for ctx fields (5-8 words max):
- Drop articles ("the", "a")
- Use action verbs
- Focus on WHAT happened

**Examples:**
- GOOD: "re-spelled after mishearing"
- GOOD: "confirmed number"
- GOOD: "name asked 3x despite refusal"
- BAD: "The agent asked the customer to spell their name again" (too long)
- BAD: "There was a clarification" (too vague)

### Example friction object

```json
"friction": {
  "turns": 25,
  "derailed_at": 5,
  "clarifications": [
    {"t": 5, "type": "name", "cause": "misheard", "ctx": "re-spelled after mishearing"},
    {"t": 11, "type": "phone", "cause": "ok", "ctx": "confirmed number"}
  ],
  "corrections": [
    {"t": 21, "sev": "moderate", "ctx": "corrected misspelled name"}
  ],
  "loops": [
    {"t": [20, 22, 25], "type": "info_retry", "subject": "name", "ctx": "asked 3x despite refusal"}
  ]
}
```

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


def analyze_transcript(transcript_path: Path, model_name: str = "gemini-3-flash-preview", client: genai.Client = None) -> dict:
    """Analyze a single transcript using the LLM.

    v3.7: Uses preprocessing to provide structured input with turn numbers.
    v3.8.6: Uses new google.genai SDK with thinking config.
    v3.9: Adds call_disposition field for funnel analysis.
    v3.9.1: Adds subject field to friction loops for granular analysis.
    v3.9.2: Supports JSON transcripts with timestamps; passes through ended_reason.
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

    # Build generation config with thinking
    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=16384,  # Increased to accommodate thinking model overhead
        thinking_config=types.ThinkingConfig(
            thinking_level="LOW"  # Fast for per-call analysis (MEDIUM for insights/review)
        ),
        system_instruction=SYSTEM_PROMPT,
    )

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
    analysis["schema_version"] = "v3.9.2"

    # v3.9.2: Pass through ended_reason from preprocessing metadata
    ended_reason = preprocessed.get("metadata", {}).get("ended_reason")
    if ended_reason:
        analysis["ended_reason"] = ended_reason

    # v3.8.5: No _preprocessing in output (bloat reduction)
    # Turn data is available in friction.turns

    return analysis


def save_analysis(analysis: dict, output_dir: Path) -> Path:
    """Save analysis JSON to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{analysis['call_id']}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Analyze call transcript (v3.9.2 - JSON format support)")
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
