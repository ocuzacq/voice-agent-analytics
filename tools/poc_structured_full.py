#!/usr/bin/env python3
"""
PoC: Test the v6.0 per-intent resolution schema with Gemini structured output.

Validates:
  1. Per-intent resolution: primary always present, secondary optional
  2. TransferDetail nested inside IntentResolution
  3. Conditional fields per outcome (fulfilled→confirmed, transferred→transfer, abandoned→stage)
  4. All other sub-models (Scores, Sentiment, Failure, Friction, etc.)

Usage:
    source .env
    python3 tools/poc_structured_full.py tests/golden/transcripts/0a434683-1ac4-4f26-83ee-c6fce1040ef4.json
    python3 tools/poc_structured_full.py tests/golden/transcripts/6940a110-24bc-4645-8b1b-a52b05878dbf.json
    python3 tools/poc_structured_full.py tests/golden/transcripts/00a05f6e-f144-4c37-97d4-fe7d22a3211b.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_transcript import preprocess_transcript, format_for_llm
from schema import CallAnalysis, CallRecord, SCHEMA_VERSION


SYSTEM_PROMPT = """You are an expert call center quality analyst. Analyze this Vacatia AI voice agent call transcript.

## Context
Vacatia is a timeshare management company. Their AI handles: RCI membership, timeshare deposits, reservations, payments, account verification.
Timeshare verification reality: callers are frequently NOT the name on the contract — spouses,
children, siblings, and parents commonly call about the same property. A name mismatch where
the lookup itself succeeded is NOT a tech failure.

The AI agent fulfills most in-scope requests by SENDING LINKS (payment links, portal links,
autopay links, rental links, RCI links) via text message or email. It does not process
transactions directly — the caller receives a link and completes the action themselves.

These text/email deliveries are not always reliable. Wrong number on file, carrier blocking,
email typos, or other delivery failures can cause friction, retries, and potentially escalation
or abandonment — even when the intent is fully in-scope and the AI has the capability.

To reduce friction, the AI attempts automatic account lookup via callerID. For routine
requests that don't require extra precaution, this lets the agent skip name/identity
verification entirely and proceed directly to fulfilling the request.

## Transcript Format
- [Turn N] AGENT: = AI voice agent
- [Turn N] USER: = Customer
Use these turn numbers EXACTLY when referencing specific turns.

## Resolution — Per-Intent Structure

Each intent is independently resolved with its own scope, outcome, and details.

### scope — Does the AI have a capability pathway for this request?

Scope reflects what the AI CAN do, not what happened on this call.
If the AI has a defined pathway for the request, it is in_scope — even if the call
fails mid-flow (verification error, tech issue, etc.). A mid-flow failure changes the
outcome or failure type, NOT the scope.

**in_scope** — AI has a defined capability:
- Payment links      — Pay maintenance fees, balance, etc. AI sends payment link via text/email
- Portal/account     — Account access, login help. AI sends portal or autopay links
- RCI membership     — RCI inquiries, activation. AI sends RCI links
- Rental program     — Rent out timeshare week. AI sends rental link
- Info lookup        — Balance, property details, contract info
- Verification       — Confirm identity (name, state, contract ID)

**out_of_scope** — No AI pathway, requires human agent:
- Payment processing — Accepting card numbers in-conversation (Live human agents can do that, no this AI agent), processing refunds, payment arrangements
- Account changes    — Update contact info, address, phone, email
- Reservations       — Book, change, or cancel stays
- Ownership          — Sell, transfer, or surrender timeshare
- Billing disputes   — Adjustments, credits, fee waivers
- Complex issues     — Legal, escalated complaints, multi-property

**no_request** — Caller never articulated a request (pre-greeting/pre-intent abandon only)

### human_requested — when caller asks for a human

human_requested captures both WHETHER and WHEN the caller asked for a human:
- null: Caller never asked for a human agent
- "initial": Caller asked for a human BEFORE the AI attempted substantive service
- "after_service": Caller asked AFTER the AI began substantive service (preventable escalation)

Substantive service = account_lookup, sending links, delivering information, processing requests.
Verification / name collection alone does NOT count as substantive service.

Set department_requested to the specific department/team name when the caller asks
for one by name (e.g., "finance", "billing", "accounting", "sales", "reservations",
"management"). Leave null for generic requests ("a representative", "a person",
"customer service", "concierge").

Scope reflects the UNDERLYING customer need, not the transfer action or department:
- "Pay my fees" + "speak to a person" from the start → scope=in_scope, human_requested="initial"
- "Pay my fees" + "speak to finance" from the start → scope=in_scope, human_requested="initial", department_requested="finance"
- "Pay my fees" → AI loops → "just give me someone" → scope=in_scope, human_requested="after_service"
- "Billing dispute" + "speak to billing" → scope=out_of_scope, human_requested="initial", department_requested="billing"
- "Give me a representative" (no stated need) → scope=out_of_scope, human_requested="initial"
- Caller asks for human but AI fulfills the need → scope=in_scope, outcome=fulfilled, human_requested="initial"

"Customer service" and "concierge" are the generic agent pool, not specific departments.

### outcome
- fulfilled includes: info provided, link sent, question answered
- fulfilled requires observable completion: for link/text/email delivery, the agent must confirm
  the send action ("I've sent it", "You should receive..."). Confirming the destination address
  alone is preparation, not completion. If the call ends before send confirmation, classify as
  abandoned with abandon_stage=mid_task.
- transferred is CORRECT behavior for out-of-scope requests — not a failure

### Secondary intent — when to split
Only create a secondary intent for a SEPARATE customer request, not for sub-steps within the
primary request. Clarifications, verifications, and troubleshooting that serve the primary goal
are steps[], not a second intent. Test: would the customer describe this as "I called about TWO
things" or "I called about one thing that had complications"?

### Key principle: transfer is neutral, not failure
An out-of-scope call correctly transferred to concierge is GOOD routing.
Only set failure when something actually went WRONG during the call.

### Transfer outcomes — classify what the transcript shows
Always populate the transfer object when a transfer is attempted, regardless of final outcome.
The outcome depends on WHERE the transcript ends relative to the transfer:

- outcome=transferred: The LAST substantive content is the transfer handoff or queue hold messages.
  The caller was still waiting/connected when the transcript ends. No post-queue conversation with the AI.
- outcome=abandoned: The caller disconnects AFTER a transfer fails or stalls. Key signals:
  the AI re-engages post-queue ("no representatives available"), caller gives up, or call cuts off
  during/after a failed transfer attempt. Set abandon_stage=mid_task.

Principle: if the AI agent speaks again after the queue (not just hold music), the transfer did not
succeed — classify based on what happened next. Both transfer and abandon_stage can coexist.

## Sentiment
- satisfied (end only) = customer explicitly pleased with outcome, not just neutral

## Failure guidance
A name not matching the record is NOT tech_issue if the lookup itself succeeded.
wrong_action evaluates the agent's decisions, not the call's outcome. A successful transfer does
not erase wrong actions taken before it (e.g., unnecessary verification on an explicit escalation
request). Examples: verifying identity for an out-of-scope request that should have been
transferred immediately, or persisting with name verification when the caller is likely
a family member.
Queue unavailability (no agents available after hold) is NOT a failure of the AI agent — the AI
correctly routed the call. But evaluate the agent's behavior across the ENTIRE call: unnecessary
loops, delayed escalation, or missed intent before the transfer can still be wrong_action even
when the transfer itself was correct.

## Actions
Track EVERY distinct tool/action attempted. List each attempt separately if retried.

## Validation Rules
1. Evaluate scope per-intent independently
2. transfer can coexist with abandon_stage when a transfer was attempted but caller abandoned
"""


def run_test(transcript_path: Path, model_name: str = "gemini-3-flash-preview") -> None:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GOOGLE_API_KEY or GEMINI_API_KEY", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    preprocessed = preprocess_transcript(transcript_path)
    call_id = preprocessed["call_id"]
    formatted = format_for_llm(preprocessed)

    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=16384,
        system_instruction=SYSTEM_PROMPT,
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        response_mime_type="application/json",
        response_schema=CallAnalysis,
    )

    print(f"Model:      {model_name}")
    print(f"Schema:     CallAnalysis {SCHEMA_VERSION} (per-intent resolution)")
    print(f"Transcript: {call_id}")
    print("---")

    t0 = time.time()
    response = client.models.generate_content(
        model=model_name,
        contents=f"Analyze this call transcript:\n\n{formatted}",
        config=config,
    )
    elapsed = time.time() - t0

    print(f"\n=== API Response ({elapsed:.1f}s) ===\n")

    if response.candidates:
        print(f"Finish reason: {response.candidates[0].finish_reason}")

    if hasattr(response, 'parsed') and response.parsed is not None:
        analysis: CallAnalysis = response.parsed

        record = CallRecord(
            **analysis.model_dump(),
            call_id=call_id,
            schema_version=SCHEMA_VERSION,
            duration_seconds=round(preprocessed.get("metadata", {}).get("duration", 0), 1) or None,
            ended_by=_map_ended_reason(preprocessed.get("metadata", {}).get("ended_reason")),
        )

        print(f"response.parsed: {type(analysis).__name__}")
        print(f"\n--- Full CallRecord ---\n")
        print(json.dumps(record.model_dump(), indent=2))

        # Resolution summary
        r = analysis.resolution
        print(f"\n--- Resolution Summary ---")
        print(f"  PRIMARY:  [{r.primary.scope}] {r.primary.request}")
        if r.primary.human_requested:
            dept = f"  dept={r.primary.department_requested}" if r.primary.department_requested else ""
            print(f"            human_requested={r.primary.human_requested}{dept}")
        print(f"            outcome={r.primary.outcome}", end="")
        if r.primary.outcome == "fulfilled":
            print(f"  confirmed={r.primary.resolution_confirmed}")
        elif r.primary.outcome == "transferred":
            t = r.primary.transfer
            print(f"  reason={t.reason}  dest={t.destination}  queue={t.queue_detected}")
        elif r.primary.outcome == "abandoned":
            print(f"  stage={r.primary.abandon_stage}")
        else:
            print()

        if r.secondary:
            print(f"  SECONDARY: [{r.secondary.scope}] {r.secondary.request}")
            if r.secondary.human_requested:
                dept = f"  dept={r.secondary.department_requested}" if r.secondary.department_requested else ""
                print(f"            human_requested={r.secondary.human_requested}{dept}")
            print(f"            outcome={r.secondary.outcome}", end="")
            if r.secondary.outcome == "fulfilled":
                print(f"  confirmed={r.secondary.resolution_confirmed}")
            elif r.secondary.outcome == "transferred":
                t = r.secondary.transfer
                print(f"  reason={t.reason}  dest={t.destination}")
            elif r.secondary.outcome == "abandoned":
                print(f"  stage={r.secondary.abandon_stage}")
            else:
                print()
        else:
            print(f"  SECONDARY: none")

        # Other checks
        print(f"\n--- Other Fields ---")
        print(f"  failure:     {analysis.failure.type + ': ' + analysis.failure.detail if analysis.failure else 'null'}")
        print(f"  derailed_at: {analysis.derailed_at}")
        print(f"  scores:      eff={analysis.scores.effectiveness} qual={analysis.scores.quality} eff={analysis.scores.effort}")
        print(f"  sentiment:   {analysis.sentiment.start} → {analysis.sentiment.end}")
        print(f"  actions:     {len(analysis.actions)}")
        for i, a in enumerate(analysis.actions):
            print(f"    [{i}] {a.type} → {a.outcome}: {a.detail}")
        print(f"  friction:    {len(analysis.friction.clarifications)}c {len(analysis.friction.corrections)}x {len(analysis.friction.loops)}l")

    else:
        print("response.parsed is None — trying manual parse")
        raw = response.text
        print(raw[:3000])
        try:
            data = json.loads(raw)
            analysis = CallAnalysis.model_validate(data)
            print("\nManual parse succeeded")
            print(json.dumps(analysis.model_dump(), indent=2))
        except Exception as e:
            print(f"\nManual parse failed: {e}")

    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        u = response.usage_metadata
        print(f"\n--- Tokens ---")
        print(f"  prompt={getattr(u, 'prompt_token_count', '?')}  response={getattr(u, 'candidates_token_count', '?')}  total={getattr(u, 'total_token_count', '?')}")


def _map_ended_reason(ended_reason: str | None) -> str:
    if not ended_reason:
        return "unknown"
    r = ended_reason.lower()
    if "assistant" in r or "agent" in r:
        return "agent"
    if "user" in r or "customer" in r:
        return "customer"
    if "error" in r:
        return "error"
    return "unknown"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test v6.0 per-intent schema")
    parser.add_argument("transcript", type=Path, nargs="?",
                        default=Path("tests/golden/transcripts/0a434683-1ac4-4f26-83ee-c6fce1040ef4.json"))
    parser.add_argument("--model", default="gemini-3-flash-preview")
    args = parser.parse_args()

    if not args.transcript.exists():
        print(f"Error: {args.transcript} not found", file=sys.stderr)
        sys.exit(1)

    run_test(args.transcript, args.model)
