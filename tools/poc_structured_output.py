#!/usr/bin/env python3
"""
PoC: Gemini Structured Output with Pydantic Models

Validates that the Gemini API can enforce a typed schema via response_schema,
eliminating the need for JSON parsing/repair. Tests against one golden transcript.

Usage:
    source .env
    python3 tools/poc_structured_output.py tests/golden/transcripts/0a434683-1ac4-4f26-83ee-c6fce1040ef4.json
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Literal, Optional

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# Import preprocessing from the existing pipeline
sys.path.insert(0, str(Path(__file__).parent))
from preprocess_transcript import preprocess_transcript, format_for_llm


# ---------------------------------------------------------------------------
# Pydantic schema — minimal but exercises: enums, optionals, nested lists,
# int constraints, and descriptions
# ---------------------------------------------------------------------------

class Action(BaseModel):
    action: Literal[
        'account_lookup', 'verification', 'send_payment_link', 'send_portal_link',
        'send_autopay_link', 'send_rental_link', 'send_clubhouse_link',
        'send_rci_link', 'transfer', 'other'
    ]
    outcome: Literal['success', 'failed', 'retry', 'unknown']
    detail: str = Field(description="5-10 words, telegraph style")


class PocAnalysis(BaseModel):
    """Minimal v5.0 analysis schema for structured-output PoC."""

    intent: str = Field(description="Customer's primary request (3-8 words, action verb)")
    call_scope: Literal['in_scope', 'out_of_scope', 'mixed', 'no_request']
    call_outcome: Literal['completed', 'escalated', 'abandoned']
    resolution: str = Field(description="What was accomplished")
    summary: str = Field(description="1-2 sentence summary")

    effectiveness: int = Field(ge=1, le=5, description="Agent understanding/response quality")
    quality: int = Field(ge=1, le=5, description="Flow, tone, clarity")
    effort: int = Field(ge=1, le=5, description="Customer effort (1=effortless, 5=painful)")

    sentiment_start: Literal['positive', 'neutral', 'frustrated', 'angry']
    sentiment_end: Literal['satisfied', 'neutral', 'frustrated', 'angry']

    escalation_trigger: Optional[Literal[
        'customer_requested', 'scope_limit', 'task_failure', 'policy_routing'
    ]] = None
    abandon_stage: Optional[Literal[
        'pre_greeting', 'pre_intent', 'mid_task', 'post_delivery'
    ]] = None
    resolution_confirmed: Optional[bool] = None

    actions: list[Action] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# System prompt — keep semantic guidance; schema enforcement is API-side
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are an expert call center quality analyst. Analyze this Vacatia AI voice agent call transcript.

Context: Vacatia is a timeshare management company. Their AI handles: RCI membership, timeshare deposits, reservations, payments, account verification.

## Transcript Format
- [Turn N] AGENT: = AI voice agent
- [Turn N] USER: = Customer

## Key Rules
- call_scope: independent of outcome. Could the AI handle this request type?
- call_outcome: what actually happened (completed/escalated/abandoned)
- escalation_trigger: required when call_outcome=escalated, null otherwise
- abandon_stage: required when call_outcome=abandoned, null otherwise
- resolution_confirmed: required when call_outcome=completed, null otherwise
- effectiveness/quality: 5=best, effort: 1=effortless 5=painful
- sentiment_end can be 'satisfied' (explicitly pleased)
"""


def run_poc(transcript_path: Path, model_name: str = "gemini-3-flash-preview") -> None:
    """Run the structured output PoC against one transcript."""

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GOOGLE_API_KEY or GEMINI_API_KEY", file=sys.stderr)
        sys.exit(1)

    client = genai.Client(api_key=api_key)

    # Preprocess
    preprocessed = preprocess_transcript(transcript_path)
    formatted = format_for_llm(preprocessed)
    user_prompt = f"Analyze this call transcript:\n\n{formatted}"

    # --- Structured output config ---
    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=16384,
        system_instruction=SYSTEM_PROMPT,
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        response_mime_type="application/json",
        response_schema=PocAnalysis,  # Pydantic class, NOT .model_json_schema()
    )

    print(f"Model: {model_name}")
    print(f"Transcript: {transcript_path.name}")
    print(f"Schema: PocAnalysis ({len(PocAnalysis.model_fields)} fields)")
    print("---")

    t0 = time.time()
    response = client.models.generate_content(
        model=model_name,
        contents=user_prompt,
        config=config,
    )
    elapsed = time.time() - t0

    # --- Results ---
    print(f"\n=== API Response ({elapsed:.1f}s) ===\n")

    # 1. Check finish reason
    if response.candidates:
        finish = response.candidates[0].finish_reason
        print(f"Finish reason: {finish}")

    # 2. Check if response.parsed exists (SDK auto-validation)
    if hasattr(response, 'parsed') and response.parsed is not None:
        parsed: PocAnalysis = response.parsed
        print(f"response.parsed type: {type(parsed).__name__}")
        print(f"\n--- Parsed Model ---")
        print(json.dumps(parsed.model_dump(), indent=2))

        # 3. Validate enum constraints
        print(f"\n--- Validation Checks ---")
        print(f"  call_scope enum valid: {parsed.call_scope in ('in_scope', 'out_of_scope', 'mixed', 'no_request')}")
        print(f"  call_outcome enum valid: {parsed.call_outcome in ('completed', 'escalated', 'abandoned')}")
        print(f"  effectiveness in [1,5]: {1 <= parsed.effectiveness <= 5} (value={parsed.effectiveness})")
        print(f"  quality in [1,5]: {1 <= parsed.quality <= 5} (value={parsed.quality})")
        print(f"  effort in [1,5]: {1 <= parsed.effort <= 5} (value={parsed.effort})")
        print(f"  actions count: {len(parsed.actions)}")
        for i, a in enumerate(parsed.actions):
            print(f"    [{i}] {a.action} → {a.outcome}: {a.detail}")

        # 4. Check Optional field behavior
        print(f"\n--- Optional Field Behavior ---")
        print(f"  escalation_trigger: {parsed.escalation_trigger!r} (expected None for completed)")
        print(f"  abandon_stage: {parsed.abandon_stage!r} (expected None for completed)")
        print(f"  resolution_confirmed: {parsed.resolution_confirmed!r} (expected True/False for completed)")

    else:
        # Fallback: parse manually from response.text
        print("response.parsed is None — falling back to response.text")
        print(f"\n--- Raw Response ---")
        raw = response.text
        print(raw[:2000])

        # Try manual Pydantic validation
        try:
            data = json.loads(raw)
            manual = PocAnalysis.model_validate(data)
            print(f"\n--- Manual Pydantic Parse Succeeded ---")
            print(json.dumps(manual.model_dump(), indent=2))
        except Exception as e:
            print(f"\nManual parse failed: {e}")

    # 5. Token usage
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        u = response.usage_metadata
        print(f"\n--- Token Usage ---")
        print(f"  prompt: {getattr(u, 'prompt_token_count', '?')}")
        print(f"  response: {getattr(u, 'candidates_token_count', '?')}")
        print(f"  thinking: {getattr(u, 'thoughts_token_count', '?')}")
        print(f"  total: {getattr(u, 'total_token_count', '?')}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PoC: Gemini structured output with Pydantic")
    parser.add_argument("transcript", type=Path, help="Path to golden transcript JSON")
    parser.add_argument("--model", default="gemini-3-flash-preview",
                        help="Model name (default: gemini-3-flash-preview)")
    args = parser.parse_args()

    if not args.transcript.exists():
        print(f"Error: {args.transcript} not found", file=sys.stderr)
        sys.exit(1)

    run_poc(args.transcript, args.model)
