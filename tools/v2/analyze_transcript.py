#!/usr/bin/env python3
"""
Single Transcript Analyzer for Vacatia AI Voice Agent Analytics (v2)

Simplified, elegant schema focused on actionable insights.
~14 fields capturing 80% of value with 80% less verbosity.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import google.generativeai as genai


# v2 Schema - Simplified and actionable
ANALYSIS_SCHEMA = """
{
  "call_id": "string (UUID from filename)",

  "outcome": "enum: 'resolved' | 'escalated' | 'abandoned' | 'unclear'",
  "resolution_type": "string (what was accomplished: 'payment processed', 'callback scheduled', 'info provided', 'none', etc.)",

  "agent_effectiveness": "integer 1-5 (did agent understand and respond appropriately?)",
  "conversation_quality": "integer 1-5 (flow, tone, clarity combined)",
  "customer_effort": "integer 1-5 (how hard did customer work? 1=effortless, 5=painful)",

  "failure_point": "enum: 'none' | 'nlu_miss' | 'wrong_action' | 'policy_gap' | 'customer_confusion' | 'tech_issue' | 'other' | null",
  "failure_description": "string or null (one sentence: what went wrong, if anything)",
  "was_recoverable": "boolean or null (could agent have saved this call?)",
  "critical_failure": "boolean (wrong info given, impossible promise made, compliance issue)",

  "escalation_requested": "boolean (did customer ask for a human?)",
  "repeat_caller_signals": "boolean (customer mentioned prior calls, ongoing frustration)",
  "training_opportunity": "string or null (specific skill gap if any: 'payment_flow', 'empathy', 'clarification', 'verification', etc.)",

  "additional_intents": "string or null (if customer had secondary requests beyond main intent, describe briefly)",

  "summary": "string (1-2 sentence summary focused on outcome and key driver)"
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
- **none**: Call succeeded
- **nlu_miss**: Agent misunderstood what customer said/wanted
- **wrong_action**: Agent understood but took wrong action
- **policy_gap**: Agent couldn't help due to business rules/limitations
- **customer_confusion**: Customer unclear or provided wrong info
- **tech_issue**: System errors, call quality problems
- **other**: Anything else

## Critical Failure Flag
Set `critical_failure: true` ONLY if:
- Agent provided incorrect information
- Agent promised something impossible
- Compliance/security issue occurred
- Customer could be harmed by following agent's guidance

## Output Requirements
- Return ONLY valid JSON
- No markdown code blocks
- Be concise and actionable
- Focus on what matters for improving the agent
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


def analyze_transcript(transcript_path: Path, model_name: str = "gemini-2.5-flash") -> dict:
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
    analysis["schema_version"] = "v2"

    return analysis


def save_analysis(analysis: dict, output_dir: Path) -> Path:
    """Save analysis JSON to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{analysis['call_id']}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Analyze call transcript (v2 - simplified schema)")
    parser.add_argument("transcript", type=Path, help="Path to transcript file")
    parser.add_argument("-o", "--output-dir", type=Path,
                        default=Path(__file__).parent.parent.parent / "analyses",
                        help="Output directory for analysis JSON")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        help="Gemini model to use")
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
