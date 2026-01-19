#!/usr/bin/env python3
"""
Single Transcript Analyzer for Vacatia AI Voice Agent Analytics

Uses LLM to analyze a transcript and produce structured JSON output.
"""

import argparse
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

import google.generativeai as genai


# Analysis schema for the LLM prompt
ANALYSIS_SCHEMA = """
{
  "call_id": "string (UUID from filename)",
  "file_path": "string (path to transcript file)",

  "funnel": {
    "connect_greet": "boolean (did the call connect and greeting occur?)",
    "intent_captured": "boolean (did the agent understand what the customer wanted?)",
    "intent_description": "string (brief description of customer's intent, or null if not captured)",
    "verification_attempted": "boolean (did agent attempt to verify customer identity?)",
    "verification_successful": "boolean or null (was verification successful? null if not attempted)",
    "solution_path_entered": "boolean (did the call proceed to actually handling the request?)",
    "closure_type": "string: 'resolved' | 'escalated' | 'abandoned' | 'timeout' | 'unknown'"
  },

  "coverage": "string: 'PRE-INTENT' | 'IN-SCOPE' | 'OUT-OF-SCOPE'",
  "outcome": "string: 'SUCCESS' | 'FAILURE' | 'UNKNOWN'",

  "actions_claimed": ["list of actions the agent claimed to perform or offered to perform"],
  "actions_with_completion_evidence": ["subset of actions that have evidence of completion"],

  "issues": [
    {
      "type": "string: 'verification_loop' | 'stuck_in_loop' | 'escalation_refused' | 'verification_failed' | 'call_timeout' | 'transcription_error' | 'other'",
      "description": "string (brief description of the issue)",
      "severity": "string: 'low' | 'medium' | 'high'"
    }
  ],

  "human_escalation_requested": "boolean (did customer ask for a human/live agent?)",
  "human_escalation_honored": "boolean or null (was the escalation request honored? null if not requested)",

  "call_duration_proxy": "integer (line count as proxy for duration)",
  "turn_count": "integer (number of speaker turns)",

  "performance_scores": {
    "efficiency": {
      "score": "integer 1-5 (1=very inefficient, 5=very efficient)",
      "rationale": "string (brief explanation)",
      "factors": {
        "call_flow_smoothness": "integer 1-5 (how smoothly did the call progress?)",
        "unnecessary_repetition": "integer 1-5 (5=no repetition, 1=excessive repetition)",
        "time_to_resolution": "integer 1-5 (relative to task complexity)"
      }
    },
    "resolution": {
      "score": "integer 1-5 (1=unresolved, 5=fully resolved)",
      "rationale": "string (brief explanation)",
      "customer_need_met": "boolean (did customer get what they called for?)",
      "partial_resolution": "boolean (was there any partial progress?)",
      "next_steps_clear": "boolean (does customer know what happens next?)"
    },
    "satisfaction_signals": {
      "overall_score": "integer 1-5 (inferred from transcript cues)",
      "rationale": "string (brief explanation)",
      "components": {
        "interaction_quality": "integer 1-5 (how pleasant was the AI interaction?)",
        "outcome_satisfaction": "integer 1-5 (satisfaction with the result)",
        "effort_required": "integer 1-5 (5=low effort, 1=high effort from customer)"
      },
      "sentiment_indicators": ["list of observed sentiment cues: frustration, confusion, gratitude, impatience, etc."]
    }
  },

  "agent_quality": {
    "nlu_accuracy": {
      "score": "integer 1-5 (how well did agent understand customer?)",
      "misunderstandings": ["list of specific misunderstandings if any"],
      "clarification_attempts": "integer (how many times agent asked for clarification)"
    },
    "response_quality": {
      "relevance": "integer 1-5 (were responses relevant to customer needs?)",
      "helpfulness": "integer 1-5 (were responses actionable/helpful?)",
      "verbosity": "string: 'too_brief' | 'appropriate' | 'too_verbose'"
    },
    "conversational_flow": {
      "naturalness": "integer 1-5 (did conversation feel natural?)",
      "turn_taking": "integer 1-5 (smooth turn transitions?)",
      "context_retention": "integer 1-5 (did agent remember earlier context?)"
    },
    "tone_and_style": {
      "professionalism": "integer 1-5",
      "empathy": "integer 1-5 (appropriate emotional response?)",
      "patience": "integer 1-5 (especially with confused/frustrated customers)"
    },
    "transcription_quality": {
      "apparent_asr_errors": "integer (count of likely speech recognition errors)",
      "impact_on_call": "string: 'none' | 'minor' | 'significant' | 'severe'"
    }
  },

  "customer_profile": {
    "apparent_age_group": "string: 'young' | 'middle' | 'senior' | 'unknown' (inferred from speech patterns)",
    "tech_comfort_level": "integer 1-5 (comfort with automated systems)",
    "emotional_state_start": "string (calm, frustrated, confused, urgent, etc.)",
    "emotional_state_end": "string (same categories)",
    "communication_clarity": "integer 1-5 (how clearly did customer communicate?)"
  },

  "summary": "string (2-3 sentence summary of what happened in the call)"
}
"""

SYSTEM_PROMPT = """You are an expert call center quality analyst. Your task is to analyze customer service call transcripts from Vacatia's AI voice agent and produce structured JSON analysis.

## Context
Vacatia is a timeshare management company. Their AI voice agent handles calls about:
- RCI membership and points
- Timeshare week deposits
- Reservations and bookings
- Payment processing
- Account verification
- Basic account inquiries

## Transcript Format
The transcript uses these markers:
- `assistant:` = The AI voice agent
- `user:` = The customer

## Analysis Guidelines

### Coverage Classification
- **PRE-INTENT**: Very short calls where the customer didn't engage (just greeting, hangup)
- **IN-SCOPE**: The customer's request is something the V1 agent should handle
- **OUT-OF-SCOPE**: The customer wants something beyond V1 capabilities (e.g., complex billing disputes, legal issues, selling timeshare)

### Outcome Classification
- **SUCCESS**: The customer's need was addressed (resolved, appropriate escalation, or helpful information provided)
- **FAILURE**: The call ended without the customer's need being met (stuck loops, failed verification, abandoned)
- **UNKNOWN**: Cannot determine from transcript

### Issue Types
- **verification_loop**: Agent repeatedly asks for the same verification info
- **stuck_in_loop**: Agent gives identical or near-identical responses repeatedly
- **escalation_refused**: Customer asked for human but agent didn't honor it
- **verification_failed**: Customer couldn't be verified despite genuine attempts
- **call_timeout**: Call appears to have hit max duration limit
- **transcription_error**: Clear ASR/transcription issues affecting understanding
- **other**: Any other significant issue

### Action Types (common)
Use these labels when applicable:
- rci_activation, rci_inquiry
- payment_processing, payment_inquiry
- escalation, callback_scheduling
- reservation_lookup, reservation_booking
- week_deposit, week_inquiry
- referral_submission
- account_lookup, account_update
- verification_attempt

You may create additional descriptive action labels for actions not covered above.

### Performance Scoring Guidelines (1-5 scale)

**Efficiency Scoring:**
- 5: Direct path to resolution, minimal turns, no unnecessary steps
- 4: Mostly efficient with minor detours
- 3: Average, some repetition or unnecessary steps
- 2: Notable inefficiencies, multiple repetitions
- 1: Very inefficient, excessive loops, wasted effort

**Resolution Scoring:**
- 5: Customer need fully met, clear closure
- 4: Need mostly met, minor gaps
- 3: Partial resolution, some progress made
- 2: Minimal resolution, significant gaps
- 1: No resolution, customer still needs help

**Satisfaction Signals (inferred from transcript):**
Look for these cues:
- Positive: "thank you", "great", "perfect", tone lightening
- Negative: sighs, "I already said...", repeated questions, frustration words
- Neutral: brief responses, no emotional indicators
Score based on observable signals, not assumptions.

**Agent Quality Scoring:**

*NLU Accuracy:*
- 5: Perfect understanding, no misinterpretations
- 3: Occasional minor misunderstandings
- 1: Frequent, significant misunderstandings

*Response Relevance/Helpfulness:*
- 5: Every response directly addresses customer need
- 3: Mostly relevant with some tangents
- 1: Responses miss the point consistently

*Conversational Flow:*
- 5: Natural, human-like conversation
- 3: Functional but robotic at times
- 1: Awkward, disjointed, repetitive

*Tone (Professionalism/Empathy/Patience):*
- 5: Appropriately warm, professional, adapts to customer state
- 3: Neutral, neither warm nor cold
- 1: Inappropriate (cold when customer frustrated, etc.)

### Customer Profile Guidelines
- **Age inference**: Speech patterns, references, comfort with technology
- **Tech comfort**: How they handle IVR navigation, understanding of process
- **Emotional state**: Look for explicit cues (sighs, word choice, urgency)
- Rate "unknown" if insufficient evidence

## Output Requirements
- Return ONLY valid JSON matching the schema
- No markdown code blocks, just raw JSON
- Be objective and precise in your analysis
- Base all conclusions on evidence in the transcript
- For PRE-INTENT calls (very short), use null or minimal values for performance scores
"""

USER_PROMPT_TEMPLATE = """Analyze the following call transcript and return a JSON object matching this schema:

{schema}

## Transcript to Analyze
File: {file_path}
Call ID: {call_id}

---
{transcript}
---

Return ONLY the JSON analysis object, no other text."""


def configure_genai():
    """Configure the Google Generative AI client."""
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError(
            "No API key found. Set GOOGLE_API_KEY or GEMINI_API_KEY environment variable."
        )
    genai.configure(api_key=api_key)


def parse_transcript(content: str) -> dict:
    """Parse transcript to extract basic metrics."""
    lines = content.strip().split('\n')
    line_count = len(lines)

    # Count turns (speaker changes)
    turn_count = 0
    current_speaker = None
    for line in lines:
        if line.startswith('assistant:'):
            if current_speaker != 'assistant':
                turn_count += 1
                current_speaker = 'assistant'
        elif line.startswith('user:'):
            if current_speaker != 'user':
                turn_count += 1
                current_speaker = 'user'

    return {
        "line_count": line_count,
        "turn_count": turn_count
    }


def extract_json_from_response(text: str) -> dict:
    """Extract JSON from LLM response, handling various formats."""
    # Try direct JSON parse first
    text = text.strip()

    # Remove markdown code blocks if present
    if text.startswith('```'):
        # Find the end of the opening fence
        first_newline = text.find('\n')
        if first_newline != -1:
            text = text[first_newline + 1:]
        # Remove closing fence
        if text.endswith('```'):
            text = text[:-3]
        text = text.strip()

    # Try to parse as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try to find JSON object in the text
    brace_start = text.find('{')
    if brace_start != -1:
        # Find matching closing brace
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

    raise ValueError(f"Could not extract valid JSON from response: {text[:500]}...")


def analyze_transcript(
    transcript_path: Path,
    model_name: str = "gemini-2.5-flash"
) -> dict:
    """Analyze a single transcript using the LLM."""
    # Read transcript
    content = transcript_path.read_text(encoding='utf-8')
    call_id = transcript_path.stem

    # Get basic metrics
    metrics = parse_transcript(content)

    # Build prompt
    user_prompt = USER_PROMPT_TEMPLATE.format(
        schema=ANALYSIS_SCHEMA,
        file_path=str(transcript_path),
        call_id=call_id,
        transcript=content
    )

    # Call LLM
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=SYSTEM_PROMPT
    )

    response = model.generate_content(
        user_prompt,
        generation_config=genai.GenerationConfig(
            temperature=0.2,
            max_output_tokens=8192,
        )
    )

    # Parse response
    analysis = extract_json_from_response(response.text)

    # Ensure required fields are present
    analysis["call_id"] = call_id
    analysis["file_path"] = str(transcript_path)
    analysis["call_duration_proxy"] = metrics["line_count"]
    analysis["turn_count"] = metrics["turn_count"]

    return analysis


def save_analysis(analysis: dict, output_dir: Path) -> Path:
    """Save analysis JSON to output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)

    call_id = analysis["call_id"]
    output_path = output_dir / f"{call_id}.json"

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(analysis, f, indent=2, ensure_ascii=False)

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Analyze a single call transcript using LLM"
    )
    parser.add_argument(
        "transcript",
        type=Path,
        help="Path to transcript file to analyze"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "analyses",
        help="Output directory for analysis JSON"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gemini-2.5-flash",
        help="Gemini model to use (default: gemini-2.5-flash)"
    )
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Print analysis to stdout instead of saving to file"
    )

    args = parser.parse_args()

    # Validate input
    if not args.transcript.exists():
        print(f"Error: Transcript file not found: {args.transcript}", file=sys.stderr)
        return 1

    # Configure API
    try:
        configure_genai()
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    # Analyze
    print(f"Analyzing: {args.transcript}", file=sys.stderr)
    try:
        analysis = analyze_transcript(args.transcript, args.model)
    except Exception as e:
        print(f"Error analyzing transcript: {e}", file=sys.stderr)
        return 1

    # Output
    if args.stdout:
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
    else:
        output_path = save_analysis(analysis, args.output_dir)
        print(f"Analysis saved to: {output_path}", file=sys.stderr)
        print(json.dumps(analysis, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    exit(main())
