#!/usr/bin/env python3
"""
Transcript Preprocessor for Vacatia AI Voice Agent Analytics (v3.7)

Converts raw transcripts to structured JSON with deterministic turn counting.
This ensures correct turn numbers for clarification/correction details.

Input: Raw transcript file (assistant:/user: format)
Output: Structured JSON with turn metadata

Benefits:
- Guaranteed correct turn numbers (deterministic, not LLM-dependent)
- Cleaner input format for LLM analysis (structured JSON vs raw text)
- Foundation for better analysis with pre-computed metadata
"""

import argparse
import json
import sys
from pathlib import Path


def preprocess_transcript(transcript_path: Path) -> dict:
    """
    Convert raw transcript to structured JSON.

    Args:
        transcript_path: Path to raw transcript file

    Returns:
        Structured dictionary with call metadata and turns
    """
    content = transcript_path.read_text(encoding='utf-8')
    call_id = transcript_path.stem

    turns = []
    turn_num = 0
    current_role = None
    current_text = []

    for line in content.strip().split('\n'):
        line = line.strip()
        if not line:
            continue

        if line.startswith('assistant:'):
            # Save previous turn if exists
            if current_role and current_text:
                text = ' '.join(current_text)
                turns.append({
                    "turn": turn_num,
                    "role": current_role,
                    "text": text,
                    "words": len(text.split())
                })
            turn_num += 1
            current_role = "assistant"
            current_text = [line[10:].strip()]  # After "assistant:"

        elif line.startswith('user:'):
            # Save previous turn if exists
            if current_role and current_text:
                text = ' '.join(current_text)
                turns.append({
                    "turn": turn_num,
                    "role": current_role,
                    "text": text,
                    "words": len(text.split())
                })
            turn_num += 1
            current_role = "user"
            current_text = [line[5:].strip()]  # After "user:"

        else:
            # Continuation of previous message
            if current_text:
                current_text.append(line)

    # Don't forget last turn
    if current_role and current_text:
        text = ' '.join(current_text)
        turns.append({
            "turn": turn_num,
            "role": current_role,
            "text": text,
            "words": len(text.split())
        })

    # Compute metadata
    user_turns = sum(1 for t in turns if t["role"] == "user")
    agent_turns = sum(1 for t in turns if t["role"] == "assistant")
    total_words = sum(t["words"] for t in turns)

    return {
        "call_id": call_id,
        "source_file": transcript_path.name,
        "metadata": {
            "total_turns": len(turns),
            "user_turns": user_turns,
            "agent_turns": agent_turns,
            "total_words": total_words,
            "avg_words_per_turn": round(total_words / len(turns), 1) if turns else 0
        },
        "turns": turns
    }


def format_for_llm(preprocessed: dict) -> str:
    """
    Format preprocessed transcript for LLM analysis.

    Produces a clean, structured format that's easier for the LLM
    to parse and reference turn numbers accurately.

    Args:
        preprocessed: Output from preprocess_transcript()

    Returns:
        Formatted string for LLM prompt
    """
    lines = [
        f"Call ID: {preprocessed['call_id']}",
        f"Total Turns: {preprocessed['metadata']['total_turns']}",
        f"User Turns: {preprocessed['metadata']['user_turns']}",
        f"Agent Turns: {preprocessed['metadata']['agent_turns']}",
        "",
        "--- TRANSCRIPT ---"
    ]

    for turn in preprocessed["turns"]:
        role_label = "AGENT" if turn["role"] == "assistant" else "USER"
        lines.append(f"[Turn {turn['turn']}] {role_label}: {turn['text']}")

    lines.append("--- END TRANSCRIPT ---")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess raw transcript to structured JSON (v3.7)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Output JSON to stdout
    python3 tools/preprocess_transcript.py transcripts/uuid.txt --stdout

    # Save to file
    python3 tools/preprocess_transcript.py transcripts/uuid.txt -o preprocessed/

    # Output LLM-ready format
    python3 tools/preprocess_transcript.py transcripts/uuid.txt --stdout --llm-format
        """
    )

    parser.add_argument("transcript", type=Path, help="Path to transcript file")
    parser.add_argument("-o", "--output-dir", type=Path,
                        help="Output directory for preprocessed JSON")
    parser.add_argument("--stdout", action="store_true",
                        help="Print to stdout instead of saving")
    parser.add_argument("--llm-format", action="store_true",
                        help="Output in LLM-ready format instead of JSON")

    args = parser.parse_args()

    if not args.transcript.exists():
        print(f"Error: File not found: {args.transcript}", file=sys.stderr)
        return 1

    preprocessed = preprocess_transcript(args.transcript)

    if args.llm_format:
        output = format_for_llm(preprocessed)
        if args.stdout:
            print(output)
        else:
            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                output_path = args.output_dir / f"{preprocessed['call_id']}.txt"
                output_path.write_text(output, encoding='utf-8')
                print(f"Saved: {output_path}", file=sys.stderr)
            else:
                print(output)
    else:
        if args.stdout:
            print(json.dumps(preprocessed, indent=2))
        else:
            if args.output_dir:
                args.output_dir.mkdir(parents=True, exist_ok=True)
                output_path = args.output_dir / f"{preprocessed['call_id']}.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(preprocessed, f, indent=2)
                print(f"Saved: {output_path}", file=sys.stderr)
            else:
                print(json.dumps(preprocessed, indent=2))

    return 0


if __name__ == "__main__":
    exit(main())
