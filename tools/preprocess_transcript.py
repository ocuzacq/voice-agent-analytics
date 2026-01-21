#!/usr/bin/env python3
"""
Transcript Preprocessor for Vacatia AI Voice Agent Analytics (v3.9.2)

Converts raw transcripts to structured JSON with deterministic turn counting.
This ensures correct turn numbers for clarification/correction details.

v3.9.2: Support for new JSON transcript format with timestamps and metadata:
- Auto-detect format by file extension (.txt vs .json)
- Coalesce consecutive same-role messages (ASR segments)
- Compute turn timing from timestamps (start, end, duration)
- Calculate agent response latency
- Pass through call metadata (duration, agent_id, ended_reason)

Input formats:
- .txt: Raw transcript (assistant:/user: format)
- .json: Structured JSON with messages array and metadata

Output: Structured JSON with turn metadata

Benefits:
- Guaranteed correct turn numbers (deterministic, not LLM-dependent)
- Cleaner input format for LLM analysis (structured JSON vs raw text)
- Foundation for better analysis with pre-computed metadata
- Rich timing data from JSON sources for response latency analysis
"""

import argparse
import json
import sys
from pathlib import Path


def seconds_to_hhmmss(seconds: float) -> str:
    """Convert seconds to HH:MM:SS format."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def preprocess_json_transcript(transcript_path: Path) -> dict:
    """
    Convert JSON transcript to structured format.

    Handles new JSON format with messages array and metadata.
    Coalesces consecutive same-role messages into single turns.
    Computes timing from timestamps.

    Args:
        transcript_path: Path to JSON transcript file

    Returns:
        Structured dictionary with call metadata and turns
    """
    with open(transcript_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    call_id = transcript_path.stem
    messages = data.get("messages", [])

    if not messages:
        return {
            "call_id": call_id,
            "source_file": transcript_path.name,
            "metadata": {
                "total_turns": 0,
                "user_turns": 0,
                "agent_turns": 0,
                "total_words": 0,
                "avg_words_per_turn": 0,
                "duration": data.get("duration"),
                "agent_id": data.get("agent_id"),
                "ended_reason": data.get("ended_reason")
            },
            "turns": []
        }

    # Get the first timestamp as the baseline for relative timing
    first_timestamp = messages[0].get("timestamp", 0)

    # Coalesce consecutive same-role messages into turns
    turns = []
    turn_num = 0
    current_role = None
    current_texts = []
    current_start_ts = None
    current_end_ts = None
    prev_turn_end_ts = None

    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "").strip()
        timestamp = msg.get("timestamp")

        if not content:
            continue

        # Normalize role names
        if role == "assistant":
            role = "assistant"
        elif role == "user":
            role = "user"
        else:
            continue  # Skip unknown roles

        if role != current_role:
            # Save previous turn if exists
            if current_role and current_texts:
                text = ' '.join(current_texts)
                turn_data = {
                    "turn": turn_num,
                    "role": current_role,
                    "text": text,
                    "words": len(text.split())
                }

                # Add timing if we have timestamps
                if current_start_ts is not None:
                    rel_start = current_start_ts - first_timestamp
                    turn_data["start"] = seconds_to_hhmmss(rel_start)

                    if current_end_ts is not None:
                        rel_end = current_end_ts - first_timestamp
                        turn_data["end"] = seconds_to_hhmmss(rel_end)
                        turn_data["duration"] = round(current_end_ts - current_start_ts, 2)

                    # Add response latency for agent turns (time from previous user turn end)
                    if current_role == "assistant" and prev_turn_end_ts is not None:
                        latency = current_start_ts - prev_turn_end_ts
                        if latency > 0:
                            turn_data["response_latency"] = round(latency, 2)

                turns.append(turn_data)
                prev_turn_end_ts = current_end_ts

            # Start new turn
            turn_num += 1
            current_role = role
            current_texts = [content]
            current_start_ts = timestamp
            current_end_ts = timestamp
        else:
            # Same role - coalesce
            current_texts.append(content)
            if timestamp is not None:
                current_end_ts = timestamp

    # Don't forget the last turn
    if current_role and current_texts:
        text = ' '.join(current_texts)
        turn_data = {
            "turn": turn_num,
            "role": current_role,
            "text": text,
            "words": len(text.split())
        }

        if current_start_ts is not None:
            rel_start = current_start_ts - first_timestamp
            turn_data["start"] = seconds_to_hhmmss(rel_start)

            if current_end_ts is not None:
                rel_end = current_end_ts - first_timestamp
                turn_data["end"] = seconds_to_hhmmss(rel_end)
                turn_data["duration"] = round(current_end_ts - current_start_ts, 2)

            if current_role == "assistant" and prev_turn_end_ts is not None:
                latency = current_start_ts - prev_turn_end_ts
                if latency > 0:
                    turn_data["response_latency"] = round(latency, 2)

        turns.append(turn_data)

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
            "avg_words_per_turn": round(total_words / len(turns), 1) if turns else 0,
            "duration": data.get("duration"),
            "agent_id": data.get("agent_id"),
            "ended_reason": data.get("ended_reason")
        },
        "turns": turns
    }


def preprocess_txt_transcript(transcript_path: Path) -> dict:
    """
    Convert raw .txt transcript to structured JSON.

    Args:
        transcript_path: Path to raw transcript file (.txt format)

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


def preprocess_transcript(transcript_path: Path) -> dict:
    """
    Convert transcript to structured JSON (auto-detects format by extension).

    Args:
        transcript_path: Path to transcript file (.txt or .json)

    Returns:
        Structured dictionary with call metadata and turns
    """
    if transcript_path.suffix == '.json':
        return preprocess_json_transcript(transcript_path)
    else:
        return preprocess_txt_transcript(transcript_path)


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
    metadata = preprocessed['metadata']
    lines = [
        f"Call ID: {preprocessed['call_id']}",
        f"Total Turns: {metadata['total_turns']}",
        f"User Turns: {metadata['user_turns']}",
        f"Agent Turns: {metadata['agent_turns']}",
    ]

    # Include JSON-source metadata if available
    if metadata.get('duration') is not None:
        lines.append(f"Call Duration: {metadata['duration']:.1f}s")
    if metadata.get('ended_reason'):
        lines.append(f"Ended Reason: {metadata['ended_reason']}")
    if metadata.get('agent_id'):
        lines.append(f"Agent ID: {metadata['agent_id']}")

    lines.append("")
    lines.append("--- TRANSCRIPT ---")

    for turn in preprocessed["turns"]:
        role_label = "AGENT" if turn["role"] == "assistant" else "USER"

        # Include timing if available
        timing_info = ""
        if "start" in turn:
            timing_info = f" [{turn['start']}"
            if "duration" in turn:
                timing_info += f", {turn['duration']:.1f}s"
            timing_info += "]"

        lines.append(f"[Turn {turn['turn']}]{timing_info} {role_label}: {turn['text']}")

    lines.append("--- END TRANSCRIPT ---")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess transcript to structured JSON (v3.9.2 - supports .txt and .json)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process .txt transcript (legacy format)
    python3 tools/preprocess_transcript.py transcripts/uuid.txt --stdout

    # Process .json transcript (new format with timestamps)
    python3 tools/preprocess_transcript.py transcripts/uuid.json --stdout

    # Save to file
    python3 tools/preprocess_transcript.py transcripts/uuid.json -o preprocessed/

    # Output LLM-ready format (includes timing data if from JSON source)
    python3 tools/preprocess_transcript.py transcripts/uuid.json --stdout --llm-format
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
