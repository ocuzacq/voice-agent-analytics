# v3.6 Release Notes: Conversation Quality & Misunderstanding Tracking

**Release Date:** 2026-01-19

## Overview

v3.6 adds **conversation quality tracking** to capture in-conversation friction that was previously invisible. While earlier versions tracked outcomes well, they missed the back-and-forth that happens *during* calls.

## Problem Solved

The v3.5.5 schema captured **outcomes** well but missed **in-conversation friction**:

1. **Misunderstanding patterns not tracked**:
   - Agent asks customer to repeat/rephrase/respell (proactive clarification)
   - Customer corrects the agent's understanding (reactive correction)
   - Agent repeats same prompt multiple times (loop behavior)

2. **No conversation complexity metrics**:
   - Total conversation turns (proxy for duration without timestamps)
   - Turns-to-failure (how long before call derails)
   - Can't differentiate quick failures from drawn-out painful ones

3. **Report gaps**:
   - Can't quantify "how often does the agent ask to respell names?"
   - Can't identify which verification steps cause most friction
   - No visibility into conversation efficiency

## New Schema Fields (5 fields, 18 → 23)

### conversation_turns
```json
"conversation_turns": 15
```
Total user+assistant exchange pairs. A "turn" = one user message followed by agent response. Serves as duration proxy since timestamps aren't available.

### turns_to_failure
```json
"turns_to_failure": 7
```
For non-resolved calls only: turn number where call started derailing. Helps measure "time to failure" using turns as proxy. Null for resolved calls.

### clarification_requests
```json
"clarification_requests": {
  "count": 3,
  "details": [
    {"type": "name_spelling", "turn": 8, "resolved": true},
    {"type": "phone_confirmation", "turn": 12, "resolved": false},
    {"type": "verification_retry", "turn": 15, "resolved": false}
  ]
}
```

**Types:**
| Type | Description | Example |
|------|-------------|---------|
| `name_spelling` | Agent asks to spell name | "Can you spell your name?" |
| `phone_confirmation` | Agent confirms phone number | "So that's 315-276-0534?" |
| `intent_clarification` | Agent asks what customer needs | "What can I help with?" after intent given |
| `repeat_request` | Agent asks to repeat | "Can you say that again?" |
| `verification_retry` | Agent asks for different verification | "Try another phone number" |

### user_corrections
```json
"user_corrections": {
  "count": 2,
  "details": [
    {"what_was_wrong": "wrong resort name", "turn": 5, "frustration_signal": true},
    {"what_was_wrong": "misheard phone number", "turn": 10, "frustration_signal": false}
  ]
}
```

Tracks when customer corrects agent's understanding. `frustration_signal` indicates exasperation, repeating forcefully, or explicit annoyance.

### repeated_prompts
```json
"repeated_prompts": {
  "count": 4,
  "max_consecutive": 3
}
```

Detects loop behavior: agent says substantially the same thing multiple times. `max_consecutive` indicates worst stuck-in-loop streak.

## New Section A Metrics

### conversation_quality
```json
"conversation_quality": {
  "turn_stats": {
    "avg_turns": 12.5,
    "median_turns": 10,
    "avg_turns_resolved": 8.3,
    "avg_turns_failed": 14.2,
    "avg_turns_to_failure": 7.5
  },
  "clarification_stats": {
    "calls_with_clarifications": 234,
    "pct_calls_with_clarifications": 0.468,
    "avg_clarifications_per_call": 1.8,
    "by_type": {
      "name_spelling": {"count": 89, "rate": 0.178},
      "phone_confirmation": {"count": 156, "rate": 0.312}
    },
    "resolution_rate": 0.72
  },
  "correction_stats": {
    "calls_with_corrections": 145,
    "pct_calls_with_corrections": 0.29,
    "avg_corrections_per_call": 1.2,
    "with_frustration_signal": 67,
    "frustration_rate": 0.46
  },
  "loop_stats": {
    "calls_with_loops": 56,
    "pct_calls_with_loops": 0.112,
    "avg_repeats": 2.3,
    "max_consecutive_overall": 5
  }
}
```

## New Section B Insights

### conversation_quality_analysis
```json
"conversation_quality_analysis": {
  "narrative": "The agent frequently requires clarification during verification...",

  "friction_hotspots": [
    {
      "pattern": "Name spelling requests",
      "frequency": "18% of calls",
      "impact": "Adds 2-3 turns, 40% lead to frustration signal",
      "recommendation": "Implement phonetic alphabet prompting"
    }
  ],

  "efficiency_insights": [
    "Resolved calls average 8 turns vs 14 for failed calls",
    "Calls with 2+ clarifications have 3x higher abandonment rate",
    "User corrections in first 5 turns predict escalation (72% correlation)"
  ],

  "turn_analysis": {
    "long_call_patterns": "What's causing unusually long calls?",
    "short_call_patterns": "What characterizes quick resolutions?",
    "turns_to_failure_insight": "Failed calls typically derail around turn 7"
  }
}
```

## New Report Section: Conversation Quality

Placed after Key Metrics for high visibility:

```markdown
## Conversation Quality

**Average Length:** 12.5 turns | **Resolved:** 8.3 turns | **Failed:** 14.2 turns | **Turns to Failure:** 7.5

### Clarification Friction

| Type | Count | % of Calls | Resolution Rate |
|------|-------|------------|-----------------|
| Phone Confirmation | 156 | 31.2% | 85% |
| Name Spelling | 89 | 17.8% | 62% |
| Verification Retry | 78 | 15.6% | 55% |

### Friction Hotspots

| Pattern | Frequency | Impact | Recommendation |
|---------|-----------|--------|----------------|
| Name spelling requests | 18% | High frustration | Phonetic alphabet |
| Phone verification retry | 15.6% | 45% abandonment | Email/contract fallback |

### Customer Corrections

- **145 calls** (29%) had customer corrections
- **67** (46%) showed frustration signals during correction

### Loop Detection

- **56 calls** (11.2%) had repeated prompts
- **Worst case:** 5 consecutive repeats
```

## Files Modified

| File | Changes |
|------|---------|
| `analyze_transcript.py` | +5 schema fields, system prompt additions |
| `compute_metrics.py` | `compute_conversation_quality_metrics()` function |
| `extract_nl_fields.py` | Clarification/correction/loop event extraction |
| `generate_insights.py` | `conversation_quality_analysis` in LLM prompt |
| `render_report.py` | "Conversation Quality" markdown section |

## Files Created

| File | Description |
|------|-------------|
| `test_v36_features.py` | Unit tests for v3.6 aggregation logic |
| `README_v3.6.md` | This release notes file |

## Migration Notes

### Backward Compatibility

- v3.6 analyses are **backward compatible** with v3.5.5 pipeline
- Schema version check accepts `v3.x` patterns
- New fields default to empty/null if missing

### Running v3.6

```bash
# Full pipeline with v3.6 features
python3 tools/run_analysis.py -n 50

# Test single transcript with new fields
python3 tools/analyze_transcript.py sampled/your-file.txt --stdout | jq '.clarification_requests'

# Run v3.6 tests
python3 tools/test_v36_features.py
```

## What v3.6 Enables

1. **Answer**: "How often does the agent ask customers to repeat/respell?"
2. **Answer**: "How long do calls last before they fail?"
3. **Answer**: "Which clarification types cause most friction?"
4. **Identify**: Calls where customer had to correct agent multiple times
5. **Detect**: Infinite loop behavior (repeated prompts)
6. **Correlate**: Early friction → escalation/abandonment outcomes

## Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Granularity** | Full details | Track each event with type, turn, resolved - enables correlation analysis |
| **Loop detection** | LLM judges similarity | More accurate than text matching for semantic loops |
| **Report placement** | After Key Metrics | High visibility for conversation efficiency |
| **Turns as time proxy** | Yes | No timestamps available; 1 turn ≈ 15-30 seconds average |

## Example Use Cases

### Finding Friction Patterns

```python
# Identify calls with high clarification burden
high_friction = [
    a for a in analyses
    if a.get("clarification_requests", {}).get("count", 0) >= 3
]
```

### Correlating Clarifications with Outcomes

```python
# Clarification types in failed calls
for a in analyses:
    if a.get("outcome") != "resolved":
        clar = a.get("clarification_requests", {})
        for d in clar.get("details", []):
            print(f"{d['type']} at turn {d['turn']} → {a['outcome']}")
```

### Detecting Loop Behavior

```python
# Worst loop offenders
loops = [a for a in analyses if a.get("repeated_prompts", {}).get("max_consecutive", 0) >= 3]
for a in loops:
    print(f"{a['call_id']}: {a['repeated_prompts']['max_consecutive']} consecutive repeats → {a['outcome']}")
```
