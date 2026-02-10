# v3.7 Release Notes: Transcript Preprocessing + Structured Event Context

**Release Date:** 2026-01-19

## Summary

v3.7 introduces two key improvements:
1. **Preprocessing**: Deterministic turn counting before LLM analysis
2. **Structured event context**: `cause` enum + `context` sentence for clarifications; `severity` enum + `context` sentence for corrections

## Why These Changes?

### Problem: Non-deterministic turn numbers
In v3.6, the LLM was responsible for counting turns, which led to occasional inconsistencies between the turn numbers reported in clarification/correction details and actual transcript positions.

### Solution: Preprocess first
Now `preprocess_transcript.py` parses the raw transcript deterministically, assigning sequential turn numbers before the LLM sees it. The LLM receives a structured format with explicit turn markers.

### Problem: Aggregation relied on LLM prose
v3.6 captured what happened but not in a structured way that enabled reliable counting. To answer "How often do customers refuse to spell their name?", we had to run another LLM pass to cluster prose descriptions.

### Solution: Enum + context hybrid
- **`cause` enum**: Provides reliable counting baseline for aggregation
- **`context` sentence**: Provides rich detail for nuanced analysis
- Best of both: aggregate LLM uses enums for metrics, context for discovering patterns

## New Schema Fields

### Clarification Details (v3.7)

```json
{
  "type": "name_spelling",
  "turn": 5,
  "resolved": false,
  "cause": "customer_refused",
  "context": "Customer said 'I already told you' and demanded a human agent"
}
```

**Cause enum values:**
| Cause | Description | Example |
|-------|-------------|---------|
| `customer_refused` | Customer declined to provide info | "No." / "I already told you" |
| `customer_unclear` | Customer provided but unclear | Mumbled, partial, ambiguous |
| `agent_misheard` | Agent failed to understand | Heard "Butchering" not "Butcherine" |
| `tech_issue` | Audio/connection problem | Cut off, static, silence |
| `successful` | Clarification worked | Use when resolved=true |

### User Correction Details (v3.7)

```json
{
  "what_was_wrong": "wrong resort name",
  "turn": 7,
  "frustration_signal": true,
  "severity": "moderate",
  "context": "Customer corrected 'Grandview' to 'Grand Pacific' with audible frustration"
}
```

**Severity enum values:**
| Severity | Description |
|----------|-------------|
| `minor` | Simple correction, no frustration |
| `moderate` | Correction with mild frustration |
| `major` | Explicit anger, multiple corrections needed |

## New Files

### `preprocess_transcript.py`

Standalone tool for transcript preprocessing.

```bash
# Output JSON to stdout
python3 tools/preprocess_transcript.py transcripts/uuid.txt --stdout

# Output LLM-ready format
python3 tools/preprocess_transcript.py transcripts/uuid.txt --stdout --llm-format

# Save to directory
python3 tools/preprocess_transcript.py transcripts/uuid.txt -o preprocessed/
```

**Output format:**
```json
{
  "call_id": "uuid",
  "source_file": "uuid.txt",
  "metadata": {
    "total_turns": 4,
    "user_turns": 2,
    "agent_turns": 2,
    "total_words": 18,
    "avg_words_per_turn": 4.5
  },
  "turns": [
    {"turn": 1, "role": "assistant", "text": "Hi, how can I help?", "words": 5},
    {"turn": 2, "role": "user", "text": "I need help with my account.", "words": 6}
  ]
}
```

## Updated Metrics

### `compute_metrics.py`

New aggregations in `clarification_stats`:
- `by_cause`: Distribution by cause type

New aggregations in `correction_stats`:
- `by_severity`: Distribution by severity level

### `extract_nl_fields.py`

Clarification and correction events now include:
- `cause` / `severity` fields
- `context` sentences

## Report Changes

### Conversation Quality Section

New subsections:
- **By Cause (v3.7)**: Table showing cause distribution
- **By Severity (v3.7)**: Table showing severity distribution
- **Clarification Cause Analysis**: LLM insights on cause patterns
- **Correction Severity Analysis**: LLM insights on severity patterns

## Pipeline Integration

Preprocessing is integrated into `analyze_transcript.py`:
- Automatically calls `preprocess_transcript()` internally
- Passes structured format to LLM
- Stores preprocessing metadata in `_preprocessing` field

No changes needed to `run_analysis.py` or `batch_analyze.py`.

## What This Enables

1. **Guaranteed correct turn numbers** (deterministic preprocessing)
2. **Answer**: "How often do customers refuse to spell their name?" (cause=customer_refused)
3. **Answer**: "How often does agent mishear?" (cause=agent_misheard)
4. **Answer**: "What's the breakdown of correction severity?" (by_severity)
5. **Correlate**: cause types with outcomes
6. **Cleaner aggregate analysis**: Count enums, don't cluster prose
7. **Human-readable event logs**: Context sentences explain each friction event

## What v3.7 Does NOT Do

1. ~~friction_patterns field~~ → Derive in aggregate analysis
2. ~~Call-level pattern detection~~ → Let insights LLM correlate existing fields

## Verification

```bash
# Run v3.7 feature tests
python3 tools/test_v37_features.py

# Test preprocessing
python3 tools/preprocess_transcript.py transcripts/00bbb5fd-*.txt --stdout

# Test full pipeline
python3 tools/run_analysis.py --quick

# Check cause/severity in analysis
python3 tools/analyze_transcript.py sampled/*.txt --stdout | \
  jq '.clarification_requests.details[] | {cause, context}'

# Check cause distribution in metrics
cat reports/metrics_v3_*.json | jq '.deterministic_metrics.conversation_quality.clarification_stats.by_cause'
```

## Migration Notes

- v3.6 analyses remain compatible (missing cause/severity fields will be null)
- New analyses will have v3.7 schema version
- No changes required to existing pipelines
