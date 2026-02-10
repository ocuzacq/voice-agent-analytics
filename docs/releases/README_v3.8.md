# v3.8 Release Notes: Agent Loops with Typed Detection

## Summary

v3.8 replaces the ambiguous `repeated_prompts` field with a clearer `agent_loops` schema that:
- Uses unambiguous terminology (no "prompt" which is confusing in LLM context)
- Tracks only friction loops (benign repetition excluded)
- Provides type enum for aggregation + context for understanding
- Captures the important "intent re-ask after identity check" pattern

## What Changed

### Schema Changes

**Before (v3.7):**
```json
"repeated_prompts": {
  "count": 3,
  "max_consecutive": 2
}
```

**After (v3.8):**
```json
"agent_loops": {
  "count": 2,
  "details": [
    {"type": "intent_retry", "context": "Asked 'how can I help' at turn 18 after customer stated intent at turn 2"},
    {"type": "info_retry", "context": "Asked for name spelling twice despite customer providing it"}
  ]
}
```

### Loop Type Enum

| Type | Meaning | Example |
|------|---------|---------|
| `info_retry` | Re-asked for info already provided | "Spell your name" asked twice |
| `intent_retry` | Re-asked for intent already stated | "How can I help?" after customer stated need |
| `deflection` | Generic questions while unable to help | "Anything else?" while stuck |
| `comprehension` | Couldn't hear, asked to repeat | "Sorry, one more time?" |
| `action_retry` | System/process retries | "Let me try that again" |

### What's Excluded (Benign, Don't Track)

- Greeting after returning from hold
- Re-engagement after silence/topic change
- Required compliance disclosures
- Confirmation before taking action

## Why This Change

1. **"prompt" is confusing** - In LLM/GenAI context, "prompt" means model input, not agent utterance
2. **Implies exact text matching** - We want semantic repetition detection
3. **No distinction** between benign ("How can I help?" after hold) vs friction (same question twice)
4. **Misses deflection pattern** - Agent can't help, keeps asking generic questions

## New Metrics

### Call-Level
- Loop count per call
- Loop types present

### Aggregate
- **By type distribution**: "35% intent_retry, 25% deflection, 20% info_retry..."
- **Intent retry rate**: % of calls with intent re-ask after identity check
- **Deflection rate**: % of calls with deflection loops (signals capability gaps)
- **Loop density**: loops / total turns (normalized for call length)

## Files Modified

| File | Changes |
|------|---------|
| `analyze_transcript.py` | Replaced `repeated_prompts` with `agent_loops` schema + updated system prompt |
| `compute_metrics.py` | Aggregate loops by type, calculate loop density |
| `extract_nl_fields.py` | Extract loop events with type and context |
| `generate_insights.py` | Added loop_type_analysis section |
| `render_report.py` | Updated Conversation Quality section with loop breakdown |
| `test_v38_features.py` | New: Unit tests for agent_loops |

## Backwards Compatibility

The code maintains backwards compatibility with v3.7 `repeated_prompts` format:
- `compute_metrics.py` checks for both `agent_loops` and `repeated_prompts`
- `extract_nl_fields.py` handles legacy format gracefully
- Existing v3.7 analyses will continue to work

## Migration Notes

No migration required. New analyses will use `agent_loops`, old analyses with `repeated_prompts` will continue to work.

## Verification

```bash
# Run v3.8 tests
python3 tools/test_v38_features.py

# Test analysis with new agent_loops field
python3 tools/analyze_transcript.py transcripts/00bbb5fd-*.txt --stdout | jq '.agent_loops'

# Run quick pipeline
python3 tools/run_analysis.py --quick

# Check loop distribution in metrics
cat reports/metrics_v3_*.json | jq '.deterministic_metrics.conversation_quality.loop_stats'
```

## Report Output Example

### Agent Loops Section

```markdown
### Agent Loops

- **12 calls** (24.0%) had friction loops
- **18 total loops** (1.5 avg per affected call)
- **Loop density:** 0.045 loops/turn

#### By Type (v3.8)

| Type | Count | % of Loops |
|------|-------|------------|
| Intent Retry | 7 | 38.9% |
| Info Retry | 5 | 27.8% |
| Deflection | 4 | 22.2% |
| Comprehension | 2 | 11.1% |
```

### LLM Analysis Section

```markdown
### Loop Type Analysis (v3.8)

Agent loops reveal systematic issues: intent_retry dominates (39%), indicating
the agent frequently re-asks what customers need after identity verification.

| Type | Frequency | Impact | Recommendation |
|------|-----------|--------|----------------|
| Intent Retry | 7 (39%) | Frustrates customers who already stated need | Cache intent pre-verification |
| Deflection | 4 (22%) | Signals capability gaps | Expand agent capabilities |

**Key Rates:**
- Intent Retry Rate: 14% of calls
- Deflection Rate: 8% of calls
```
