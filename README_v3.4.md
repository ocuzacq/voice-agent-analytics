# Voice Agent Analytics v3.4 Release Notes

**Release Date:** 2025-01-18

## Overview

v3.4 focuses on improving report readability by moving descriptions inline into tables, adding sub-breakdowns for major failure types, and fixing the customer ask clustering to provide accurate counts.

## What Changed

### 1. Inline Descriptions (4th Column)

All major tables now have a **Context** column providing run-specific descriptions inline with the data.

**Before (v3.3):**
```markdown
| Failure Type | Count | % |
|--------------|-------|---|
| policy_gap   | 175   | 57.6% |

*policy_gap: The AI understood but couldn't fulfill...*
```

**After (v3.4):**
```markdown
| Failure Type | Count | % | Context |
|--------------|-------|---|---------|
| policy_gap   | 175   | 57.6% | Dead-end escalation logic; no callback capability |
```

### 2. Key Metrics Context (NEW)

The Key Metrics table now includes context explaining **WHY** each metric has its value, not just what threshold it crosses.

```markdown
| Metric | Value | Assessment | Context |
|--------|-------|------------|---------|
| Success Rate | 39.2% | ❌ | Driven by dead-end escalation; verified customers can't reach humans |
| Escalation Rate | 20.8% | ⚠️ | 45% explicitly requested human; most blocked by capacity limits |
```

### 3. Sub-breakdowns for Major Failure Types (NEW)

Failure types representing ≥5% of total failures now get automatic sub-breakdowns showing patterns within that category.

### 4. Fixed Customer Ask Clustering (BUG FIX)

**Problem:** "Top Unmet Customer Needs" showed artificially low counts (e.g., 9 calls) because:
- `compute_metrics.py` aggregated by exact string match ("Speak to a representative" ≠ "Representative" ≠ "talk to a live agent")
- LLM only saw 15 of 200+ policy_gap_details for clustering

**Fix:**
- `extract_nl_fields.py` now extracts ALL customer_asks as `all_customer_asks` list
- `generate_insights.py` passes the complete list to the LLM with explicit clustering instructions
- LLM can now properly cluster semantically similar asks

**Before (v3.3):**
```markdown
1. **Request Human Agent Transfer** - 9 calls
2. **Payment Method Friction** - 4 calls
```

**After (v3.4):**
```markdown
1. **Request Human Agent Transfer** - 45 calls
2. **Payment/Billing Assistance** - 28 calls
3. **Make Reservation** - 22 calls
```

```markdown
#### Other Breakdown

| Pattern | Count | Description |
|---------|-------|-------------|
| caller_hangup | 28 | Customer disconnected mid-call before resolution |
| unclear_outcome | 14 | Call ended without clear resolution status |
```

## Schema Changes

### New Fields in `nl_summary_v3`

```json
{
  "all_customer_asks": ["ask1", "ask2", "..."]  // Raw list for LLM semantic clustering
}
```

### New Fields in `llm_insights`

```json
{
  "key_metrics_descriptions": {
    "success_rate": "WHY - main driver/cause (max 15 words)",
    "escalation_rate": "WHY - main driver/cause",
    "failure_rate": "WHY - main driver/cause",
    "customer_effort": "WHY - main driver/cause"
  },

  "failure_type_descriptions": {
    "<failure_type>": "concise run-specific description (max 15 words)"
  },

  "policy_gap_descriptions": {
    "<category>": "concise run-specific description (max 15 words)"
  },

  "major_failure_breakdowns": {
    "<failure_type>": {
      "patterns": [
        {"pattern": "name", "count": N, "description": "context"}
      ]
    }
  }
}
```

### Removed Fields

The following verbose fields from v3.3 are replaced by simpler key-value formats:

- `failure_type_explanations` → `failure_type_descriptions`
- `policy_gap_explanations` → `policy_gap_descriptions`

## Migration Notes

- **No breaking changes** - v3.4 is backwards compatible
- If `*_descriptions` fields are missing, tables render with empty Context columns
- Existing v3.3 reports will render correctly (graceful degradation)

## Files Modified

| File | Changes |
|------|---------|
| `tools/generate_insights.py` | Updated schema, added instructions for descriptions and breakdowns, passes all customer_asks for clustering |
| `tools/render_report.py` | 4-column tables, sub-breakdown rendering |
| `tools/extract_nl_fields.py` | Added `all_customer_asks` extraction for LLM clustering |
| `tools/test_v34_features.py` | **NEW** - Unit tests for v3.4 features |

## Running Tests

```bash
# Run v3.4 feature tests
python3 tools/test_v34_features.py

# Run full test suite
python3 tools/test_framework.py
```

## Example Output

### Key Metrics at a Glance
| Metric | Value | Assessment | Context |
|--------|-------|------------|---------|
| Success Rate | 39.2% | ❌ | Driven by dead-end escalation; verified customers can't reach humans |
| Escalation Rate | 20.8% | ⚠️ | 45% explicitly requested human; most blocked by capacity limits |
| Failure Rate | 60.8% | ❌ | 58% are policy gaps; primarily missing callback/queue capability |
| Customer Effort | 2.76/5 | ⚠️ | Verification succeeds but leads nowhere (verify-then-dump) |

### Failure Point Breakdown
| Failure Type | Count | % of Failures | Context |
|--------------|-------|---------------|---------|
| policy_gap | 175 | 57.6% | Dead-end escalation logic; no callback capability |
| other | 52 | 17.1% | Caller hangups and unclear outcomes |
| tech_issue | 47 | 15.5% | Latency, audio glitches, infinite loops |
| nlu_miss | 17 | 5.6% | Entity extraction failures on names/dates |
| wrong_action | 8 | 2.6% | Incorrect department transfers |
| customer_confusion | 5 | 1.6% | Caller misunderstanding of options |

#### Other Breakdown
| Pattern | Count | Description |
|---------|-------|-------------|
| caller_hangup | 28 | Customer disconnected mid-call before resolution |
| unclear_outcome | 14 | Call ended without clear resolution status |
| system_timeout | 10 | Session expired due to inactivity |

#### Tech Issue Breakdown
| Pattern | Count | Description |
|---------|-------|-------------|
| latency | 25 | Response delays causing customer frustration |
| audio_glitch | 12 | Audio quality issues disrupted conversation |
| infinite_loop | 10 | Agent stuck repeating same response |
