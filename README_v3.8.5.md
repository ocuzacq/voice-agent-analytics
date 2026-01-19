# v3.8.5 Release Notes: Streamlined Friction Tracking

## Overview

v3.8.5 consolidates the verbose friction tracking fields introduced in v3.6-v3.8 into a single, compact `friction` object. This reduces output size by ~31% while preserving all analytical value.

## Problem Statement

v3.8 added valuable friction tracking (clarifications, corrections, loops with typed enums), but the implementation was bloated:
- **+101% output size** (1,443 → 2,906 bytes/call)
- Verbose nested structures with separate count fields
- Long enum values that added unnecessary bytes
- `_preprocessing` metadata in output (internal, not needed)

## What Changed

### 1. Single `friction` Object

**Before (v3.8):** 5 separate top-level fields
```json
{
  "conversation_turns": 25,
  "turns_to_failure": 5,
  "clarification_requests": { "count": 2, "details": [...] },
  "user_corrections": { "count": 1, "details": [...] },
  "agent_loops": { "count": 1, "details": [...] }
}
```

**After (v3.8.5):** Single compact object
```json
{
  "friction": {
    "turns": 25,
    "derailed_at": 5,
    "clarifications": [...],
    "corrections": [...],
    "loops": [...]
  }
}
```

### 2. Compact Object Format

| Field | v3.8 | v3.8.5 |
|-------|------|--------|
| Turn number | `"turn": 5` | `"t": 5` |
| Context | `"context": "..."` | `"ctx": "..."` |
| Severity | `"severity": "major"` | `"sev": "major"` |

### 3. Shorter Enum Values

| v3.8 Value | v3.8.5 Value |
|------------|--------------|
| `name_spelling` | `name` |
| `phone_confirmation` | `phone` |
| `intent_clarification` | `intent` |
| `repeat_request` | `repeat` |
| `verification_retry` | `verify` |
| `agent_misheard` | `misheard` |
| `customer_unclear` | `unclear` |
| `customer_refused` | `refused` |
| `tech_issue` | `tech` |
| `successful` | `ok` |

### 4. Loops Now Include Turn Numbers

**v3.8:** Loops had no turn field
```json
{"type": "info_retry", "context": "name asked 3x despite refusal"}
```

**v3.8.5:** Loops include `t` array
```json
{"t": [20, 22, 25], "type": "info_retry", "ctx": "name asked 3x"}
```

### 5. Telegraph-Style Context

**v3.8:** 10-20 word descriptions
```json
"context": "Agent asked the customer to spell their name again after the first attempt was unclear"
```

**v3.8.5:** 5-8 word telegraph style
```json
"ctx": "re-spelled after mishearing"
```

### 6. Removed Fields

| Field | Why Removed |
|-------|-------------|
| `_preprocessing` | Internal metadata, not needed in output |
| `resolved` boolean | Inferrable from outcome |
| `what_was_wrong` | Often duplicates context |
| `count` wrappers | Array length = count |

## Size Savings

| Metric | v3.8 | v3.8.5 | Reduction |
|--------|------|--------|-----------|
| Friction fields | ~1,316 bytes | ~550 bytes | **58%** |
| Total per call | ~2,906 bytes | ~2,000 bytes | **31%** |
| Output tokens | ~800 | ~550 | **31%** |

## Complete Example

### v3.8.5 friction Object
```json
{
  "friction": {
    "turns": 25,
    "derailed_at": 5,
    "clarifications": [
      {"t": 5, "type": "name", "cause": "misheard", "ctx": "re-spelled after mishearing"},
      {"t": 11, "type": "phone", "cause": "ok", "ctx": "confirmed number"},
      {"t": 20, "type": "name", "cause": "misheard", "ctx": "re-asked spelling"}
    ],
    "corrections": [
      {"t": 21, "sev": "moderate", "ctx": "corrected misspelled name"},
      {"t": 28, "sev": "minor", "ctx": "clarified area code"}
    ],
    "loops": [
      {"t": [20, 22, 25], "type": "info_retry", "ctx": "name asked 3x despite refusal"},
      {"t": [2, 41], "type": "intent_retry", "ctx": "re-asked intent from turn 2"}
    ]
  }
}
```

## Backwards Compatibility

v3.8.5 includes full backwards compatibility with v3.8 format:

```python
# Parsing helpers in compute_metrics.py
from compute_metrics import (
    parse_clarifications,  # Works with both formats
    parse_corrections,     # Works with both formats
    parse_loops,           # Works with both formats
    get_conversation_turns, # friction.turns or conversation_turns
    get_turns_to_failure,   # friction.derailed_at or turns_to_failure
)

# Enum mapping for aggregation
CLARIFICATION_TYPE_MAP = {
    "name": "name_spelling",      # v3.8.5 → canonical
    "name_spelling": "name_spelling",  # v3.8 pass-through
    ...
}
```

All existing v3.8 analysis files will continue to work without modification.

## Migration

No migration needed - the pipeline automatically handles both formats:

1. **New analyses**: Generated in v3.8.5 compact format
2. **Old analyses**: Parsed via backwards-compatible helpers
3. **Aggregation**: Uses canonical enum values for consistent metrics

## Testing

```bash
# Run v3.8.5 feature tests
python3 tools/test_v385_features.py

# Run full pipeline
python3 tools/run_analysis.py --quick

# Verify output size
ls -la analyses/*.json | awk '{sum+=$5; count++} END {print "Avg:", sum/count, "bytes"}'
```

## Files Modified

| File | Changes |
|------|---------|
| `analyze_transcript.py` | New `friction` schema, shorter enums, telegraph context |
| `compute_metrics.py` | Parsing helpers, enum mapping, backwards compat |
| `extract_nl_fields.py` | Extract from friction object |
| `generate_insights.py` | Version references updated |
| `render_report.py` | Version references updated |
| `test_v385_features.py` | **NEW**: Unit tests for v3.8.5 |

## Success Criteria

| Criteria | Status |
|----------|--------|
| Size reduction ≥28% | Target ~31% |
| Same aggregation capabilities | Preserved |
| Turn precision via `t` field | Added to loops |
| Human-readable context | 5-8 words, telegraph style |
| Backwards compatibility | Full support for v3.8 |
| All tests pass | 7/7 passing |
