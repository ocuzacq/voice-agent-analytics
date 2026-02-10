# Voice Agent Analytics v4.0 Release Notes

## Overview

v4.0 is a **schema cleanup release** that adds primary intent tracking, sentiment analysis, and restructures the schema for better clarity and consistency.

**Release Date**: 2026-01-21

---

## What Changed

### New Fields (4 fields added)

| Field | Type | Purpose |
|-------|------|---------|
| `intent` | string | **WHAT** the customer wants (normalized phrase, 3-8 words) |
| `intent_context` | string/null | **WHY** they need it (underlying situation) |
| `sentiment_start` | enum | Customer mood at call start |
| `sentiment_end` | enum | Customer mood at call end |

### Unified Fields (2 fields merged)

| v3.9.x Fields | v4.0 Field | Notes |
|---------------|------------|-------|
| `outcome` + `call_disposition` | `disposition` | Single source of truth for call outcome |

### Renamed Fields (12 fields renamed)

| v3.9.x Name | v4.0 Name | Rationale |
|-------------|-----------|-----------|
| `agent_effectiveness` | `effectiveness` | Shorter, still clear |
| `conversation_quality` | `quality` | Shorter |
| `customer_effort` | `effort` | Shorter |
| `customer_verbatim` | `verbatim` | Obvious what it is |
| `agent_miss_detail` | `coaching` | More actionable name |
| `resolution_type` | `resolution` | Shorter |
| `resolution_steps` | `steps` | Shorter |
| `failure_point` | `failure_type` | More descriptive |
| `failure_description` | `failure_detail` | Consistent naming |
| `was_recoverable` | `failure_recoverable` | Grouped with failure |
| `critical_failure` | `failure_critical` | Grouped with failure |
| `repeat_caller_signals` | `repeat_caller` | Shorter |
| `ended_reason` | `ended_by` | Simplified values |

### Flattened Structure (5 fields promoted)

| v3.9.x Location | v4.0 Location |
|-----------------|---------------|
| `friction.turns` | `turns` |
| `friction.derailed_at` | `derailed_at` |
| `friction.clarifications` | `clarifications` |
| `friction.corrections` | `corrections` |
| `friction.loops` | `loops` |

### Removed Fields (1 field)

| Field | Why Removed | Mitigation |
|-------|-------------|------------|
| `training_opportunity` | Redundant with `coaching` | Free-text coaching captures same info with more detail |

---

## Field Mapping: v3.9.2 → v4.0

| v3.9.2 Field | v4.0 Field | Status |
|--------------|------------|--------|
| `call_id` | `call_id` | Preserved |
| `schema_version` | `schema_version` | Now "v4.0" |
| `outcome` | — | Removed (merged into disposition) |
| `resolution_type` | `resolution` | Renamed |
| `ended_reason` | `ended_by` | Simplified |
| `agent_effectiveness` | `effectiveness` | Renamed |
| `conversation_quality` | `quality` | Renamed |
| `customer_effort` | `effort` | Renamed |
| `failure_point` | `failure_type` | Renamed |
| `failure_description` | `failure_detail` | Renamed |
| `was_recoverable` | `failure_recoverable` | Renamed |
| `critical_failure` | `failure_critical` | Renamed |
| `escalation_requested` | `escalation_requested` | Preserved |
| `repeat_caller_signals` | `repeat_caller` | Renamed |
| `training_opportunity` | — | Merged into coaching |
| `additional_intents` | `secondary_intent` | Renamed |
| `summary` | `summary` | Preserved |
| `call_disposition` | `disposition` | Renamed (primary) |
| `policy_gap_detail` | `policy_gap` | Renamed |
| `customer_verbatim` | `verbatim` | Renamed |
| `agent_miss_detail` | `coaching` | Renamed |
| `resolution_steps` | `steps` | Renamed |
| `friction.turns` | `turns` | Promoted |
| `friction.derailed_at` | `derailed_at` | Promoted |
| `friction.clarifications` | `clarifications` | Promoted |
| `friction.corrections` | `corrections` | Promoted |
| `friction.loops` | `loops` | Promoted |
| — | `intent` | **New** |
| — | `intent_context` | **New** |
| — | `sentiment_start` | **New** |
| — | `sentiment_end` | **New** |

---

## Intent Field Design

### Philosophy: Free Text with Normalization Guidance

Instead of rigid enums, v4.0 uses **prompted free text** with clear guidance:

```
intent: "Check maintenance fee balance"
intent_context: "Has not received invoice by mail as usual"
```

### Normalization Guidelines (in prompt)

- Start with an action verb when possible
- Be specific but concise (3-8 words)
- Use consistent terminology

**Examples:**
| intent | intent_context |
|--------|----------------|
| "Log into Clubhouse portal" | "Forgot which email address is registered" |
| "Check maintenance fee balance" | "Has not received invoice by mail as usual" |
| "Make a payment" | "Received past-due notice" |
| "Get reservation dates" | "Booking confirmation email was lost" |
| "Speak to a representative" | "Agent could not resolve booking code issue" |

---

## Sentiment Field Design

### Values

```python
sentiment_start: "positive" | "neutral" | "frustrated" | "confused" | "angry"
sentiment_end: "satisfied" | "neutral" | "frustrated" | "dissatisfied" | "angry"
```

### Usage

Tracks emotional journey through the call:
- `neutral → satisfied`: Resolution improved mood
- `frustrated → angry`: Call made things worse
- `neutral → neutral`: No emotional change

---

## Disposition Values

Unified from `outcome` + `call_disposition`:

| Value | Description |
|-------|-------------|
| `pre_intent` | Hung up before stating need (≤2 turns) |
| `out_of_scope_handled` | Request outside scope, redirected gracefully |
| `out_of_scope_failed` | Request outside scope, customer left unhappy |
| `in_scope_success` | Resolved with explicit customer confirmation |
| `in_scope_partial` | Completed but no explicit satisfaction |
| `in_scope_failed` | Could not resolve in-scope request |
| `escalated` | Transferred to human agent |

---

## v4.0 Schema Example

```json
{
  "call_id": "2364cfbd-38dd-4a23-8fdb-d41eb1e69c72",
  "schema_version": "v4.0",

  "turns": 21,
  "ended_by": "agent",

  "intent": "Log into Clubhouse portal",
  "intent_context": "Registration link not arriving via email",
  "secondary_intent": "Pay maintenance fees",

  "disposition": "in_scope_success",
  "resolution": "callback scheduled",
  "steps": ["greeted customer", "identified intent", "attempted link send", "scheduled callback"],

  "effectiveness": 4,
  "quality": 4,
  "effort": 3,
  "sentiment_start": "neutral",
  "sentiment_end": "satisfied",

  "failure_type": null,
  "failure_detail": null,
  "failure_recoverable": null,
  "failure_critical": false,
  "policy_gap": null,

  "derailed_at": null,
  "clarifications": [{"turn": 3, "type": "intent", "cause": "ok", "note": "clarified login vs delayed bill"}],
  "corrections": [],
  "loops": [{"turns": [13, 15], "type": "info_retry", "subject": "name", "note": "asked for last name spelling twice"}],

  "summary": "Customer called to log into Clubhouse. Link delivery failed; callback scheduled.",
  "verbatim": "Stay on the line until I get the link, please.",
  "coaching": "Avoid re-asking for name after customer already provided it clearly.",

  "escalation_requested": false,
  "repeat_caller": false
}
```

---

## Backwards Compatibility

All pipeline tools support both v4.0 and v3.x analyses:

| Tool | Compatibility |
|------|---------------|
| `compute_metrics.py` | Reads v4.0 field names, falls back to v3.x |
| `extract_nl_fields.py` | Extracts from both schema versions |
| `generate_insights.py` | Produces intent/sentiment analysis for v4.0, skips for v3.x |
| `render_report.py` | Renders Intent/Sentiment sections when available |
| `ask.py` | Displays v4.0 fields (intent, sentiment) when present |

**Pattern used:**
```python
disposition = analysis.get("disposition") or analysis.get("call_disposition", "unknown")
```

---

## Report Enhancements

### New Sections

1. **Intent Analysis**
   - Top intent clusters with success rates
   - Context patterns (WHY customers need help)
   - Unmet needs (capability expansion opportunities)

2. **Sentiment Analysis**
   - Emotional journey patterns
   - Improvement drivers (what makes customers happier)
   - Degradation drivers (what makes things worse)
   - Sentiment health metrics

### Updated Metrics

- `intent_stats`: Intent distribution and success correlation
- `sentiment_stats`: Journey pattern distribution and health metrics

---

## Migration Notes

### For Existing v3.x Analyses

- No action required - all tools read v3.x format with fallbacks
- Can run mixed batches (some v3.x, some v4.0)
- Metrics aggregate correctly from both versions

### For New Analyses

- Set `GOOGLE_API_KEY` environment variable
- Run pipeline as normal - v4.0 schema produced automatically
- Intent/sentiment fields populated by LLM

---

## Files Modified

| File | Changes |
|------|---------|
| `analyze_transcript.py` | v4.0 schema + prompt with intent/sentiment guidelines |
| `compute_metrics.py` | `compute_intent_stats()`, `compute_sentiment_stats()`, backwards compat helpers |
| `extract_nl_fields.py` | Extracts `intent_data`, `sentiment_data` from analyses |
| `generate_insights.py` | `intent_analysis`, `sentiment_analysis` in output schema |
| `render_report.py` | Intent Analysis, Sentiment Analysis sections |
| `ask.py` | v4.0 field display with backwards compatibility |
| `test_v40_features.py` | Comprehensive test suite (14 tests) |

---

## Testing

Run the v4.0 test suite:

```bash
python3 tools/test_v40_features.py
```

Expected output: 14 tests pass, covering:
- Schema field presence
- Prompt guidelines
- Backwards compatibility
- Field extraction
- Report rendering
- Version references

---

## What's Next

Potential v4.1 enhancements:
- Intent clustering via embeddings (deterministic, not LLM)
- Sentiment correlation with friction events
- Intent success rate by resolution type
