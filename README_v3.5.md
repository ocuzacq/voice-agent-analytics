# Voice Agent Analytics v3.5 Release Notes

**Release Date:** 2025-01-19

## Overview

v3.5 surfaces previously buried training insights and introduces emergent pattern detection. Training opportunities like "clarification" (489 calls) are now analyzed with narrative context explaining WHY gaps exist and HOW they correlate with failure types. The LLM can now discover patterns not fitting existing categories.

## Problem Solved

Important patterns were buried in raw tables and never surfaced in narrative sections because:

1. **`training_opportunity` was orphaned**: Captured per-call but NOT passed to LLM
2. **NL extraction skipped training data**: `extract_nl_fields.py` didn't include `training_opportunity`
3. **LLM couldn't generate narrative**: Never saw training context with associated failures
4. **No emergent pattern detection**: No mechanism to surface unexpected correlations
5. **`additional_intents` field unused**: Captured but never analyzed

## What Changed

### 1. Training & Development Section (Restructured)

**Before (v3.4):**
```markdown
## Training Priorities

| Skill Gap | Count |
|-----------|-------|
| verification | 666 |
| clarification | 489 |
| escalation_handling | 269 |
```

**After (v3.5):**
```markdown
## Training & Development

The agent shows systematic weakness in verification flows and escalation handling.
These are interconnected: failed verifications often trigger escalation requests
which are then mishandled. Root cause: rigid phone-only lookup with no fallback.

### Priority Skills

| Skill | Count | Root Cause | Recommended Action |
|-------|-------|------------|-------------------|
| verification | 666 | Phone lookup fails for valid customers | Add contract ID/address fallback |
| clarification | 489 | Repeats same question vs. rephrasing | Train on intent recognition patterns |
| escalation_handling | 269 | Ignores explicit "representative" requests | Prioritize human transfer intent |

### Cross-Dimensional Patterns

- **verification + auth_restriction** (145 calls): Customers who fail phone lookup are completely blocked
- **clarification + nlu_miss** (89 calls): Agent doesn't rephrase after NLU failure
```

### 2. Emergent Patterns Section (NEW)

LLM can now identify patterns that don't fit standard categories:

```markdown
## Emergent Patterns

*Patterns discovered that don't fit standard categories:*

### Verify-Then-Dump
**Frequency:** ~30% of abandoned calls

**Significance:** Major driver of customer frustration and repeat calls

_Agent completes full verification, then terminates call due to closed department._

Examples: 9b6b3888, a07abc2f, 33195b90
```

### 3. Secondary Customer Needs Section (NEW)

Additional customer intents are now clustered and analyzed:

```markdown
## Secondary Customer Needs

Many customers have secondary needs beyond their primary intent that go unaddressed.

| Need | Count | Implication |
|------|-------|-------------|
| Exit/Sell Timeshare | 50 | High-value retention opportunity being missed |
| Check Reservation Status | 35 | Could be handled proactively during primary flow |
```

## Data Flow Change

```
BEFORE (v3.4):
  training_opportunity → compute_metrics → raw count table only

AFTER (v3.5):
  training_opportunity → extract_nl_fields → generate_insights → narrative + table
                              ↓
                    (grouped by type with context)
```

## Schema Changes

### New Fields in `nl_summary_v3`

```json
{
  "training_details": [
    {
      "call_id": "abc123",
      "opportunity": "verification",
      "outcome": "escalated",
      "failure_point": "auth_restriction",
      "failure_description": "Customer failed phone verification",
      "agent_miss_detail": "Did not offer contract ID fallback"
    }
  ],
  "all_additional_intents": [
    {
      "call_id": "abc123",
      "outcome": "resolved",
      "intent": "Also wanted to check reservation status"
    }
  ]
}
```

### New Fields in `llm_insights`

```json
{
  "training_analysis": {
    "narrative": "2-3 sentences synthesizing training gaps and root causes",
    "top_priorities": [
      {
        "skill": "verification",
        "count": 666,
        "why": "WHY this skill gap exists (root cause)",
        "action": "Specific training intervention"
      }
    ],
    "cross_correlations": [
      {
        "pattern": "verification + auth_restriction",
        "count": 145,
        "insight": "What this correlation reveals"
      }
    ]
  },

  "emergent_patterns": [
    {
      "name": "Pattern name",
      "frequency": "count or percentage",
      "description": "What you observed",
      "significance": "Why this matters",
      "example_call_ids": ["id1", "id2"]
    }
  ],

  "secondary_intents_analysis": {
    "narrative": "Summary of secondary needs",
    "clusters": [
      {"cluster": "Exit/Sell Timeshare", "count": 50, "implication": "..."}
    ]
  }
}
```

## LLM Prompt Guidelines Added

New guidelines 12-15 in `generate_insights.py`:

12. **Training analysis - CONNECT THE DOTS**: Analyze WHY training gaps exist and how they connect to failures
13. **Cross-correlations**: Look for patterns spanning training + failure dimensions
14. **Emergent patterns**: Actively identify patterns NOT covered by existing categories
15. **Secondary intents analysis**: Cluster additional customer needs

## Migration Notes

- **No breaking changes** - v3.5 is backwards compatible
- If new fields are missing, sections render with fallback to raw counts (training) or don't render (emergent patterns, secondary intents)
- Existing v3.4 reports will render correctly (graceful degradation)

## Files Modified

| File | Changes |
|------|---------|
| `tools/extract_nl_fields.py` | Added `training_details` and `all_additional_intents` extraction |
| `tools/generate_insights.py` | Added training data to prompt, new schema fields, guidelines 12-15 |
| `tools/render_report.py` | Restructured Training section, added Emergent Patterns and Secondary Needs sections |
| `tools/test_v35_features.py` | **NEW** - Unit tests for v3.5 features |

## Running Tests

```bash
# Run v3.5 feature tests
python3 tools/test_v35_features.py

# Run full test suite
python3 tools/test_framework.py
```

## Verification

```bash
# Re-run with existing analyses
python3 tools/run_analysis.py --skip-sampling --skip-analysis

# Verify training section has narrative
grep -A 20 "## Training & Development" reports/executive_summary_v3_*.md

# Verify emergent patterns appear
grep -A 10 "## Emergent Patterns" reports/executive_summary_v3_*.md

# Verify cross-correlations
grep "Cross-Dimensional" reports/executive_summary_v3_*.md
```
