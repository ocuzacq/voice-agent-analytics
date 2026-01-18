# Voice Agent Analytics v3.3 Release Notes

**Release Date:** 2026-01-18

## Summary

v3.3 focuses on **report quality improvements** addressing four key issues identified during executive report review:

1. **Semantic duplicates** in customer needs
2. **Coarse failure categories** lacking context
3. **Confusing policy gap categories** without explanations
4. **Insights lacking traceability** to specific calls

Additional fixes:
5. **Model upgrade**: Gemini 3 Flash/Pro models
6. **Scope coherence**: Metrics/NL extraction now respect manifest.csv
7. **Token limit**: Increased to 32000 for large datasets

## What's New

### 1. Validation Bug Fix

**File:** `tools/compute_metrics.py`

Added `validate_failure_consistency()` function that flags data quality issues:
- Detects `failure_point='none'` for non-resolved calls (abandoned, escalated, unclear)
- Returns warnings in `validation_warnings.failure_point_inconsistencies`
- Helps identify LLM analysis inconsistencies for correction

```python
# New function
validate_failure_consistency(analyses) -> dict

# New field in metrics output
{
  "validation_warnings": {
    "failure_point_inconsistencies": [
      {"call_id": "abc", "outcome": "abandoned", "issue": "failure_point='none' invalid for non-resolved call"}
    ]
  }
}
```

### 2. Call ID References Throughout

**File:** `tools/generate_insights.py`

All NL data in the LLM prompt now includes `call_id` for traceability:
- Failure entries: `[call_id] [outcome] description`
- Verbatims: `[call_id] [outcome] "quote"`
- Policy gaps: `[call_id] [category] Gap: ... | Ask: ...`
- Failed call flows: `[call_id] [outcome] steps`

**New schema fields:**
- `actionable_recommendations[].supporting_call_ids` - 2-5 example calls per recommendation
- `failure_type_explanations[].example_call_ids`
- `policy_gap_explanations[].example_call_ids`
- `customer_ask_clusters[].example_call_ids`

### 3. Explanatory Qualifiers

**Files:** `tools/generate_insights.py`, `tools/render_report.py`

**Failure Type Explanations:**
```json
{
  "failure_type_explanations": {
    "policy_gap": {
      "explanation": "System limitations prevented resolution",
      "common_patterns": "Transfer failures, payment restrictions",
      "example_call_ids": ["call1", "call2"]
    }
  }
}
```

**Policy Gap Category Explanations:**
```json
{
  "policy_gap_explanations": {
    "capability_limit": {
      "definition": "Features not yet built",
      "this_run_context": "Primarily transfer-to-human and payment processing",
      "top_gaps": ["live agent transfer", "payment processing"],
      "example_call_ids": ["call1", "call2"]
    }
  }
}
```

Rendered in the report after breakdown tables with italic context:
```markdown
### Failure Point Breakdown
| Failure Type | Count | % of Failures |
|--------------|-------|---------------|
| policy_gap | 62 | 62.0% |

*policy_gap: System limitations prevented resolution*
```

### 4. Semantic Clustering of Customer Asks

**Files:** `tools/generate_insights.py`, `tools/render_report.py`

Near-duplicate customer asks are now grouped semantically:

**Before (v3.2):**
```
1. speak to a representative - 5 calls
2. speak with a representative - 3 calls
3. speak to a live agent - 2 calls
```

**After (v3.3):**
```
### Top Unmet Customer Needs (Clustered)
1. **Request human agent transfer** - 10 calls
   *Examples: "speak to a representative", "speak with a rep", "talk to a live agent"*
```

**New schema:**
```json
{
  "customer_ask_clusters": [
    {
      "canonical_label": "Request human agent transfer",
      "member_asks": ["speak to a representative", "speak with a rep", "talk to a live agent"],
      "total_count": 10,
      "example_call_ids": ["call1", "call2", "call3"]
    }
  ]
}
```

## Migration Notes

### Backward Compatibility

- All new fields are **additive** - no breaking changes
- Existing v3 analyses work unchanged
- `validation_warnings` is informational only
- Clustered asks supplement (don't replace) raw metrics

### Running the Pipeline

No changes to the pipeline command:

```bash
# Full pipeline
python3 tools/run_analysis.py -n 200

# Skip sampling/analysis, regenerate insights only
python3 tools/run_analysis.py --skip-sampling --skip-analysis
```

### Verification

```bash
# Run v3.3 unit tests
python3 -m pytest tools/test_v33_features.py -v

# Check for validation warnings in output
grep "failure_point_inconsistencies" reports/metrics_v3_*.json

# Check for clustered asks in report
grep -A2 "Clustered" reports/executive_summary_v3_*.md

# Check for explanatory qualifiers
grep -E "^\*[a-z_]+:" reports/executive_summary_v3_*.md
```

## Updated LLM Prompt Schema

Full v3.3 output schema:

```json
{
  "executive_summary": "...",
  "root_cause_analysis": {...},
  "actionable_recommendations": [
    {
      "priority": "P0|P1|P2",
      "category": "capability|training|prompt|process",
      "recommendation": "...",
      "expected_impact": "...",
      "evidence": "...",
      "supporting_call_ids": ["id1", "id2"]  // NEW
    }
  ],
  "trend_narratives": {...},
  "verbatim_highlights": {...},

  // NEW v3.3 sections:
  "failure_type_explanations": {
    "<type>": {
      "explanation": "...",
      "common_patterns": "...",
      "example_call_ids": [...]
    }
  },
  "policy_gap_explanations": {
    "<category>": {
      "definition": "...",
      "this_run_context": "...",
      "top_gaps": [...],
      "example_call_ids": [...]
    }
  },
  "customer_ask_clusters": [
    {
      "canonical_label": "...",
      "member_asks": [...],
      "total_count": N,
      "example_call_ids": [...]
    }
  ]
}
```

### 5. Model Upgrade

**Files:** `tools/analyze_transcript.py`, `tools/batch_analyze.py`, `tools/generate_insights.py`, `tools/run_analysis.py`

Upgraded to Gemini 3 models with different tiers for analysis vs insights:

| Purpose | Model | Why |
|---------|-------|-----|
| Transcript analysis | `gemini-3-flash-preview` | Fast, cost-effective for high-volume analysis |
| Insights generation | `gemini-3-pro-preview` | Higher quality for executive-facing output |

```bash
# run_analysis.py now has separate model arguments
python3 tools/run_analysis.py -n 500 \
    --analysis-model gemini-3-flash-preview \
    --insights-model gemini-3-pro-preview
```

### 6. Run Isolation (Scope Coherence)

**Files:** `tools/compute_metrics.py`, `tools/extract_nl_fields.py`, `tools/run_analysis.py`

**Problem:** Previous runs accumulated analyses in `analyses/`, causing:
- batch_analyze to skip samples that were previously analyzed (73 out of 500 samples!)
- Metrics/NL extraction to mix old and new data

**Solution - Two-layer fix:**

1. **Clear analyses/ on fresh runs** (run_analysis.py):
   - `analyses/` is now cleared before batch analysis (unless `--resume` or `--no-clear-analyses`)
   - Ensures each run starts with a clean slate

2. **Manifest filtering** (compute_metrics.py, extract_nl_fields.py):
   - Both scripts now filter by `manifest.csv` from sampled directory
   - Only includes analyses matching the current run's samples

```bash
# Fresh run (clears analyses/, full isolation)
python3 tools/run_analysis.py -n 500

# Resume interrupted run (preserves analyses/)
python3 tools/run_analysis.py --resume

# Keep old analyses (for incremental runs)
python3 tools/run_analysis.py -n 500 --no-clear-analyses

# Manual usage with scope filtering
python3 tools/compute_metrics.py -s sampled/  # Uses manifest.csv
python3 tools/extract_nl_fields.py -s sampled/

# Disable scope filtering (include all analyses)
python3 tools/compute_metrics.py --no-scope-filter
```

### 7. Token Limit Increase

**File:** `tools/generate_insights.py`

Increased `max_output_tokens` from 4096 to 32000 to handle large datasets with the expanded v3.3 schema.

## Files Modified

| File | Changes |
|------|---------|
| `tools/compute_metrics.py` | Added `validate_failure_consistency()`, `validation_warnings`, scope filtering |
| `tools/generate_insights.py` | Updated prompt schema, call_ids, gemini-3-pro-preview, 32K tokens |
| `tools/render_report.py` | Clustered asks, explanatory qualifiers, call_ids in recommendations |
| `tools/extract_nl_fields.py` | Added scope filtering via manifest.csv |
| `tools/analyze_transcript.py` | Default model: gemini-3-flash-preview |
| `tools/batch_analyze.py` | Default model: gemini-3-flash-preview |
| `tools/run_analysis.py` | Split --model, pass -s to metrics/NL, clear analyses/ on fresh runs |
| `tools/test_v33_features.py` | **NEW** - Unit tests for v3.3 features |
| `README_v3.3.md` | **NEW** - This file |

## Testing

Run the full test suite:

```bash
# v3.3 feature tests
python3 -m pytest tools/test_v33_features.py -v

# Existing test framework
python3 tools/test_framework.py
```

## Known Limitations

1. **Clustering quality** depends on LLM interpretation - may occasionally miss semantic similarities
2. **Call ID references** require v3 schema analyses - older v2 analyses may show "unknown" placeholders
3. **Explanatory qualifiers** only appear when LLM generates them - fallback to no qualifier if missing
