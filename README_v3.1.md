# Voice Agent Analytics - v3.1 Release Notes

**Released**: 2026-01-18

## What's New in v3.1

### Dedicated NL Field Extraction

**New file**: `tools/extract_nl_fields.py`

Previously, natural language fields were extracted inline during metrics computation. v3.1 introduces a dedicated extraction step that:

- Produces condensed `nl_summary_v3_{timestamp}.json` format
- Achieves ~70% size reduction from full analyses
- Groups data optimally for LLM context:
  - `by_failure_type` - Failures grouped by type
  - `all_verbatims` - Customer quotes
  - `all_agent_misses` - Coaching opportunities
  - `policy_gap_details` - Structured gap analysis
  - `failed_call_flows` - Resolution step sequences

### Updated Pipeline

```
v3.0: sample → analyze → metrics → insights → report
v3.1: sample → analyze → metrics → extract_nl → insights → report
                                      ↑ NEW
```

### Test Harness

**New file**: `tools/test_framework.py`

Minimal test harness (stdlib only, no pytest) with 17 tests across 4 categories:
- Schema validation (4 tests)
- Metrics computation (4 tests)
- NL extraction (4 tests)
- Integration (5 tests)

**New file**: `tools/test_data/expected_metrics.json`

Ground truth for test assertions based on 5 test v3 analyses.

### Files Changed

| File | Change |
|------|--------|
| `tools/extract_nl_fields.py` | **NEW** - Dedicated NL extraction |
| `tools/test_framework.py` | **NEW** - Test harness |
| `tools/test_data/expected_metrics.json` | **NEW** - Test ground truth |
| `tools/run_analysis.py` | Added Step 4: extract_nl_fields |
| `tools/generate_insights.py` | Now reads from nl_summary_v3_*.json |
| `tools/compute_metrics.py` | Removed --export-nl-fields flag |

## Migration from v3.0

No breaking changes. Existing v3 analyses are fully compatible.

To use the new extraction step manually:
```bash
python3 tools/extract_nl_fields.py
```

Or run the full pipeline which now includes it automatically:
```bash
python3 tools/run_analysis.py
```

## Test Data

5 test v3 analyses included in `analyses/`:
- `test-v3-policy-gap-001.json` - capability_limit
- `test-v3-policy-gap-002.json` - data_access
- `test-v3-nlu-miss-001.json` - NLU failure
- `test-v3-resolved-001.json` - successful call
- `test-v3-auth-fail-001.json` - auth_restriction

## Verification

```bash
# Run test harness
python3 tools/test_framework.py
# Expected: 17/17 tests passed

# Test NL extraction
python3 tools/extract_nl_fields.py
# Expected: 5 v3 analyses, 4 calls with NL data
```
