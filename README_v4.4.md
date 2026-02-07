# v4.4 Release Notes — Handle Time / AHT

## What Changed

**New field**: `duration_seconds` (float, nullable) — schema grows from 22 to 23 fields.

Every transcript JSON already contains a root-level `duration` field (seconds, float). v4.4 injects this into the analysis output deterministically — no LLM extraction needed, zero additional cost.

### Schema Change

```json
{
  "call_id": "uuid",
  "schema_version": "v4.4",
  "duration_seconds": 161.6,
  ...
}
```

### New Metric: `handle_time`

`compute_metrics.py` now produces a `handle_time` section:

```json
{
  "handle_time": {
    "overall": {"n": 50, "mean": 132.5, "median": 118.0, "std": 67.3, "min": 12.4, "max": 412.8},
    "by_disposition": {
      "in_scope_success": {"n": 18, "mean": 98.2, "median": 89.0, ...},
      "in_scope_failed": {"n": 12, "mean": 185.4, "median": 172.0, ...},
      "escalated": {"n": 8, "mean": 201.1, "median": 195.0, ...},
      "pre_intent": {"n": 5, "mean": 22.3, "median": 18.0, ...}
    }
  }
}
```

## Why

An expert schema assessment (see `docs/v4x_schema_assessment.md`) identified AHT as the #1 gap. `turns` serves as a complexity proxy but doesn't capture actual call duration. Operations leadership needs AHT for capacity planning, SLA tracking, and per-disposition cost analysis.

## Implementation Details

- **Deterministic injection**: `duration` read from raw transcript JSON, rounded to 1 decimal, injected into analysis dict after LLM call
- **Zero LLM cost**: No prompt changes, no additional API calls
- **Backwards compatible**: Existing v4.0-v4.3 analyses still load fine (field is simply absent)
- **Nullable**: `null` if transcript lacks duration data

### Files Modified

| File | Change |
|------|--------|
| `tools/analyze_transcript.py` | Read `duration` from transcript metadata, inject `duration_seconds`, bump `schema_version` to `v4.4` |
| `tools/compute_metrics.py` | New `compute_handle_time_stats()` — overall AHT + per-disposition breakdown using `safe_stats()` |
| `README.md` | Field count 22→23, v4.4 in version table |
| `CLAUDE.md` | Updated schema version reference |
| `docs/v4x_schema_assessment.md` | New expert schema assessment document |

## Migration Notes

- No breaking changes — v4.0-v4.3 analyses remain fully compatible
- New analyses will have `schema_version: "v4.4"` and `duration_seconds` populated
- `compute_metrics.py` gracefully handles mixed-version runs (only computes AHT for analyses that have the field)
- To backfill existing analyses, re-run `batch_analyze.py` on the same run
