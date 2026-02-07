# v5.0 — Orthogonal Disposition Model

## What Changed

v5.0 replaces the single `disposition` enum with two independent dimensions: **call_scope** (what was the request about?) and **call_outcome** (how did the call end?). This eliminates the combinatorial explosion that forced v4.x to pack scope + outcome into compound enum values like `in_scope_success` and `out_of_scope_failed`.

### Old Model (v4.5)

```
disposition: in_scope_success | in_scope_partial | in_scope_failed | escalated | out_of_scope_handled | out_of_scope_failed | pre_intent
```

7 values conflating two dimensions. Adding a new scope or outcome required adding multiple new enum values.

### New Model (v5.0)

```
call_scope:   in_scope | out_of_scope | mixed | no_request
call_outcome: completed | escalated | abandoned
```

4 x 3 = 12 theoretical combinations (11 valid — `no_request:completed` is invalid by design).

## Why

1. **Cleaner analytics**: Scope and outcome are independent analytical dimensions. Grouping by scope answers "what do callers want?", grouping by outcome answers "what happened?"
2. **Containment rate**: The key operational metric (`in_scope:completed / in_scope:total`) falls out naturally.
3. **Mixed scope**: Calls with both in-scope and out-of-scope requests now get their own category instead of being forced into one bucket.
4. **Extensibility**: Adding a new scope or outcome is one enum value, not N combinations.

## Schema Changes

### New Fields

| Field | Type | When Populated |
|-------|------|----------------|
| `call_scope` | enum | Always. `in_scope`, `out_of_scope`, `mixed`, `no_request` |
| `call_outcome` | enum | Always. `completed`, `escalated`, `abandoned` |
| `escalation_trigger` | enum \| null | When `call_outcome = escalated` |
| `abandon_stage` | enum \| null | When `call_outcome = abandoned` |

### Conditional Qualifiers

Each outcome has one conditional field:

| Outcome | Qualifier | Values |
|---------|-----------|--------|
| `completed` | `resolution_confirmed` | `true` (customer confirmed) \| `false` (action taken, unconfirmed) |
| `escalated` | `escalation_trigger` | `customer_requested` \| `scope_limit` \| `task_failure` \| `policy_routing` |
| `abandoned` | `abandon_stage` | `pre_greeting` \| `pre_intent` \| `mid_task` \| `post_delivery` |

### Removed Fields

| Field | Reason |
|-------|--------|
| `disposition` | Replaced by `call_scope` + `call_outcome` (bridge function available) |
| `escalation_initiator` | Replaced by `escalation_trigger` (why, not who) |
| `escalation_requested` | Covered by `escalation_trigger = customer_requested` |
| `pre_intent_subtype` | Generalized to `abandon_stage` (applies to all abandoned calls) |
| `abandoned_path_viable` | Covered by `call_scope` + `abandon_stage` |
| `failure_recoverable` | Covered by `failure_type` + `call_outcome` |
| `failure_critical` | Rare, covered by `coaching` |

### Kept Unchanged

`resolution_confirmed`, `actions[]`, `transfer_destination`, `transfer_queue_detected`, and all quality/friction/insight fields.

## Mapping from v4.5

| v4.5 `disposition` | v5.0 `call_scope` | v5.0 `call_outcome` |
|--------------------|--------------------|---------------------|
| `in_scope_success` | `in_scope` | `completed` |
| `in_scope_partial` | `in_scope` | `completed` (rc=false) |
| `in_scope_failed` | `in_scope` | `abandoned` |
| `escalated` | `in_scope` or `mixed` | `escalated` |
| `out_of_scope_handled` | `out_of_scope` | `completed` |
| `out_of_scope_failed` | `out_of_scope` | `abandoned` |
| `pre_intent` | `no_request` | `abandoned` |

## Bridge Function

For v4.x consumers, `get_disposition()` in `analyze_transcript.py` synthesizes the legacy disposition field:

```python
def get_disposition(call_scope, call_outcome, resolution_confirmed):
    if call_scope == "no_request":
        return "pre_intent"
    if call_outcome == "escalated":
        return "escalated"
    if call_scope == "out_of_scope":
        return "out_of_scope_handled" if call_outcome == "completed" else "out_of_scope_failed"
    if call_outcome == "completed":
        return "in_scope_success" if resolution_confirmed else "in_scope_partial"
    return "in_scope_failed"
```

## Post-Analysis Enforcement

The LLM occasionally sets conditional fields for the wrong outcome. Post-analysis code enforces:
- `resolution_confirmed` → `null` unless `call_outcome == completed`
- `escalation_trigger` → `null` unless `call_outcome == escalated`
- `abandon_stage` → `null` unless `call_outcome == abandoned`

## Key Metric: Containment Rate

```
containment_rate = in_scope:completed / in_scope:total
```

This measures "of calls where the agent can help, how often does it actually resolve the issue?" — the most actionable operational metric.

## Validation

Validated on 200 transcripts from `run_20260204_234844` with zero schema errors. Golden test set expanded to 23 transcripts covering 9 of 12 valid scope x outcome combinations (see `tests/golden/manifest.json`).

## SQL Analytics

```bash
python3 tools/load_duckdb.py runs/run_XXXX/analyses/
python3 tools/query.py runs/run_XXXX/ --dashboard
```

Dashboard queries include scope x outcome cross-tab, containment rate, escalation trigger breakdown, and abandon stage analysis.

## Migration Notes

- **Schema version**: `v5.0` (set automatically by `analyze_transcript.py`)
- **Re-analysis required**: Existing v4.x analyses must be re-analyzed to get v5.0 fields
- **Bridge compatibility**: `disposition` field is still emitted via bridge function for v4.x consumers
- **DuckDB**: `load_duckdb.py` handles both v4.x and v5.0 schemas
