# v4.5 — Performance Dashboard Fields

## What Changed

v4.5 adds 7 new fields to the per-transcript analysis schema, enabling a performance dashboard with finer-grained resolution tracking, action performance metrics, and transfer quality data. It also introduces a DuckDB analytics layer for flexible SQL-based querying.

## Why

The Phonely Concierge v2.0 dashboard requires metrics that go beyond the existing disposition/intent/sentiment data:
- **Confirmed vs. unconfirmed resolution** — did the customer actually acknowledge the outcome?
- **Pre-intent subtypes** — why did pre-intent calls end (silence, abandoned, escalated)?
- **Escalation initiator** — who drove the escalation (user or agent)?
- **Abandonment quality** — was a viable path available when the customer left?
- **Per-action tracking** — success/failure rates for each agent tool (account lookup, verification, link sends)
- **Transfer metadata** — destination and queue detection

## New Schema Fields

| Field | Type | When Populated |
|-------|------|----------------|
| `resolution_confirmed` | bool \| null | disposition = in_scope_success or in_scope_partial |
| `pre_intent_subtype` | enum \| null | disposition = pre_intent |
| `escalation_initiator` | enum \| null | disposition = escalated |
| `abandoned_path_viable` | bool \| null | in_scope_failed with ended_by=customer |
| `actions` | array of objects | always (empty array if none) |
| `transfer_destination` | string \| null | transfer occurred |
| `transfer_queue_detected` | bool | always (default false) |

### actions array entries

```json
{
  "action": "account_lookup | verification | send_payment_link | send_portal_link | send_autopay_link | send_rental_link | send_clubhouse_link | send_rci_link | transfer | other",
  "outcome": "success | failed | retry | unknown",
  "detail": "5-10 words telegraph style"
}
```

## New Dashboard Metrics

### Call Funnel (MECE)
- Pre-Intent Exit (by subtype: abandoned, silence, escalated, other)
- Intent Captured → Out-of-Scope vs In-Scope

### In-Scope Outcomes
- AI Resolved (confirmed vs unconfirmed)
- Escalated (agent-initiated vs user-initiated)
- Abandoned (path viable vs dead-end)
- Failed (non-abandon)

### Action Performance
- Per-action-type: attempted, success, retry, failed, unknown + rates

### Transfer Quality
- Total transfers, by destination, queue detection rate

## New Tools

### `tools/load_duckdb.py` — DuckDB Analytics Loader
Loads analysis JSONs into a DuckDB database with `calls` table + 4 detail views (`call_actions`, `call_clarifications`, `call_corrections`, `call_loops`).

```bash
python3 tools/load_duckdb.py runs/run_XXXX/analyses/
# Creates: runs/run_XXXX/analytics.duckdb
```

### `tools/query.py` — SQL Query CLI
Run SQL queries against a run's DuckDB database. Includes predefined dashboard queries.

```bash
python3 tools/query.py runs/run_XXXX/ --dashboard
python3 tools/query.py runs/run_XXXX/ -q "SELECT action, COUNT(*) FROM call_actions GROUP BY 1"
```

### `tools/validate_v45_schema.py` — Schema Validator
Validates analysis JSONs against all v4.5 field constraints.

```bash
python3 tools/validate_v45_schema.py runs/run_XXXX/analyses/
python3 tools/validate_v45_schema.py runs/run_XXXX/analyses/ --check-funnel
```

## Files Modified

| File | Change |
|------|--------|
| `tools/analyze_transcript.py` | +7 schema fields, classification guidance, validation rules, defaults |
| `tools/compute_metrics.py` | +4 aggregation functions (call_funnel, in_scope_outcomes, action_performance, transfer_quality) |
| `tools/ask.py` | v4.5 fields in format_call_for_prompt + system prompt |
| `tools/extract_nl_fields.py` | v4.5 fields in extract_condensed_call |
| `tools/validate_v45_schema.py` | New — schema validation script |
| `tools/load_duckdb.py` | New — JSON→DuckDB loader |
| `tools/query.py` | New — SQL query CLI |
| `CLAUDE.md` | Version bump to v4.5, SQL analytics section |
| `docs/BACKLOG.md` | New — deferred items |

## Dependency

`duckdb` — install with `pip install duckdb` (~15MB, pure Python, no server required). Only needed for `load_duckdb.py` and `query.py`; the rest of the pipeline has no new dependencies.
