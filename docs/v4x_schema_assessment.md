# Voice Agent Analytics v4.x — Expert Schema Assessment

## Schema: 22 Fields Across 6 Domains

The v4.0 schema (unchanged through v4.1 and v4.3) captures:

| Domain | Fields | Highlights |
|--------|--------|-----------|
| **Call Metadata** (2) | `turns`, `ended_by` | Turn count as complexity proxy |
| **Customer Intent** (3) | `intent`, `intent_context`, `secondary_intent` | Separates WHAT from WHY — rare in the industry |
| **Resolution/Disposition** (3) | `disposition` (7 values), `resolution`, `steps` | Funnel-ready: pre_intent → in_scope_success/partial/failed → escalated |
| **Quality & Sentiment** (5) | `effectiveness`, `quality`, `effort`, `sentiment_start`, `sentiment_end` | CES-equivalent + emotional journey tracking |
| **Failure Analysis** (5) | `failure_type` (6 causes), `failure_detail`, `failure_recoverable`, `failure_critical`, `policy_gap` | Structured root cause with policy gap decomposition |
| **Friction Events** (4) | `derailed_at`, `clarifications[]`, `corrections[]`, `loops[]` | Turn-level granularity with typed causes and subjects |

Plus narrative fields: `summary`, `verbatim`, `coaching`, `escalation_requested`, `repeat_caller`.

## Key Strengths vs Industry

- **Friction at event level** — most QA platforms score at call level; this tracks individual clarifications, corrections, and loops at the turn number
- **7-value disposition funnel** — distinguishes confirmed success from "completed but unconfirmed" (partial), and in-scope vs out-of-scope failures
- **Intent + context separation** — captures both the request and the underlying situation
- **Per-call coaching** — actionable feedback as a first-class field, not an afterthought
- **Progressive pipeline** — `--sample-only` (zero LLM cost) → default (N analysis calls) → `--insights` (full report)

## Schema Evolution (v3.6 → v4.0)

v4.0 was a deliberate cleanup: **+4 fields** (intent, intent_context, sentiment_start/end), **merged 2→1** (outcome + call_disposition → disposition), **flattened** friction from nested object to top-level, **renamed 12+ fields** for brevity. v4.1 and v4.3 changed zero schema fields — purely infrastructure (run isolation, target-based augmentation).

## Priority Gaps Identified

1. **Handle Time / AHT** — the #1 gap. `turns` is a proxy; if timestamps exist in transcripts, actual duration should be computed deterministically
2. **First Contact Resolution (FCR)** — `repeat_caller` is self-reported; true FCR needs cross-session linking by customer ID
3. **Transfer destination tracking** — `escalated` captures that it happened, but not where or whether it succeeded
4. **Compliance/QA structure** — `failure_critical` is a boolean flag; regulated industries need structured checkpoint tracking
5. **Time-series comparison** — run isolation enables it, but no automated trend tool exists yet
6. **Segment dimensions** — no built-in support for time-of-day, property, or customer-type drill-downs

## Bottom Line

The framework is significantly more sophisticated than typical call center analytics tooling. The combination of structured per-call LLM analysis, deterministic aggregation, and narrative insights creates layered capability most commercial platforms lack. Priority additions should be AHT (if timestamp data exists) and FCR — the two metrics operations leadership will ask for first.
