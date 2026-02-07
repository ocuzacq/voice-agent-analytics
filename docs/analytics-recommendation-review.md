# Principal Engineer Review: `analytics_recommendation.md` vs Existing Framework

## TL;DR

The recommendation proposes a **production logging schema** (`fact_calls`) for a system that doesn't exist yet — backend-generated structured events written to S3. Our framework solves a fundamentally different (harder) problem: **extracting structured analytics from unstructured transcripts** using LLM analysis. The recommendation is a good north-star for what the *backend should eventually emit*, but as a guide for what to build *next*, it misses ~60% of what we already have and underspecifies the areas that actually matter.

---

## 1. Architectural Mismatch

| | Recommendation (`fact_calls`) | Our Framework (v4.4) |
|---|---|---|
| **Data source** | Backend emits structured JSON at call end | Raw transcripts (unstructured conversation logs) |
| **Analysis method** | Application code decides outcomes | LLM extracts intent, disposition, friction, coaching |
| **When computed** | Real-time, at call hangup | Post-hoc, batch pipeline |
| **Storage** | S3 → Parquet → DuckDB | Local JSON → run-based isolation |
| **Strengths** | Fast queries, standardized enums, no LLM cost | Deep qualitative analysis, friction events, coaching insights |

**Verdict**: These are complementary, not competing. The recommendation is an *operational telemetry* layer. Our framework is an *analytical intelligence* layer. Both are needed; neither replaces the other.

---

## 2. Field-by-Field Gap Analysis

### What the recommendation proposes that we ALREADY HAVE (and often better)

| Rec Field | Our Equivalent | Our Advantage |
|---|---|---|
| `call_id` | `call_id` | Same |
| `duration_seconds` | `duration_seconds` (v4.4) | Same — injected from raw transcript |
| `primary_intent` | `intent` + `intent_context` + `secondary_intent` | **Richer** — we separate WHAT from WHY, and track secondary intents |
| `outcome_category` (3 values) | `disposition` (7 values) | **Much richer** — distinguishes confirmed success vs partial, in-scope vs out-of-scope failures, pre-intent dropoffs |
| `outcome_sub_category` | Folded into `disposition` enum | Our 7-value enum covers the same ground without a second field |
| `abandonment_stage` | `derailed_at` (turn number) + `disposition=pre_intent` | **More precise** — exact turn number, not coarse bucket |
| `abandonment_timing` (EARLY/MID/LATE) | Computable from `duration_seconds` + `turns` | We could derive this trivially but haven't needed to |
| `is_in_scope` | Implicit in `disposition` (`out_of_scope_*` vs `in_scope_*`) | Folded into disposition — cleaner single field |
| `quality_metrics.interruption_count` | `corrections[]` with turn-level detail | **Richer** — we track severity, cause, and context per event |
| `quality_metrics.agent_repeat_requests` | `clarifications[]` with typed causes | **Richer** — we distinguish name/phone/intent/repeat/verify with cause codes |

### What the recommendation proposes that we DON'T HAVE

| Rec Field | Gap Severity | Notes |
|---|---|---|
| `customer_id` | **HIGH** — blocks FCR | Not in raw transcripts. Requires upstream change to voice agent logging. |
| `agent_version` | **MEDIUM** | Raw transcripts have `agent_id` but no version/prompt variant tag. Needed for A/B testing prompt changes. |
| `actions_executed[]` (structured) | **MEDIUM** | Our `steps[]` is LLM-extracted free text. The rec wants backend-logged structured actions with latency/retry/error. Fundamentally different data source. |
| `transfer_details.destination` | **LOW-MEDIUM** | LLM could extract this from transcript text ("connect you to concierge"). Not structured in raw data. |
| `transfer_details.wait_time_seconds` | **LOW** | Computable from message timestamps during hold segments. |
| `quality_metrics.latency_p50/p90_ms` | **LOW** | Per-message timestamps exist in raw transcripts. Could compute inter-turn latency. |
| `quality_metrics.voice_speed_wpm` | **N/A** | Requires audio analysis, not available from text transcripts. |
| `quality_metrics.hallucination_detected` | **LOW** | Would require a separate verification pass (LLM or rule-based). |
| `call_category` (SILENCE, ROBO, WRONG_NUMBER) | **LOW** | Our `pre_intent` disposition + `turns` count covers most of this. Robo/wrong-number detection would need new logic. |
| `start_ts` / `end_ts` | **LOW** | Available from first/last message timestamps in raw transcripts. Not currently extracted. |

### What WE HAVE that the recommendation completely MISSES

| Our Field/Capability | Value | Why It Matters |
|---|---|---|
| **`intent_context`** | High | WHY the customer needs something — not just WHAT. Drives root cause analysis. |
| **`sentiment_start` → `sentiment_end`** | High | Emotional journey tracking. Shows whether the call made things better or worse. |
| **`coaching`** | High | Per-call actionable feedback. No equivalent in the rec — it only tracks what happened, not what *should* have happened. |
| **`verbatim`** | High | Direct customer quotes. Powers executive narratives and customer voice programs. |
| **`failure_type` taxonomy (6 values)** | High | Root cause classification. The rec only has `error_reason` on actions, missing NLU misses, policy gaps, customer confusion. |
| **`policy_gap` structured object** | High | 4-field decomposition (category, specific_gap, customer_ask, blocker). Critical for product roadmap prioritization. |
| **Friction event arrays** (clarifications, corrections, loops) | High | Turn-level granularity with typed causes and subjects. The rec has interrupt/repeat *counts* only — no detail. |
| **`effort` score (1-5)** | Medium | CES-equivalent. Industry-standard metric missing from rec. |
| **Loop subject tracking** | Medium | Knows *what* is being looped on (name, phone, transfer). The rec has no equivalent. |
| **`failure_recoverable`** | Medium | Could the agent have saved this? Drives training priorities. |
| **LLM-powered insights pipeline** | High | Executive summaries, recommendations, cross-dimensional patterns. The rec is raw data only — no insight layer. |

---

## 3. What the Recommendation Gets Right

1. **DuckDB + Parquet for scale** — our JSON-file-per-analysis approach works for hundreds, not millions. If we scale to full corpus (36K+ transcripts), we'll need columnar storage.
2. **Backend-emitted structured events** — `actions_executed[]` with latency/retry/error is genuinely valuable and can't be reconstructed from transcripts alone.
3. **`customer_id` for FCR** — the #1 gap our expert assessment also identified. Cross-session linking requires upstream changes.
4. **`agent_version` for A/B testing** — critical for measuring prompt iteration impact. We have `agent_id` but not version.
5. **S3 partitioning by date** — good operational practice for when this moves to production.

## 4. What the Recommendation Gets Wrong or Underspecifies

1. **3-value outcome is too coarse** — `RESOLVED / ESCALATED / ABANDONED` loses the nuance of our 7-value disposition. "In-scope partial" vs "in-scope success" is a critical distinction for QA.
2. **No qualitative analysis** — no coaching, no verbatim, no sentiment journey, no root cause taxonomy. This is a *telemetry* schema, not an *analytics* schema.
3. **Flat quality metrics** — counts of interruptions/repeats without cause classification. Our typed clarifications (name/phone/intent with cause codes) are far more actionable.
4. **No policy gap analysis** — the rec tracks action-level errors (API_TIMEOUT) but not business-level gaps (can't process payments, can't update contact info). These are different failure modes.
5. **"True Resolution" SQL assumes `customer_id`** — which doesn't exist in our data. The 72-hour callback check is aspirational, not implementable today.
6. **No insight generation** — the rec stops at SQL queries. Our framework generates executive summaries, recommendations, and training priorities.

---

## 5. Recommended Path Forward

### Phase 1: Quick Wins (use what we have)

| Action | Effort | Value |
|---|---|---|
| Extract `start_ts`/`end_ts` from message timestamps | 1h | Enables time-of-day segmentation |
| Compute inter-turn latency stats from timestamps | 2h | Approximates response latency metrics |
| Extract transfer destination via LLM (add to schema) | 2h | Fills transfer gap without backend changes |
| Add `agent_id` to analysis output (deterministic injection like `duration_seconds`) | 30min | Enables per-agent performance comparison |

### Phase 2: Schema Enrichment (moderate effort)

| Action | Effort | Value |
|---|---|---|
| Add abandonment timing buckets (EARLY/MID/LATE from `duration_seconds`) | 1h | Matches rec's `abandonment_timing` |
| Add call category (SILENCE/ROBO/WRONG_NUMBER) to pre-intent calls | 2h | Better triage of non-productive calls |
| Export to Parquet/DuckDB for large-scale queries | 4h | Enables SQL-based exploration at corpus scale |

### Phase 3: Requires Upstream Changes (backend team)

| Action | Dependency | Value |
|---|---|---|
| Add `customer_id` to transcript logging | Voice agent backend | Unlocks FCR, repeat caller verification |
| Add `agent_version` / prompt variant tag | Voice agent backend | Unlocks A/B testing of prompt changes |
| Add structured `actions_executed[]` logging | Voice agent backend | Backend-level action tracking with latency/errors |

---

## 6. Bottom Line

**The recommendation is a reasonable v1 telemetry schema for a greenfield system.** But we're not greenfield — we have a sophisticated analytical framework that already exceeds what `fact_calls` proposes in most dimensions. The right move is to cherry-pick the genuinely missing pieces (timestamps, agent_id injection, transfer destination) rather than rebuild around a flatter, less capable schema.

The recommendation's biggest blind spot is assuming analytics = SQL on structured events. Our LLM-powered pipeline (intent extraction, sentiment tracking, coaching, friction analysis, policy gap decomposition) delivers insight that no amount of SQL on backend logs will match. These approaches are complementary: backend telemetry for operational dashboards, LLM analysis for strategic intelligence.
