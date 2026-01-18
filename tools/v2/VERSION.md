# Version 2 (v2) - Simplified Actionable Schema

**Date:** 2026-01-18
**Status:** Previous (Superseded by v3)

## Philosophy

80% of insights with 80% less verbosity. Every field must:
1. **Drive a decision** (training, product fix, escalation policy)
2. **Aggregate meaningfully** across 100s of calls
3. **Be orthogonal** (measure distinct things, not correlated variants)

## Schema (14 fields)

```json
{
  "call_id": "uuid",
  "schema_version": "v2",

  "outcome": "resolved | escalated | abandoned | unclear",
  "resolution_type": "payment processed | callback scheduled | info provided | ...",

  "agent_effectiveness": 4,     // 1-5: Did agent understand and respond appropriately?
  "conversation_quality": 4,    // 1-5: Flow, tone, clarity combined
  "customer_effort": 2,         // 1-5: How hard did customer work? (1=effortless)

  "failure_point": "nlu_miss",  // none | nlu_miss | wrong_action | policy_gap | customer_confusion | tech_issue | other
  "failure_description": "Agent confused 'ownership' with 'payment account'",
  "was_recoverable": true,
  "critical_failure": false,    // Wrong info, impossible promise, compliance issue

  "escalation_requested": false,
  "repeat_caller_signals": false,
  "training_opportunity": "verification_flow",

  "additional_intents": "Also asked about cancellation policy",

  "summary": "Customer called to make payment. Resolved after brief verification."
}
```

## What v2 Consolidates

| v1 Fields | v2 Field | Rationale |
|-----------|----------|-----------|
| nlu_accuracy, response_quality.relevance, response_quality.helpfulness | `agent_effectiveness` | Highly correlated |
| conversational_flow (3), tone_and_style (4), verbosity | `conversation_quality` | All measure "good conversation" |
| efficiency (4 fields), satisfaction components | `customer_effort` | Inverted lens - more actionable |
| issues[] array with severity | `failure_point` + `failure_description` | One primary cause matters |
| 9 agent_quality sub-scores | 2 composite scores | Sub-scores don't drive different actions |

## What v2 Drops

| Cut | Why |
|-----|-----|
| Customer profile (age, tech comfort, emotions) | Not actionable - can't train agent per demographic |
| Transcription quality | Measures ASR, not agent - separate analysis |
| Funnel stages | Derivable from outcome + resolution_type |
| actions_claimed vs completion_evidence | Captured in resolution_type |
| Coverage field | Derivable from outcome |

## Aggregate Queries This Enables

```sql
-- Success rate by week
SELECT week, AVG(outcome = 'resolved') FROM calls GROUP BY week

-- Failure root causes
SELECT failure_point, COUNT(*) FROM calls WHERE outcome != 'resolved' GROUP BY failure_point

-- High-effort successes (resolved but painful)
SELECT * FROM calls WHERE outcome = 'resolved' AND customer_effort >= 4

-- Training backlog
SELECT training_opportunity, COUNT(*) FROM calls WHERE training_opportunity IS NOT NULL GROUP BY training_opportunity

-- Repeat caller patterns
SELECT resolution_type, COUNT(*) FROM calls WHERE repeat_caller_signals = true GROUP BY resolution_type
```

## Usage

```bash
# From project root
export GOOGLE_API_KEY="..."
python3 tools/v2/batch_analyze.py
python3 tools/v2/compute_metrics.py
```

## Comparison

| Aspect | v0 | v1 | v2 | v3 |
|--------|----|----|----|-----|
| Fields | ~15 | ~50+ | 14 | 18 |
| Token cost | Low | High | Low | Low |
| Actionability | Medium | High (noisy) | High (focused) | High (focused + insights) |
| Setup complexity | Low | High | Low | Low |
| Report output | JSON | JSON | JSON | JSON + Markdown |

**Note:** v3 extends v2 with 4 new fields (policy_gap_detail, customer_verbatim, agent_miss_detail, resolution_steps) and adds LLM-powered executive insights.
