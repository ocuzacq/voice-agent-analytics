# v3.9 Release Notes: Call Disposition Classification

## Overview

v3.9 adds a single `call_disposition` field that enables funnel analysis of call outcomes. This field classifies each call into one of six mutually exclusive categories based on customer intent, agent scope, and completion status.

## Problem Statement

Previous schema versions tracked outcomes (`resolved`, `escalated`, `abandoned`, `unclear`) and failures, but lacked a unified view of the call funnel that would enable:
- Understanding **why** calls don't result in confirmed satisfaction
- Distinguishing between out-of-scope limitations and in-scope failures
- Identifying pre-intent dropoff for IVR/routing optimization
- Measuring the gap between "completed" and "customer confirmed"

## New Schema Field

```json
{
  "call_disposition": "pre_intent | out_of_scope_handled | out_of_scope_abandoned | in_scope_success | in_scope_partial | in_scope_failed"
}
```

### Disposition Definitions

| Value | Definition | Detection Signals |
|-------|------------|-------------------|
| `pre_intent` | Call ends before customer states actionable request | ≤2 turns, greeting-only, hang-up before stating need |
| `out_of_scope_handled` | Request outside capabilities, agent explained alternatives | Policy gap mentioned, customer redirected/informed |
| `out_of_scope_abandoned` | Request outside capabilities, customer gave up | Policy gap + abandoned/escalated, no resolution |
| `in_scope_success` | Request handled, customer confirmed satisfaction | Explicit thanks, "that helps", "perfect", "got it" |
| `in_scope_partial` | Request handled, no explicit confirmation | Action completed but customer just said "okay" or hung up |
| `in_scope_failed` | Request was in-scope but couldn't be completed | Verification failed, system error, agent couldn't help |

### Decision Tree for LLM

```
1. Did customer state a specific, actionable request?
   NO → pre_intent
   YES → continue

2. Could the agent handle this type of request?
   NO → Was customer redirected/informed of alternatives?
        YES → out_of_scope_handled
        NO → out_of_scope_abandoned
   YES → continue

3. Did agent complete the requested action?
   NO → in_scope_failed
   YES → continue

4. Did customer express explicit satisfaction?
   ("thank you", "that helps", "perfect", "great")
   YES → in_scope_success
   NO → in_scope_partial
```

## Funnel Metrics

v3.9 computes three key funnel metrics:

| Metric | Formula | Purpose |
|--------|---------|---------|
| **In-Scope Success Rate** | `in_scope_success / (in_scope_success + in_scope_partial + in_scope_failed)` | Measures confirmed satisfaction among actionable calls |
| **Out-of-Scope Recovery Rate** | `out_of_scope_handled / (out_of_scope_handled + out_of_scope_abandoned)` | Measures graceful deflection quality |
| **Pre-Intent Rate** | `pre_intent / total` | Identifies possible IVR/routing issues |

## Report Output

### Disposition Breakdown Table

```markdown
## Call Disposition Breakdown (n=100)

| Disposition | Count | % |
|-------------|-------|---|
| In Scope Success | 35 | 35% |
| In Scope Partial | 25 | 25% |
| In Scope Failed | 10 | 10% |
| Out Of Scope Handled | 15 | 15% |
| Out Of Scope Abandoned | 10 | 10% |
| Pre Intent | 5 | 5% |

### Funnel Metrics
- **In-Scope Success Rate:** 58% (60 in-scope calls confirmed satisfaction)
- **Out-of-Scope Recovery:** 60% (25 out-of-scope calls handled gracefully)
- **Pre-Intent Rate:** 5% (possible IVR/routing issues)
```

### Disposition Insights Table

```markdown
### Disposition Insights

| Disposition | Root Cause | Recommendation |
|-------------|------------|----------------|
| In Scope Partial | No confirmation prompt | Add "Did that help?" at call end |
| In Scope Failed | Verification failures | Implement retry/bypass for verified customers |
| Out Of Scope Abandoned | No reservation capability | Add capability or clearer redirect |
```

## Actionable Insights by Disposition

| Disposition | Focus Area | Example Actions |
|-------------|------------|-----------------|
| `in_scope_partial` | Confirmation prompts | Add "Did that help?" / "Anything else?" |
| `in_scope_failed` | Training/capability gaps | Identify failure reasons, add capabilities |
| `out_of_scope_abandoned` | Capability expansion | Build requested features or improve redirects |
| `pre_intent` | IVR/routing optimization | Review greeting, reduce initial friction |

## Agent Capability Reference

The decision tree includes a scope reference to ensure consistent classification:

### IN-SCOPE (agent CAN handle)

**Information:**
- Maintenance fee amounts and due dates
- Account balance lookups
- Property/resort information
- Payment history/status

**Send Links (via SMS, email, or both):**
- Mortgage Easy Payment Link
- Maintenance Easy Payment Link
- Mortgage Auto Pay Update Link
- Maintenance Account History Link
- Rental Agreement Link
- Clubhouse Register Link
- RCI website link

**Verification:**
- Account verification via phone/name/state

**Callback/Transfer:**
- Transfer to concierge (on-hours)
- Route to IVR callback (off-hours)

### OUT-OF-SCOPE (agent CANNOT handle)

**Actions it cannot perform:**
- Process payments directly (only sends links)
- Update contact info (email, phone, address)
- Book/modify/cancel reservations
- Process RCI exchanges directly
- Week banking/rollover

**Transfers it cannot do:**
- Live transfers when teams are unavailable

**Complex Issues:**
- Deceased owner / heir access
- Disputed charges
- Tax breakdowns
- Website/technical troubleshooting
- International customers (non-US verification)

## Files Modified

| File | Changes |
|------|---------|
| `analyze_transcript.py` | New `call_disposition` field + decision tree in prompt |
| `compute_metrics.py` | `compute_disposition_breakdown()` function + funnel metrics |
| `extract_nl_fields.py` | `disposition_summary` extraction for LLM insights |
| `generate_insights.py` | `disposition_analysis` output section |
| `render_report.py` | Disposition breakdown table + funnel metrics |
| `test_v39_features.py` | **NEW**: 8 unit tests for v3.9 features |

## Backwards Compatibility

v3.9 is fully backwards compatible with earlier analyses:

- Analyses without `call_disposition` are counted as `unknown` in aggregation
- All existing metrics continue to work unchanged
- NL extraction handles missing disposition gracefully
- Report renders disposition section only when data is present

## Testing

```bash
# Run v3.9 feature tests
python3 tools/test_v39_features.py

# Test single analysis with new field
python3 tools/analyze_transcript.py transcripts/00cb31f0-*.txt --stdout | jq '.call_disposition'

# Run full pipeline
python3 tools/run_analysis.py --quick

# Check disposition breakdown in report
grep -A20 "Call Disposition" reports/executive_summary_v3_*.md
```

## Verification

```bash
# Verify schema includes new field
grep -n "call_disposition" tools/analyze_transcript.py

# Verify aggregation works
python3 -c "from tools.compute_metrics import compute_disposition_breakdown; print(compute_disposition_breakdown([{'call_disposition': 'in_scope_success'}]))"

# Run all tests
python3 tools/test_v39_features.py
```

## Success Criteria

| Criteria | Status |
|----------|--------|
| New field populated for all analyses | 6 valid values + unknown |
| Decision tree in prompt | Step-by-step classification |
| Scope reference in prompt | IN-SCOPE / OUT-OF-SCOPE |
| Funnel metrics calculated | 3 key rates |
| Disposition breakdown in report | Table + insights |
| Backwards compatible | Pre-v3.9 analyses work |
| All tests pass | 8/8 passing |

## Migration Notes

No migration needed - the field is additive. Existing analyses will:
1. Continue to work with all existing metrics
2. Be counted as `unknown` in disposition breakdown
3. Not affect funnel metrics (calculated only from analyses with disposition)

Re-run analysis on existing transcripts to populate the new field:

```bash
# Re-analyze a sample to get disposition data
python3 tools/run_analysis.py -n 50 --force-reanalyze
```
