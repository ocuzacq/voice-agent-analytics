# Golden Test Set

36 curated transcripts for regression-testing schema changes, prompt tuning, and dashboard development. Expanded from 23 to 36 on 2026-02-13 for full enum coverage.

## Directory Structure

```
tests/golden/
  transcripts/   # 36 raw call transcripts (.json)
  analyses_v7/   # Current baseline analyses (prompt_v7, schema v7.0)
```

Other `analyses_*` directories are historical snapshots from earlier prompt versions. They're not actively used but can serve as ad-hoc comparison targets.

## Enum Coverage

Every enum value present in the 4,471-call corpus is represented in the golden set.

| Dimension | Values Covered |
|-----------|---------------|
| `ended_reason` | 5/5: customer-ended-call, assistant-ended-call, assistant-forwarded-call, hangup-during-warm-transfer, silence-timeout |
| `queue_result` | 3/3: connected, unavailable, caller_abandoned |
| `abandon_stage` | 3/3 in-corpus: mid_task, pre_intent, post_delivery (pre_greeting=0 in corpus) |
| `request_category` | 16/16: all values including rare ones (tax_documents, mortgage_payment, payment_arrangement) |
| `transfer_destination` | 2/2: concierge, specific_department (ivr=0 in corpus) |
| `scope x outcome` | 7 combos: in_scope:fulfilled, in_scope:transferred, in_scope:abandoned, out_of_scope:fulfilled, out_of_scope:transferred, out_of_scope:abandoned, no_request:abandoned |

**Not in corpus (skipped):** pre_greeting abandon (0 calls), unknown-error ended_reason (2 calls, too rare), ivr destination (0 calls).

## Transcript Inventory

**Legend:** NEW = added 2026-02-13, sec = has secondary intent

| # | Call ID | Scope | Outcome | Category | ended_reason | Key Coverage |
|---|---------|-------|---------|----------|--------------|-------------|
| 1 | `0043088a` NEW | out_of_scope | transferred | human_transfer | assistant-forwarded-call | queue_result=connected |
| 2 | `0094a0d7` NEW | out_of_scope | transferred | billing_dispute | hangup-during-warm-transfer | imp=transfer_queue |
| 3 | `00a05f6e` | in_scope | fulfilled | maintenance_payment | customer-ended-call | sec: account_details_update |
| 4 | `00abd708` | in_scope | abandoned | maintenance_payment | customer-ended-call | human=after_service, imp=verification |
| 5 | `00b6db65` | in_scope | transferred | maintenance_payment | customer-ended-call | imp=verification |
| 6 | `00cccc03` NEW | no_request | abandoned | none | silence-timeout | abandon=pre_intent |
| 7 | `0107c02c` | in_scope | abandoned | maintenance_payment | customer-ended-call | human=after_service, imp=transfer_queue |
| 8 | `018e6a70` NEW | out_of_scope | abandoned | billing_dispute | customer-ended-call | dest=specific_department |
| 9 | `042dfaf7` | out_of_scope | abandoned | ownership_exit | customer-ended-call | agent_issue, imp=transfer_queue |
| 10 | `0460e184` NEW | out_of_scope | fulfilled | ownership_exit | assistant-ended-call | rare scope x outcome combo |
| 11 | `06378125` NEW | in_scope | fulfilled | exchange_program | assistant-ended-call | clean success |
| 12 | `064e5d82` NEW | in_scope | fulfilled | mortgage_payment | assistant-ended-call | sec: exchange_program |
| 13 | `0720df23` NEW | in_scope | abandoned | maintenance_payment | assistant-ended-call | sec: account_details_update:transferred |
| 14 | `0a095b02` | out_of_scope | abandoned | reservation | customer-ended-call | imp=other |
| 15 | `0a0aea06` | in_scope | fulfilled | unit_resort_info | assistant-ended-call | sec: exchange_program |
| 16 | `0a195db9` | in_scope | abandoned | maintenance_payment | customer-ended-call | imp=transfer_queue |
| 17 | `0a2feaf7` | in_scope | fulfilled | maintenance_payment | assistant-ended-call | clean success |
| 18 | `0a3660cb` | in_scope | abandoned | maintenance_payment | customer-ended-call | imp=other, agent_issue |
| 19 | `0a434683` | in_scope | fulfilled | portal_account | assistant-ended-call | clean success |
| 20 | `0a5afc8b` | out_of_scope | abandoned | ownership_exit | customer-ended-call | imp=transfer_queue |
| 21 | `0acca17b` | in_scope | abandoned | portal_account | customer-ended-call | no impediment |
| 22 | `0b59f63d` | out_of_scope | transferred | reservation | customer-ended-call | sec: mortgage_payment |
| 23 | `0d558884` NEW | in_scope | fulfilled | contract_points_info | assistant-ended-call | sec: exchange_program |
| 24 | `0dd63c66` NEW | in_scope | fulfilled | maintenance_payment | hangup-during-warm-transfer | sec: other:transferred |
| 25 | `24cdd3bf` NEW | in_scope | fulfilled | other | assistant-ended-call | sec: maintenance_payment |
| 26 | `25d30736` | in_scope | abandoned | maintenance_payment | customer-ended-call | imp=link_delivery |
| 27 | `286652e9` NEW | in_scope | abandoned | tax_documents | customer-ended-call | abandon=post_delivery |
| 28 | `30d955ef` | out_of_scope | abandoned | billing_dispute | assistant-ended-call | dept-name scope test |
| 29 | `4bacd0aa` NEW | out_of_scope | abandoned | payment_arrangement | assistant-ended-call | queue_result=unavailable |
| 30 | `6940a110` | in_scope | abandoned | maintenance_payment | customer-ended-call | imp=verification |
| 31 | `6c7001c9` | no_request | abandoned | none | customer-ended-call | abandon=pre_intent |
| 32 | `743d7b2a` | in_scope | fulfilled | unit_resort_info | assistant-ended-call | clean success |
| 33 | `77d49f12` | out_of_scope | transferred | human_transfer | customer-ended-call | human=initial |
| 34 | `8c10f101` | in_scope | fulfilled | portal_account | assistant-ended-call | clean success |
| 35 | `97b367dd` | in_scope | abandoned | rental_program | assistant-ended-call | imp=other, agent_issue |
| 36 | `d187d1d7` | in_scope | fulfilled | maintenance_payment | assistant-ended-call | clean success |

## Usage

```bash
# Analyze golden set with current schema/prompt
python3 tools/batch_analyze_v7.py \
    --input-dir tests/golden/transcripts/ \
    --output-dir tests/golden/analyses_v7_test/

# Compare against baseline
python3 tools/compare_golden.py tests/golden/analyses_v7 tests/golden/analyses_v7_test

# Run dashboard
python3 tools/dashboard_v7.py tests/golden/analyses_v7/

# Stability test on volatile transcripts
python3 tools/stability_test.py
```

## Prompt Tuning Workflow

1. Edit prompt in `poc_structured_full.py`
2. `python3 tools/stability_test.py` on known-volatile transcripts
3. `python3 tools/batch_analyze_v7.py --input-dir tests/golden/transcripts/ --output-dir tests/golden/analyses_v7_test/`
4. `python3 tools/compare_golden.py tests/golden/analyses_v7 tests/golden/analyses_v7_test`
5. Review diffs, accept or iterate

## History

| Date | Change |
|------|--------|
| 2026-02-13 | Expanded 23 -> 36 transcripts for full enum coverage; re-analyzed all 36 with prompt_v7 |
| 2026-02-11 | prompt_v7: impediment/agent_issue model, analyses_v7_rc created |
| 2026-02-11 | prompt_v5: human_requested/department_requested fields |
| 2026-02-10 | prompt_v4: transfer bright-lines, stability fixes |
| 2026-02-10 | prompt_v3: initial structured output |
| 2026-02-07 | Initial golden set: 23 transcripts selected for scope x outcome coverage |
