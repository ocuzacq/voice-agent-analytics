# v3.9.1 Release Notes: Loop Subject Granularity + Custom Questions

**Release Date:** 2026-01-19

## Summary

v3.9.1 adds two major features:

1. **Loop Subject Granularity**: A `subject` field in friction loops identifying WHAT is being retried/deflected
2. **Custom Questions**: User-provided analytical questions answered by the LLM in the report

Additionally, the **review step is now disabled by default** (use `--enable-review` to enable).

## Key Changes

### Schema Change

**Before (v3.9):**
```json
"loops": [
  {
    "t": [20, 22, 25],
    "type": "info_retry",
    "ctx": "name asked 3x despite refusal"
  }
]
```

**After (v3.9.1):**
```json
"loops": [
  {
    "t": [20, 22, 25],
    "type": "info_retry",
    "subject": "name",
    "ctx": "asked 3x despite refusal"
  }
]
```

### Subject Values by Loop Type

The subject field uses guided values per loop type with freeform fallback for edge cases:

| Loop Type | Guided Values |
|-----------|---------------|
| `info_retry` | `name`, `phone`, `address`, `zip`, `state`, `account`, `email` |
| `intent_retry` | `fee_info`, `balance`, `payment_link`, `autopay_link`, `history_link`, `rental_link`, `clubhouse_link`, `rci_link`, `transfer`, `callback` |
| `deflection` | `anything_else`, `other_help`, `clarify_request` |
| `comprehension` | `unclear_speech`, `background_noise`, `connection` |
| `action_retry` | `verification`, `link_send`, `lookup`, `transfer_attempt` |

For edge cases not in these lists, use descriptive `lowercase_with_underscores`.

## New Features

### 1. Subject Field in Loops (analyze_transcript.py)
- LLM now extracts what is being looped on
- Guided values ensure consistency while allowing flexibility
- Context (ctx) field now focuses on the HOW, not the WHAT

### 2. Loop Subject Aggregation (compute_metrics.py)
- `loops_with_subject`: Count of loops that have subject data
- `by_subject`: Subject distribution per loop type
- `top_subjects`: Overall most frequent loop subjects

### 3. Loop Subject Pairs (extract_nl_fields.py)
- `loop_subject_pairs`: (loop_type, subject) pairs for LLM clustering
- Enables semantic grouping of similar subjects

### 4. Loop Subject Insights (generate_insights.py)
- `loop_subject_clusters`: LLM-generated clustering and analysis
- `by_loop_type`: Top subjects per loop type with counts and insights
- `high_impact_patterns`: (loop_type, subject) combinations affecting outcomes

### 5. Report Rendering (render_report.py)
- Loop Subject Analysis section in Conversation Quality
- Subject Breakdown by Loop Type table
- High-Impact Patterns table

## Report Output

```markdown
### Loop Subject Analysis (v3.9.1)

45% of friction loops involve verification data re-requests, with name spelling
being the top subject. This suggests ASR/name recognition as a high-impact
improvement area.

#### Subject Breakdown by Loop Type

**Info Retry:**

| Subject | Count | % |
|---------|-------|---|
| name | 45 | 45% |
| phone | 30 | 30% |
| account | 15 | 15% |
| other | 10 | 10% |

#### High-Impact Patterns

| Loop Type | Subject | Impact | Recommendation |
|-----------|---------|--------|----------------|
| Info Retry | name | 45% failure correlation | Improve ASR for names |
| Intent Retry | fee_info | Common after verification | Context retention |
```

## Backwards Compatibility

- Old analyses without `subject` field: Skipped in subject aggregation
- All existing metrics continue to work
- No breaking changes to existing fields
- `loops_with_subject` count shows v3.9.1 adoption
- `--skip-review` flag still works but is now the default behavior

## Custom Questions Feature

### Usage

Create a questions file (one question per line):

```text
# questions.txt
What are the main cases where the agent struggles understanding the user?
Which types of requests lead to the longest call durations?
What capabilities are customers most frequently asking for that the agent cannot handle?
```

Run the pipeline with custom questions:

```bash
python3 tools/run_analysis.py -n 50 --questions questions.txt
```

### Report Output

The Custom Analysis section appears after the Executive Summary:

```markdown
## Custom Analysis

*Answers to user-provided analytical questions:*

### 1. What are the main cases where the agent struggles understanding the user?

**Confidence:** HIGH

The agent struggles most with name comprehension (45% of clarification events involve
name spelling), especially with uncommon names and strong accents. Phone number
confirmation is the second most common issue (30% of clarifications).

**Supporting Evidence:**
- 45% of info_retry loops involve name re-asking
- 30% of clarifications are phone number confirmations
- Agent misheard cause accounts for 25% of all clarifications
```

### Tips for Good Questions

- Ask specific, data-driven questions
- Questions about patterns, correlations, and root causes work well
- The LLM has access to metrics, failure details, customer verbatims, and conversation quality data
- See `questions_example.txt` for more examples

## Review Step Changes

The report review step is now **disabled by default**:

```bash
# Run without review (default in v3.9.1)
python3 tools/run_analysis.py -n 50

# Enable review explicitly
python3 tools/run_analysis.py -n 50 --enable-review
```

## Testing

```bash
# Run v3.9.1 feature tests
python3 tools/test_v391_features.py

# Test schema changes
python3 tools/analyze_transcript.py transcripts/sample.txt --stdout | jq '.friction.loops[].subject'

# Run full pipeline
python3 tools/run_analysis.py --quick

# Check subject breakdown in report
grep -A10 "Loop Subject" reports/executive_summary_v3_*.md
```

## Migration

No migration required. The subject field is additive:
- New analyses will have subject populated
- Old analyses work unchanged (subject = null)
- Metrics gracefully handle missing subjects

## Files Modified

| File | Changes |
|------|---------|
| `analyze_transcript.py` | Added subject field to schema + LLM prompt guidance |
| `compute_metrics.py` | Added subject aggregation to loop_stats |
| `extract_nl_fields.py` | Added loop_subject_pairs extraction |
| `generate_insights.py` | Added loop_subject_clusters + custom questions support |
| `render_report.py` | Added Loop Subject Analysis + Custom Analysis sections |
| `run_analysis.py` | Added --questions flag, review disabled by default |
| `test_v391_features.py` | **NEW**: Unit tests for subject extraction |
| `questions_example.txt` | **NEW**: Example questions file template |
| `README_v3.9.1.md` | **NEW**: This release notes file |

## Future Enhancements

Potential v3.9.2+ improvements:
- Subject normalization rules (e.g., "customer_name" → "name")
- Cross-loop type subject correlation
- Subject-based training recommendations
