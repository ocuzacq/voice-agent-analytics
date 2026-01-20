# Vacatia AI Voice Agent Analytics Framework

Analytical framework to evaluate the Vacatia AI voice agent's performance using ~5,800 call transcripts.

## Philosophy

**Two-part report architecture** separating deterministic metrics from LLM-powered insights:
- **Section A**: Python-calculated metrics (reproducible, auditable)
- **Section B**: LLM-generated insights (executive narratives, recommendations)

**v3.9.1 Enhancements**:
- Loop Subject Granularity: New `subject` field in friction loops identifies WHAT is being looped on
- Guided values by loop type (info_retry: name, phone; intent_retry: fee_info, balance; etc.)
- Subject aggregation: breakdown per loop type for targeted improvements
- LLM semantic clustering of loop subjects for pattern discovery
- Report includes Loop Subject Analysis section with high-impact patterns

**v3.9 Enhancements**:
- Call Disposition Classification: Single `call_disposition` field for funnel analysis
- 6 disposition values: pre_intent, out_of_scope_handled, out_of_scope_abandoned, in_scope_success, in_scope_partial, in_scope_failed
- Decision tree for classification based on customer intent, agent scope, and completion status
- Funnel metrics: In-scope success rate, out-of-scope recovery rate, pre-intent rate
- Actionable insights by disposition (e.g., in_scope_partial → add confirmation prompts)

**v3.8.5 Enhancements**:
- Streamlined friction tracking: Single compact `friction` object
- Shorter enum values (name vs name_spelling, misheard vs agent_misheard)
- Compact keys (t, ctx, sev) and telegraph-style context (5-8 words)
- Loops now include turn numbers in `t` array
- ~31% output size reduction (2,906 → ~2,000 bytes/call)
- Full backwards compatibility with v3.8 format

**v3.8 Enhancements**:
- Agent Loops: Replaces `repeated_prompts` with typed `agent_loops` schema
- Loop Types: info_retry, intent_retry, deflection, comprehension, action_retry
- Only friction loops tracked (benign repetition excluded)
- Loop density metric: loops / total turns (normalized for call length)

**v3.7 Enhancements**:
- Transcript Preprocessing: Deterministic turn counting before LLM analysis
- Structured Event Context: `cause` enum + `context` for clarifications
- Structured Event Context: `severity` enum + `context` for corrections
- Cause/severity aggregation and analysis in reports

**v3.6 Enhancements**:
- Conversation Quality Tracking: Turn counts, clarification requests, user corrections, loop detection
- Friction Hotspots: Identify which clarification types cause most friction
- Turn Analysis: Avg turns by outcome, turns-to-failure for failed calls

**v3.5.5 Enhancements**:
- Report Review: Automated editorial pass with Gemini 3 Pro
- Pipeline Suggestions: Generated improvement ideas after each run
- Preserved originals: Both original and refined reports kept

**v3.5 Enhancements**:
- Training & Development: Narrative-first section with priorities, root causes, and recommended actions
- Cross-Dimensional Patterns: Training gaps correlated with failure types
- Emergent Patterns: LLM-discovered patterns not fitting standard categories
- Secondary Customer Needs: Clustered additional intents

**v3.4 Enhancements**:
- Inline descriptions in 4th column (Context) for all major tables
- Key metrics context explaining WHY each value (drivers/causes, not thresholds)
- Sub-breakdowns for major failure types (≥5% of failures)
- Fixed customer ask clustering: LLM now sees ALL asks for accurate semantic grouping

**v3.3 Enhancements**:
- Semantic clustering of customer asks (eliminates duplicates)
- Explanatory qualifiers for failure types and policy gap categories
- Call ID references throughout for traceability
- Validation warnings for data quality issues

**v3.2 Enhancements**:
- Configurable parallel processing (default 3 workers) for faster batch analysis
- Run isolation: `sampled/` cleared by default between runs
- Scope coherence: `batch_analyze.py` respects `manifest.csv`

## Overview

- **19-field analysis schema** (v3.8.5) - hybrid metrics + insights + streamlined friction tracking
- **3 quality scores** - agent effectiveness, conversation quality, customer effort
- **Policy gap breakdown** - structured categorization of capability limitations
- **Customer verbatim** - direct quotes capturing frustration/needs
- **Agent coaching insights** - specific missed opportunities
- **Resolution steps** - call flow timeline
- **Conversation quality tracking** - turns, clarifications, corrections, loops (v3.6)
- **Executive-ready Markdown reports**

## Quick Start

```bash
# 1. Set your Google AI API key
export GOOGLE_API_KEY="your-api-key-here"

# 2. Run the full pipeline (50 transcripts, 3 parallel workers)
python3 tools/run_analysis.py

# OR quick test with 5 transcripts
python3 tools/run_analysis.py --quick

# OR larger batch with more parallelization
python3 tools/run_analysis.py -n 200 --workers 5

# OR with custom questions (v3.9.1)
python3 tools/run_analysis.py -n 50 --questions questions.txt

# OR step-by-step:
python3 tools/sample_transcripts.py -n 50
python3 tools/batch_analyze.py --workers 3   # v3.2: Parallel processing
python3 tools/compute_metrics.py
python3 tools/extract_nl_fields.py           # v3.1: Condensed NL data for LLM
python3 tools/generate_insights.py --questions questions.txt  # v3.9.1: Custom questions
python3 tools/render_report.py
```

## Tools

### `run_analysis.py` (Orchestrator)

End-to-end pipeline that runs all steps in sequence.

```bash
# Full pipeline with 50 transcripts (3 parallel workers)
python3 tools/run_analysis.py

# Quick test with 5 transcripts
python3 tools/run_analysis.py --quick

# Larger batch with more parallelization
python3 tools/run_analysis.py -n 200 --workers 5

# Custom sample size with reproducible seed
python3 tools/run_analysis.py -n 100 --seed 42

# Resume an interrupted run (uses existing manifest)
python3 tools/run_analysis.py --resume

# v3.9.1: Custom questions (answered in report)
python3 tools/run_analysis.py -n 50 --questions questions.txt

# v3.9.1: Enable report review (disabled by default)
python3 tools/run_analysis.py -n 50 --enable-review

# Append to existing samples (don't clear sampled/)
python3 tools/run_analysis.py -n 50 --no-clear

# Skip sampling/analysis (use existing data)
python3 tools/run_analysis.py --skip-sampling --skip-analysis

# Metrics only (no LLM insights)
python3 tools/run_analysis.py --skip-insights
```

### `sample_transcripts.py`

Randomly selects N transcripts stratified by file size (proxy for call complexity).

```bash
python3 tools/sample_transcripts.py -n 50
python3 tools/sample_transcripts.py -n 100 --seed 42  # Reproducible
python3 tools/sample_transcripts.py -n 50 --no-clear  # Append to existing samples
```

**v3.2**: Clears `sampled/` by default before copying new files (run isolation). Use `--no-clear` to append.

### `analyze_transcript.py`

Analyzes a single transcript using LLM (Gemini) and produces v3 JSON schema.

```bash
python3 tools/analyze_transcript.py sampled/some-uuid.txt
python3 tools/analyze_transcript.py sampled/some-uuid.txt --stdout
```

### `batch_analyze.py`

Batch analyzes multiple transcripts with configurable parallelization (v3.2).

```bash
python3 tools/batch_analyze.py                    # Default: 3 parallel workers
python3 tools/batch_analyze.py --workers 5        # More parallelization
python3 tools/batch_analyze.py --workers 1        # Sequential (v3.1 behavior)
python3 tools/batch_analyze.py --limit 10         # First 10 only
python3 tools/batch_analyze.py --no-skip-existing # Re-analyze all
```

**v3.2**: If `manifest.csv` exists in input directory, only processes files listed in it (scope coherence).

### `compute_metrics.py`

Computes Section A: Deterministic Metrics from analysis JSON files.

```bash
python3 tools/compute_metrics.py
python3 tools/compute_metrics.py --json-only
```

### `extract_nl_fields.py` (v3.1)

Extracts and condenses natural language fields from v3 analyses for optimized LLM context usage.

```bash
python3 tools/extract_nl_fields.py
python3 tools/extract_nl_fields.py --limit 50  # Test with subset
```

**Benefits over embedded extraction:**
- ~70% smaller than full analysis JSONs
- Explicit pipeline step (better architecture)
- LLM-ready grouping by failure type

### `generate_insights.py`

Generates Section B: LLM-powered insights from metrics + NL summary.

```bash
python3 tools/generate_insights.py
python3 tools/generate_insights.py --model gemini-3-pro-preview  # Default
```

### `render_report.py`

Renders the full report as executive-ready Markdown.

```bash
python3 tools/render_report.py
python3 tools/render_report.py --stdout  # Print to console
```

### `review_report.py` (v3.5.5)

Editorial review and refinement of rendered reports.

```bash
python3 tools/review_report.py                    # Review latest report
python3 tools/review_report.py -i report.md       # Review specific report
python3 tools/review_report.py --no-suggestions   # Skip pipeline suggestions
python3 tools/review_report.py --model gemini-3-pro-preview  # Default
```

**Benefits:**
- Finds inconsistencies and logical gaps
- Tightens prose and removes redundancies
- Generates pipeline improvement suggestions
- Preserves original report alongside refined version

### `ask.py`

Ad-hoc Q&A about call data without generating full reports.

```bash
python3 tools/ask.py "Why do calls fail?"
python3 tools/ask.py "What causes name issues?" --limit 50
python3 tools/ask.py "Main friction patterns?" --limit 250 --stats
python3 tools/ask.py "Top customer frustrations?" --verbose
```

**How it works:**
- Single question → single text answer
- Random sampling from analyses/ directory (default 100 calls, configurable via `--limit`)
- Cites 2-4 illustrative examples (not exhaustive)
- Auto-saves to `asks/<timestamp>/` with question.txt, answer.md, metadata.json
- Token usage stats available with `--stats` or `--verbose` flags

**Use cases:**
- Quick hypothesis testing without full pipeline runs
- Exploring specific questions between report runs
- Rapid iteration on specific failure patterns or friction points

## Output Files

```
reports/
├── metrics_v3_{timestamp}.json                    # Section A: Deterministic metrics
├── nl_summary_v3_{timestamp}.json                 # v3.1: Condensed NL fields for LLM
├── report_v3_{timestamp}.json                     # Combined Section A + B
├── executive_summary_v3_{timestamp}.md            # Markdown executive report
├── executive_summary_v3_{timestamp}_reviewed.md   # v3.5.5: Refined report
└── pipeline_suggestions_v3_{timestamp}.md         # v3.5.5: Improvement ideas

asks/
└── Jan20-10h40/                                   # Timestamped Q&A sessions
    ├── question.txt                               # Original question
    ├── answer.md                                  # LLM response
    └── metadata.json                              # Sample info + token usage
```

## Directory Structure

```
.
├── transcripts/           # 5822 raw transcript files
├── tools/
│   ├── run_analysis.py    # End-to-end orchestrator
│   ├── sample_transcripts.py
│   ├── analyze_transcript.py
│   ├── batch_analyze.py
│   ├── compute_metrics.py
│   ├── extract_nl_fields.py  # v3.1: NL extraction
│   ├── generate_insights.py
│   ├── render_report.py
│   ├── review_report.py   # v3.5.5: Editorial review
│   ├── ask.py             # Ad-hoc Q&A without full reports
│   ├── v0/                # Archived: Simple schema (~15 fields)
│   ├── v1/                # Archived: Verbose schema (~50 fields)
│   ├── v2/                # Previous: Actionable schema (14 fields)
│   └── v3/                # Current: Hybrid schema (18 fields)
├── sampled/               # Output from sampling script
├── analyses/              # JSON output from analyzer
├── reports/               # Aggregate metrics and reports
└── asks/                  # Q&A session results
```

## Version History

| Version | Fields | Focus | Status | Release Notes |
|---------|--------|-------|--------|---------------|
| **v0** | ~15 | Funnel, coverage, outcome | Archived | `tools/v0/VERSION.md` |
| **v1** | ~50 | +Performance, agent quality, customer profile | Archived | `tools/v1/VERSION.md` |
| **v2** | 14 | Actionable insights, failure analysis, training | Previous | `tools/v2/VERSION.md` |
| **v3** | 18 | Hybrid metrics + insights, policy gaps, verbatims | Previous | `tools/v3/VERSION.md` |
| **v3.1** | 18 | Dedicated NL extraction for optimized LLM context | Previous | [`README_v3.1.md`](README_v3.1.md) |
| **v3.2** | 18 | Configurable parallel processing (default 3 workers) | Previous | [`README_v3.2.md`](README_v3.2.md) |
| **v3.3** | 18 | Report quality: clustering, explanations, call IDs | Previous | [`README_v3.3.md`](README_v3.3.md) |
| **v3.4** | 18 | Inline descriptions, key metrics context, sub-breakdowns | Previous | [`README_v3.4.md`](README_v3.4.md) |
| **v3.5** | 18 | Training insights, emergent patterns, secondary intents | Previous | [`README_v3.5.md`](README_v3.5.md) |
| **v3.5.5** | 18 | Report review, pipeline suggestions | Previous | [`README_v3.5.5.md`](README_v3.5.5.md) |
| **v3.6** | 23 | Conversation quality: turns, clarifications, corrections, loops | Previous | [`README_v3.6.md`](README_v3.6.md) |
| **v3.7** | 23 | Preprocessing + structured event context (cause/severity) | Previous | [`README_v3.7.md`](README_v3.7.md) |
| **v3.8** | 23 | Agent loops: typed detection replacing repeated_prompts | Previous | [`README_v3.8.md`](README_v3.8.md) |
| **v3.8.5** | 19 | Streamlined friction: compact object, shorter enums, ~31% size reduction | Previous | [`README_v3.8.5.md`](README_v3.8.5.md) |
| **v3.9** | 20 | Call disposition classification for funnel analysis | Previous | [`README_v3.9.md`](README_v3.9.md) |
| **v3.9.1** | 20 | Loop subject granularity: subject field for targeted friction analysis | **Current** | [`README_v3.9.1.md`](README_v3.9.1.md) |

### Versioning Guidelines

**Always create a `README_vX.Y.md`** when releasing a new version, documenting:
- What changed from the previous version
- Why the changes were made
- Migration notes if applicable

See [`CLAUDE.md`](CLAUDE.md) for full versioning guidelines and project instructions.

## Analysis Schema (v3.8.5)

Each transcript analysis produces a JSON with 19 fields (consolidated from 23 in v3.8):

```json
{
  "call_id": "uuid",
  "schema_version": "v3.8.5",

  // === OUTCOME ===
  "outcome": "resolved",
  "resolution_type": "payment processed",

  // === QUALITY SCORES (1-5) ===
  "agent_effectiveness": 4,
  "conversation_quality": 4,
  "customer_effort": 2,

  // === FAILURE ANALYSIS ===
  "failure_point": "none",
  "failure_description": null,
  "was_recoverable": null,
  "critical_failure": false,

  // === ACTIONABLE FLAGS ===
  "escalation_requested": false,
  "repeat_caller_signals": false,
  "training_opportunity": null,
  "additional_intents": null,

  "summary": "Customer called to make payment. Resolved after brief verification.",

  // === v3 FIELDS ===
  "policy_gap_detail": null,
  "customer_verbatim": null,
  "agent_miss_detail": null,
  "resolution_steps": ["greeted customer", "verified identity", "processed payment", "confirmed success"],

  // === v3.8.5 FRICTION (consolidated from v3.6-v3.8) ===
  "friction": {
    "turns": 12,
    "derailed_at": null,
    "clarifications": [{"t": 3, "type": "phone", "cause": "ok", "ctx": "confirmed number"}],
    "corrections": [],
    "loops": []
  }
}
```

### New v3 Fields

| Field | Type | Description |
|-------|------|-------------|
| `policy_gap_detail` | object | Structured breakdown when failure_point="policy_gap" |
| `customer_verbatim` | string | Key quote capturing customer frustration/need |
| `agent_miss_detail` | string | What agent should have done differently |
| `resolution_steps` | array | Sequence of actions taken during the call |

### v3.8.5 Friction Object (consolidated from v3.6-v3.8)

| Field | Type | Description |
|-------|------|-------------|
| `friction.turns` | integer | Total conversation turns (proxy for call duration) |
| `friction.derailed_at` | integer/null | Turn where call started failing (non-resolved only) |
| `friction.clarifications` | array | Agent asks customer to clarify [{t, type, cause, ctx}] |
| `friction.corrections` | array | Customer corrects agent [{t, sev, ctx}] |
| `friction.loops` | array | Agent friction loops [{t, type, subject, ctx}] with turn array |

#### Clarification Types (v3.8.5 short enums)

| Type | Description | Example |
|------|-------------|---------|
| `name` | Agent asks to spell name | "Can you spell your name?" |
| `phone` | Agent confirms phone number | "So that's 315-276-0534?" |
| `intent` | Agent asks what customer needs | "What can I help with?" |
| `repeat` | Agent asks to repeat | "Can you say that again?" |
| `verify` | Agent asks for different verification | "Try another phone number" |

#### Clarification Cause Types (v3.8.5 short enums)

| Cause | Description |
|-------|-------------|
| `refused` | Customer declined to provide info |
| `unclear` | Customer provided but unclear |
| `misheard` | Agent failed to understand |
| `tech` | Audio/connection problem |
| `ok` | Clarification worked |

#### Correction Severity Types (v3.7)

| Severity | Description |
|----------|-------------|
| `minor` | Simple correction, no frustration |
| `moderate` | Correction with mild frustration |
| `major` | Explicit anger, multiple corrections |

#### Agent Loop Types (v3.8)

| Type | Description | Example |
|------|-------------|---------|
| `info_retry` | Re-asked for info already provided | "Spell your name" twice |
| `intent_retry` | Re-asked for intent already stated | "How can I help?" after customer stated need |
| `deflection` | Generic questions while unable to help | "Anything else?" while stuck |
| `comprehension` | Couldn't hear, asked to repeat | "Sorry, one more time?" |
| `action_retry` | System/process retries | "Let me try that again" |

**Note:** Only friction loops are tracked. Benign repetition (greeting after hold, compliance disclosures) is excluded.

#### Loop Subject Values (v3.9.1)

The `subject` field identifies WHAT is being looped on. Use guided values per loop type:

| Loop Type | Guided Subjects |
|-----------|-----------------|
| `info_retry` | name, phone, address, zip, state, account, email |
| `intent_retry` | fee_info, balance, payment_link, autopay_link, history_link, rental_link, clubhouse_link, rci_link, transfer, callback |
| `deflection` | anything_else, other_help, clarify_request |
| `comprehension` | unclear_speech, background_noise, connection |
| `action_retry` | verification, link_send, lookup, transfer_attempt |

For edge cases not in these lists, use descriptive `lowercase_with_underscores`.

### Policy Gap Detail Structure

```json
{
  "category": "capability_limit | data_access | auth_restriction | business_rule | integration_missing",
  "specific_gap": "what exactly couldn't be done",
  "customer_ask": "what the customer wanted",
  "blocker": "why it couldn't be fulfilled"
}
```

## Two-Part Report Structure

### Section A: Deterministic Metrics

All metrics computed via Python - reproducible and auditable:

- Outcome distribution with rates
- Key rates: success, containment, escalation, failure
- Quality score statistics (mean, median, std)
- Failure analysis by failure_point
- Policy gap breakdown by category
- Top specific gaps and customer asks
- Actionable flags
- Training priorities

### Section B: LLM-Generated Insights

Strategic analysis generated by LLM:

- **Executive summary** - 2-3 sentence takeaway
- **Root cause analysis** - Primary driver + contributing factors
- **Recommendations** - P0/P1/P2 prioritized actions with expected impact
- **Trend narratives** - Failure patterns, customer experience, agent performance
- **Verbatim highlights** - Most frustrated, common ask, biggest miss

## Metrics Reference

### Key Rates

| Metric | Description |
|--------|-------------|
| **Success Rate** | resolved / total |
| **Containment Rate** | (resolved + abandoned) / total |
| **Escalation Rate** | escalated / total |
| **Failure Rate** | non-resolved / total |

### Quality Scores (1-5 scale)

| Score | What It Measures | 1 = | 5 = |
|-------|-----------------|-----|-----|
| `agent_effectiveness` | Understanding + response quality | Poor | Perfect |
| `conversation_quality` | Flow, tone, clarity | Stilted | Natural |
| `customer_effort` | Customer work required | Effortless | Painful |

### Failure Point Taxonomy

| Type | Description |
|------|-------------|
| `none` | Call succeeded |
| `nlu_miss` | Agent misunderstood customer |
| `wrong_action` | Agent understood but acted incorrectly |
| `policy_gap` | Business rules prevented resolution |
| `customer_confusion` | Customer unclear or wrong info |
| `tech_issue` | System errors, call quality |
| `other` | Anything else |

### Policy Gap Categories

| Category | Description |
|----------|-------------|
| `capability_limit` | Feature not built yet |
| `data_access` | Agent can't access needed info |
| `auth_restriction` | Verification prevented action |
| `business_rule` | Policy prevents action |
| `integration_missing` | System integration unavailable |

## Validation Rules (v3)

1. **failure_point consistency**: If outcome ∈ {abandoned, escalated, unclear} → failure_point ≠ "none"
2. **policy_gap_detail required**: If failure_point = "policy_gap" → policy_gap_detail must be populated
3. **agent_miss_detail conditional**: If was_recoverable = true → agent_miss_detail should be populated

## Requirements

- Python 3.10+
- `google-generativeai` package

```bash
pip install google-generativeai
```

## Configuration

### API Key

Set your Google AI API key:

```bash
export GOOGLE_API_KEY="your-key"
# or
export GEMINI_API_KEY="your-key"
```

### LLM Models

**Always use Gemini 3 models.** Never use older models (gemini-2.5-flash, etc.)

| Use Case | Model | Thinking Level | Description |
|----------|-------|----------------|-------------|
| Transcript analysis | `gemini-3-flash-preview` | `LOW` | Fast per-call analysis (~3-6s/call) |
| Aggregate insights | `gemini-3-pro-preview` | default (none set) | Deep reasoning for patterns/recommendations |
| Report review | `gemini-3-pro-preview` | default (none set) | Editorial quality and consistency |
| Ad-hoc Q&A (ask.py) | `gemini-3-pro-preview` | default (none set) | Analytical question answering |

## Sample Output

### Executive Summary (Markdown)

```markdown
# Vacatia AI Voice Agent Performance Report
Generated: 2026-01-18 14:30 | Calls Analyzed: 50

## Executive Summary
The AI agent resolves 35% of calls, with policy gaps accounting for
44% of failures. Authentication restrictions and missing capabilities
are the primary blockers. Quick wins: adding address change capability
could resolve 8% of abandoned calls.

## Key Metrics at a Glance
| Metric | Value | Assessment | Context |
|--------|-------|------------|---------|
| Success Rate | 34.9% | ⚠️ | Driven by dead-end escalation; verified customers can't reach humans |
| Escalation Rate | 20.9% | ✅ | 45% explicitly requested human; most blocked by capacity limits |
| Customer Effort | 3.26/5 | ⚠️ | Verification succeeds but leads nowhere (verify-then-dump) |

## Failure Point Breakdown
| Failure Type | Count | % of Failures | Context |
|--------------|-------|---------------|---------|
| policy_gap | 22 | 44.0% | Dead-end escalation logic; no callback capability |
| other | 10 | 20.0% | Caller hangups and unclear outcomes |
...
```

## Existing Analysis

The `cowork/` directory contains previous manual analysis:
- `Vacatia_AI_Agent_Analysis_FINAL.xlsx`
- `Vacatia_Transcript_Analysis_Results.xlsx`
- `Vacatia_AI_Agent_Codebook_v1.docx`
