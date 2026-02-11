# Voice Agent Analytics - Project Instructions (v5.0)

## Environment

**API Key available in `.env` file** - load with `source .env` or use python-dotenv.

```bash
source .env  # Sets GOOGLE_API_KEY
```

## Versioning Guidelines

### Always Create Version-Specific READMEs

**CRITICAL**: When creating a new version (v3.2, v4.0, etc.), always create a dedicated `docs/releases/README_vX.Y.md` file documenting:

1. **What changed** from the previous version
2. **Why** the changes were made
3. **Migration notes** if applicable
4. **New features/fields** with examples

Example:
```
README.md                      # Current version (always up-to-date)
docs/releases/README_v5.0.md   # v5.0 release notes and changes
docs/releases/README_v4.5.md   # v4.5 release notes
docs/releases/README_v4.3.md   # v4.3 release notes
```

### Version Numbering

- **Major (vX.0)**: Breaking schema changes, new architecture
- **Minor (vX.Y)**: New features, non-breaking additions
- **Patch (vX.Y.Z)**: Bug fixes, documentation updates

### Archived Versions

Keep archived tool versions in `tools/vX/` directories with their own `VERSION.md` explaining:
- When it was active
- Why it was superseded
- Key differences from current

## Pipeline Architecture (v4.3)

### Two Q&A Tools

| Tool | Reads | Prerequisite | Best for |
|------|-------|-------------|----------|
| `ask_raw.py` | raw transcripts (`sampled/`) | sample only | Fast exploration, no per-call LLM cost |
| `ask.py` | LLM analyses (`analyses/`) | sample + analyze | Structured queries on extracted fields |

### Pipeline Modes

```
# Sample only (--sample-only) → ask_raw.py
transcripts/ → sample → ask_raw.py

# Sample + analyze (default) → ask.py
transcripts/ → sample → analyze (parallel) → ask.py

# Full pipeline (--insights) → report
transcripts/ → sample → analyze → metrics → extract_nl → insights → report
```

### Typical Workflows

```bash
# --- Fast exploration (ask_raw.py, no analysis cost) ---
python3 tools/run_analysis.py -n 30 --sample-only
python3 tools/ask_raw.py "What are customers calling about?"
python3 tools/ask_raw.py "What friction patterns do you see?"

# --- Standard workflow (ask.py, with per-call analysis) ---
python3 tools/run_analysis.py -n 50
python3 tools/ask.py "Why do calls fail?"
python3 tools/ask.py "Main friction patterns?"

# --- Grow iteratively, then report ---
python3 tools/run_analysis.py -n 200        # adds 150 to reach 200
python3 tools/ask.py "How do loops correlate with failures?"
python3 tools/run_analysis.py --insights    # full report
```

### Target-Based Run Augmentation (v4.3)

```bash
# Day 1: Create run with 50 transcripts
python3 tools/run_analysis.py -n 50

# Day 2: Grow to 200 total (system calculates delta: +150)
python3 tools/run_analysis.py -n 200

# Day 3: Grow to 1000 total (system calculates delta: +800)
python3 tools/run_analysis.py -n 1000

# Target already met? Nothing added
python3 tools/run_analysis.py -n 100  # "Run already has 1000 transcripts. Nothing to add."

# Use specific run directory
python3 tools/run_analysis.py -n 500 --run-dir runs/run_xxx
```

**Key behavior:**
- New runs: `-n 50` = sample 50 transcripts
- Existing runs: `-n 200` = TARGET 200 total, system adds difference
- Manifest tracks all sampled files, excluding already-sampled on augment

### Pipeline Steps

1. `sample_transcripts.py` - Random stratified sampling
2. `batch_analyze.py` - LLM analysis with parallel processing (default 3 workers)
3. `compute_metrics.py` - Section A: Deterministic metrics (requires `--insights`)
4. `extract_nl_fields.py` - Condensed NL data for LLM (requires `--insights`)
5. `generate_insights.py` - Section B: LLM insights (requires `--insights`)
6. `render_report.py` - Markdown executive summary (requires `--insights`)
7. `review_report.py` - Editorial review (requires `--insights --enable-review`)

### SQL Analytics (v5.0)

```bash
# Load analyses into DuckDB
python3 tools/load_duckdb.py runs/run_XXXX/analyses/

# Run dashboard queries (scope x outcome, containment rate, etc.)
python3 tools/query.py runs/run_XXXX/ --dashboard

# Ad-hoc SQL
python3 tools/query.py runs/run_XXXX/ -q "SELECT call_scope, call_outcome, COUNT(*) FROM calls GROUP BY 1, 2"
```

### V6 Dashboard (v6.0 schema)

19-section narrative dashboard reading v6.0 analysis JSONs via DuckDB `read_json_auto()`:

```bash
# Run on any directory of v6.0 analysis JSONs
python3 tools/dashboard_v6.py tests/golden/analyses_v6_prompt_v5/

# Older data without human_requested — Act 2 auto-skipped
python3 tools/dashboard_v6.py tests/golden/analyses_v6_review/batch_50/
```

**4-act structure:**
1. **The Big Picture** — KPIs, funnel, scope x outcome, top requests
2. **Human-Request Phenomenon** — human_requested rates, organic containment, departments (v5+ only)
3. **Quality & Failure** — failure modes, preventable escalations, scores, sentiment
4. **Operational Details** — duration, actions, transfers, friction, abandons, secondary intents

### Ad-hoc Q&A Tools

**`ask.py`** — queries LLM analysis outputs (structured fields)
```
analyses/ → ask.py → asks/<timestamp>/
```
- Reads from `analyses/` (requires `batch_analyze.py` to have run)
- Random sampling from analyses (default 100, configurable)
- Cites 2-4 illustrative examples, auto-saves to `asks/`

**`ask_raw.py`** — queries raw transcripts directly (no analysis needed)
```
sampled/ → ask_raw.py → asks_raw/<timestamp>/
```
- Reads from `sampled/` (only needs `sample_transcripts.py`)
- Preprocesses on-the-fly (coalesces fragmented ASR messages)
- Lower default limit (30) due to larger per-call context
- Auto-saves to `asks_raw/`

### Parallelization (v3.2)

```bash
# Default: 3 parallel workers
python3 tools/run_analysis.py -n 200

# More aggressive
python3 tools/run_analysis.py -n 300 --workers 5

# Sequential (v3.1 behavior)
python3 tools/run_analysis.py -n 50 --workers 1
```

## Testing

Always run the test harness before releases:
```bash
python3 tools/test_framework.py

# v4.0: Run intent + sentiment analysis tests
python3 tools/test_v40_features.py

# v3.9: Run call disposition classification tests
python3 tools/test_v39_features.py

# v3.8.5: Run streamlined friction tracking tests
python3 tools/test_v385_features.py

# v3.7: Run preprocessing + structured event context tests
python3 tools/test_v37_features.py

# v3.6: Run conversation quality feature tests
python3 tools/test_v36_features.py
```

### v6.0 Schema Testing (Structured Output)

Per-intent resolution schema with Gemini structured output (`response_schema`).

**Current prompt version**: v5 — scope fix + `human_requested` / `department_requested` fields.

```bash
# Single transcript analysis
python3 tools/poc_structured_full.py tests/golden/transcripts/<uuid>.json

# Batch all 23 golden transcripts
python3 tools/batch_golden_v6.py --output-dir tests/golden/analyses_v6_prompt_v5

# Compare two prompt versions (finds regressions)
python3 tools/compare_golden.py tests/golden/analyses_v6_prompt_v4 tests/golden/analyses_v6_prompt_v5

# Stability test: run volatile transcripts N times to verify consistency
python3 tools/stability_test.py              # default: 3 volatile transcripts × 5 reps
python3 tools/stability_test.py -n 5 --transcripts path1.json path2.json
```

**Prompt tuning workflow**: edit prompt in `poc_structured_full.py` → run `stability_test.py` on affected transcripts → run `batch_golden_v6.py` → run `compare_golden.py` to check for regressions.

## LLM Provider

**CRITICAL: Always use Gemini 3 models. Never use older models (gemini-2.5-flash, etc.)**

| Use Case | Model | Thinking Level |
|----------|-------|----------------|
| Per-transcript analysis | `gemini-3-flash-preview` | LOW (fast, ~3-6s/call) |
| Aggregate insights | `gemini-3-pro-preview` | default (none set) |
| Report review | `gemini-3-pro-preview` | default (none set) |
| Report rendering | `gemini-3-pro-preview` | default (none set) |
| Ad-hoc Q&A (ask.py) | `gemini-3-pro-preview` | default (none set) |

**Note:** When thinking level is "default (none set)", no `thinking_config` is specified in the GenerateContentConfig, allowing the model to use its native default behavior.

### Thinking Configuration

**Current Strategy:**

1. **Per-transcript analysis** (analyze_transcript.py): Use LOW thinking level for speed
   ```python
   thinking_config=genai.types.ThinkingConfig(thinking_level="LOW")
   ```

2. **Insights, reports, and Q&A** (generate_insights.py, render_report.py, review_report.py, ask.py):
   - Use **default** (no thinking_config specified)
   - Allows model to use its native reasoning without constraints

**Available Thinking Levels (Gemini 3 Flash only):**
- `minimal`: No thinking, fastest latency
- `low`: Minimal reasoning, fast (used for per-call analysis)
- `medium`: Balanced reasoning
- `high`: Deep reasoning, slower

**Note:** Gemini 3 Pro only supports LOW/HIGH when explicitly set. Thinking tokens count against `max_output_tokens`, so use higher limits (16384+) when using thinking configs.

### API Key

Set `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variable.
