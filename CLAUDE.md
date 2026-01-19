# Voice Agent Analytics - Project Instructions (v3.8.5)

## Versioning Guidelines

### Always Create Version-Specific READMEs

**CRITICAL**: When creating a new version (v3.2, v4.0, etc.), always create a dedicated `README_vX.Y.md` file documenting:

1. **What changed** from the previous version
2. **Why** the changes were made
3. **Migration notes** if applicable
4. **New features/fields** with examples

Example:
```
README.md          # Current version (always up-to-date)
README_v3.1.md     # v3.1 release notes and changes
README_v3.0.md     # v3.0 release notes
README_v2.0.md     # v2.0 release notes (if retroactively created)
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

## Pipeline Architecture (v3.8.5)

```
transcripts/ → sample → preprocess → analyze (parallel) → metrics → extract_nl → insights → report → review
```

1. `sample_transcripts.py` - Random stratified sampling
2. `preprocess_transcript.py` - Deterministic turn counting (v3.7: integrated into analyze)
3. `batch_analyze.py` - LLM analysis with parallel processing (v3.2: default 3 workers)
4. `compute_metrics.py` - Section A: Deterministic metrics (v3.7: +cause/severity aggregation)
5. `extract_nl_fields.py` - Condensed NL data for LLM (v3.7: +cause/severity/context)
6. `generate_insights.py` - Section B: LLM insights (v3.7: +cause/severity analysis)
7. `render_report.py` - Markdown executive summary (v3.7: +cause/severity breakdowns)
8. `review_report.py` - Editorial review and pipeline suggestions (v3.5.5)

### Report Review (v3.5.5)

```bash
# Full pipeline with review (default)
python3 tools/run_analysis.py -n 50

# Skip review for faster runs
python3 tools/run_analysis.py -n 50 --skip-review

# Review only (no pipeline suggestions)
python3 tools/run_analysis.py -n 50 --no-suggestions
```

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

# v3.8.5: Run streamlined friction tracking tests
python3 tools/test_v385_features.py

# v3.7: Run preprocessing + structured event context tests
python3 tools/test_v37_features.py

# v3.6: Run conversation quality feature tests
python3 tools/test_v36_features.py
```

## LLM Provider

**CRITICAL: Always use Gemini 3 models. Never use older models (gemini-2.5-flash, etc.)**

| Use Case | Model | Thinking Level |
|----------|-------|----------------|
| Per-transcript analysis | `gemini-3-flash-preview` | MEDIUM |
| Aggregate insights | `gemini-3-pro-preview` | MEDIUM |
| Report review | `gemini-3-pro-preview` | MEDIUM |
| Report rendering | `gemini-3-pro-preview` | MEDIUM |

### Thinking Configuration

Gemini 3 models support thinking levels. Use `MEDIUM` for balanced latency/quality:

```python
generation_config=genai.GenerationConfig(
    temperature=0.2,
    max_output_tokens=16384,
    thinking_config=genai.types.ThinkingConfig(
        thinking_level="MEDIUM"
    )
)
```

**Thinking Levels (Gemini 3 Flash):**
- `minimal`: No thinking, fastest latency
- `low`: Minimal reasoning, fast
- `medium`: Balanced (recommended for structured output)
- `high`: Deep reasoning, slower

**Note:** Thinking tokens count against `max_output_tokens`. Use higher limits (16384+) to prevent truncation.

### API Key

Set `GOOGLE_API_KEY` or `GEMINI_API_KEY` environment variable.
