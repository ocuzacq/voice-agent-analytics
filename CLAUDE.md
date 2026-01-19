# Voice Agent Analytics - Project Instructions (v3.5)

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

## Pipeline Architecture (v3.2)

```
transcripts/ → sample → analyze (parallel) → metrics → extract_nl → insights → report
```

1. `sample_transcripts.py` - Random stratified sampling
2. `batch_analyze.py` - LLM analysis with parallel processing (v3.2: default 3 workers)
3. `compute_metrics.py` - Section A: Deterministic metrics
4. `extract_nl_fields.py` - Condensed NL data for LLM (v3.1)
5. `generate_insights.py` - Section B: LLM insights
6. `render_report.py` - Markdown executive summary

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
```

## LLM Provider

- Use `gemini-2.5-flash` for all LLM calls
- API key: `GOOGLE_API_KEY` or `GEMINI_API_KEY`
