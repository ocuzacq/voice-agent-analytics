# Voice Agent Analytics v4.3 - Target-Based Run Augmentation

## What Changed

**v4.3** replaces the manual `--augment` / `--no-clear` workflow with automatic target-based sizing. The `-n` flag becomes a **TARGET** when a run already has samples, and the system calculates the delta automatically.

**Release Date**: 2026-01-23

---

## The Problem

Before v4.3, growing a run iteratively required manual bookkeeping:

```bash
# Day 1: Create run with 50
python3 tools/run_analysis.py -n 50

# Day 2: Want 200 total — had to know you already have 50, compute 150 yourself
python3 tools/run_analysis.py --augment --no-clear -n 150  # Error-prone
```

You had to remember the current count, compute the delta, and pass the right combination of flags. Getting any of this wrong risked overwriting existing work or sampling duplicates.

## The Solution

With v4.3, just state the total you want:

```bash
# Day 1: Create run with 50
python3 tools/run_analysis.py -n 50

# Day 2: Grow to 200 total (system figures out: +150)
python3 tools/run_analysis.py -n 200

# Day 3: Grow to 1000 total (system figures out: +800)
python3 tools/run_analysis.py -n 1000

# Already met? Nothing happens
python3 tools/run_analysis.py -n 500
# → "Run already has 1000 transcripts (target: 500). Nothing to add."
```

---

## How It Works

### Target Calculation

When a run's `manifest.csv` already exists:

1. Count existing entries in manifest → `existing_count`
2. Compute `delta = target - existing_count`
3. If `delta <= 0` → skip sampling, print "Nothing to add"
4. If `delta > 0` → sample only `delta` new transcripts in append mode

### Duplicate Prevention

In append mode, already-sampled file stems are read from `manifest.csv` and excluded from the candidate pool. This guarantees no transcript is sampled twice.

### Manifest Merging

New entries are appended to the existing `manifest.csv` (not replaced). The merged manifest is sorted by file size for readability.

### Implementation Locations

| File | What It Does |
|------|-------------|
| `run_analysis.py:233-252` | Reads manifest, computes delta, sets `sample_size = delta` |
| `sample_transcripts.py:282-305` | Target-based logic + `--force-count` override |
| `sample_transcripts.py:63-88` | `get_already_sampled_stems()` — reads manifest for exclusion |
| `sample_transcripts.py:174-203` | `write_manifest()` — append mode merges with existing |

---

## Usage

### Basic Workflow: Iterative Growth

```bash
# Step 1: Start small
python3 tools/run_analysis.py -n 5 --run-id test_augment

# Step 2: Grow when ready
python3 tools/run_analysis.py -n 20 --run-dir runs/test_augment

# Step 3: Full scale
python3 tools/run_analysis.py -n 200 --run-dir runs/test_augment
```

### Using sample_transcripts.py Directly

```bash
# Target-based (default behavior)
python3 tools/sample_transcripts.py -n 100 --run-dir runs/my_run

# Force exact count (overrides target logic, clears existing)
python3 tools/sample_transcripts.py -n 50 --run-dir runs/my_run --force-count
```

### Three Pipeline Modes

```bash
# Mode 1: Sample only (fastest — for ask_raw.py workflow)
python3 tools/run_analysis.py -n 50 --sample-only
python3 tools/ask_raw.py "What are the main failure patterns?"

# Mode 2: Sample + analyze (default — for ask.py workflow)
python3 tools/run_analysis.py -n 50
python3 tools/ask.py "What are the main failure patterns?"

# Mode 3: Full pipeline with insights/report (opt-in)
python3 tools/run_analysis.py -n 200 --insights
```

### Two Q&A Tools

| | `ask_raw.py` | `ask.py` |
|-|-------------|----------|
| **Needs** | `--sample-only` (sample only) | Default (sample + analyze) |
| **Speed** | Fastest (1 LLM call total) | N analysis calls + 1 Q&A call |
| **Data** | Full conversation text | Structured fields (intent, disposition, scores) |
| **Best for** | Exploration, open-ended questions | Targeted queries on extracted metrics |

---

## CLI Changes

### run_analysis.py

| Argument | Behavior |
|----------|----------|
| `-n 50` (new run) | Sample 50 transcripts, create run |
| `-n 200` (existing run with 50) | Add 150 more transcripts to reach 200 |
| `-n 30` (existing run with 50) | "Nothing to add" — target already exceeded |
| `--sample-only` | Stop after sampling (for `ask_raw.py` workflow) |
| `--insights` | Run full pipeline: metrics, NL extraction, insights, report |
| `--run-dir runs/X` | Use specific existing run directory |
| `--run-id name` | Create or use `runs/name/` |

**Note:** `--skip-insights` is deprecated (no-op). Insights are now off by default; use `--insights` to enable.

### sample_transcripts.py

| Argument | Behavior |
|----------|----------|
| `-n 100` (existing manifest with 30) | Add 70 to reach 100 |
| `-n 100 --force-count` | Clear existing, sample exactly 100 |
| `--no-clear` | Legacy append mode (still works) |

---

## Console Output

### New Run

```
Creating isolated run: runs/run_20260206_120000
  (saved to .last_run - all tools will use this automatically)

Configuration:
  Mode: ISOLATED
  Run directory: runs/run_20260206_120000
  Sample size: 50
```

### Augmenting Existing Run

```
Found existing run: runs/run_20260206_120000

Run has 50 transcripts. Adding 150 to reach target of 200.

Configuration:
  Mode: ISOLATED
  Run directory: runs/run_20260206_120000
  Target: 200 transcripts (50 existing + 150 new)
```

### Target Already Met

```
Run already has 200 transcripts (target: 100). Nothing to add.
```

---

## Key Design Decisions

### Why TARGET, Not COUNT?

The mental model is "I want N transcripts total" rather than "add N more." This is:
- **Safer**: No risk of accidentally doubling your sample
- **Idempotent**: Running the same command twice is a no-op
- **Intuitive**: "I want 200 analyses" is clearer than "add 150 more"

### Why --force-count?

Escape hatch for when you genuinely want to clear and resample. Useful for:
- Starting over with a different random seed
- Changing stratification parameters
- Debugging sampling issues

### Manifest as Source of Truth

The manifest (not directory listing) determines existing count. This is more reliable because:
- Handles cases where files are manually deleted
- Tracks original source paths
- Preserves stratification metadata (large/small category)

---

## Backwards Compatibility

- `--no-clear` flag still works as before (manual append mode)
- `--legacy` flag still works for flat directory mode
- Runs created before v4.3 work fine — they just won't have a manifest, so first `-n` creates one
- No schema changes (still v4.0 analysis schema)

---

## Files Modified

| File | Changes |
|------|---------|
| `run_analysis.py` | Target-based delta calculation; `--force-count` fix; `--sample-only` mode; `--insights` opt-in (replaces `--skip-insights`) |
| `sample_transcripts.py` | Target logic + `--force-count` flag + append-mode exclusion |
| `ask_raw.py` | Added `--run-dir` / `--run-id` support for run isolation |
| `ask.py` | Improved analysis loading (skip non-analysis files, validate `call_id`) |
| `CLAUDE.md` | Pipeline modes, two Q&A tools documentation |
| `README.md` | Three pipeline modes, `ask_raw.py` tool section, updated Quick Start |

### Bug Fix: Double Target-Based Logic (2026-02-06)

The initial v4.3 implementation had a bug where the target-based delta was computed twice:
1. `run_analysis.py` computes `delta = target - existing` and passes `-n delta`
2. `sample_transcripts.py` re-applies target logic: sees manifest has N entries, target=delta, computes a second delta of 0

**Fix**: When `run_analysis.py` has already computed the delta, it now passes `--force-count` to `sample_transcripts.py` so the subprocess treats `-n` as an exact count rather than a target.

---

## Testing

```bash
# End-to-end augmentation test (sample + analyze)
python3 tools/run_analysis.py -n 5 --run-id test_v43
python3 tools/run_analysis.py -n 10 --run-dir runs/test_v43

# Verify manifest grew from 5 to 10
wc -l runs/test_v43/sampled/manifest.csv  # Should show 11 (header + 10 entries)

# Verify no duplicates
cut -d, -f1 runs/test_v43/sampled/manifest.csv | sort | uniq -d  # Should be empty

# Verify target-met behavior
python3 tools/run_analysis.py -n 5 --run-dir runs/test_v43
# → "Run already has 10 transcripts (target: 5). Nothing to add."

# Force-count override
python3 tools/sample_transcripts.py -n 3 --run-dir runs/test_v43 --force-count
wc -l runs/test_v43/sampled/manifest.csv  # Should show 4 (header + 3 entries)

# Sample-only mode (for ask_raw.py)
python3 tools/run_analysis.py -n 20 --run-id test_raw --sample-only
python3 tools/ask_raw.py "What are common caller concerns?"
# → Uses sampled transcripts directly, no analysis needed
```

---

## What's Next

Potential v4.4+ enhancements:
- Analysis-aware augmentation: only analyze newly-added transcripts (currently re-analyzes all)
- Progress bar during augmentation showing existing/new/total
- Manifest diffing: show what changed between augmentation steps
- Custom question sets per run (stored in `config.json`)
