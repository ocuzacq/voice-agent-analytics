# Voice Agent Analytics v4.1 - Run-Based Isolation

## What Changed

**v4.1** introduces run-based isolation to eliminate the "mutable shared state" problem where each new analysis run would overwrite previous results in `sampled/`, `analyses/`, and `reports/`.

### New Features

1. **Isolated Run Directories**: Each run is self-contained in `runs/<run_id>/`
2. **Interactive Run Selection**: First-time use prompts to create/select a run
3. **Per-Terminal Tracking**: Each terminal maintains its own "last run" pointer
4. **Reproducibility**: `config.json` captures all CLI parameters for exact replay
5. **Progress Tracking**: `status.json` tracks pipeline stage completion
6. **Concurrent Runs**: Multiple analyses can run simultaneously without conflicts
7. **Resume Support**: `--run-dir` + `--resume` continues interrupted runs
8. **Scoped Q&A**: `ask.py --run-dir` queries analyses from specific runs
9. **Latest Symlink**: `runs/latest` always points to most recent run
10. **Legacy Mode**: `--legacy` flag preserves flat directory behavior

## Run Directory Structure

```
runs/
├── run_20260121_143000/
│   ├── config.json           # All CLI args + source stats
│   ├── status.json           # Pipeline progress tracking
│   ├── manifest.csv          # Sample scope anchor
│   ├── sampled/              # Transcript copies
│   ├── analyses/             # Analysis JSONs
│   └── reports/              # Metrics, insights, markdown
└── latest -> run_20260121_143000  # Symlink to most recent
```

### config.json Example

```json
{
  "run_id": "run_20260121_143000",
  "created_at": "2026-01-21T14:30:00",
  "version": "4.1",
  "args": {
    "sample_size": 50,
    "workers": 3,
    "seed": null,
    "analysis_model": "gemini-3-flash-preview",
    "insights_model": "gemini-3-pro-preview"
  },
  "source": {
    "transcripts_dir": "/path/to/transcripts",
    "total_available": 6154
  }
}
```

### status.json Example

```json
{
  "run_id": "run_20260121_143000",
  "updated_at": "2026-01-21T14:35:00",
  "overall_status": "completed",
  "stages": {
    "sampling": {"status": "completed", "count": 50},
    "analysis": {"status": "completed", "count": 48},
    "metrics": {"status": "completed"},
    "nl_extraction": {"status": "completed"},
    "insights": {"status": "completed"},
    "report": {"status": "completed"}
  }
}
```

## Usage Examples

### Returning User: Confirmation Prompt

When a `.last_run` file exists, tools show the resolved run and ask for confirmation:

```
============================================================
CONTINUE WITH RUN? (from .last_run)
============================================================

  Run ID:    experiment_a
  Path:      runs/experiment_a
  Modified:  2026-01-21 14:30
  Contents:  50 sampled, 48 analyses

Options:
  [Enter] Continue with this run
  [c]     Choose a DIFFERENT run
  [q]     Quit

Select option: _
```

### First-Time Use: Selection Menu

When no run is specified and no `.last_run` file exists, tools show a selection menu:

```
============================================================
RUN SELECTION
============================================================

Existing runs (3):
  [1] experiment_b  (2026-01-21 15:30)
  [2] experiment_a  (2026-01-21 14:00)
  [3] run_20260120_093000  (2026-01-20 09:30)

Options:
  [n] Create NEW run (auto-generated timestamp ID)
  [c] Create NEW run with CUSTOM ID
  [1-5] Use existing run (number from list above)
  [l] Use LEGACY mode (flat directories, no isolation)
  [q] Quit

Select option: _
```

### Default: Isolated Run

```bash
# Creates runs/run_20260121_143000/
python3 tools/run_analysis.py -n 50
```

### Custom Run ID

```bash
# Creates runs/experiment_a/
python3 tools/run_analysis.py -n 50 --run-id experiment_a
```

### Resume Interrupted Run

```bash
# Continues from where it left off
python3 tools/run_analysis.py --run-dir runs/experiment_a --resume
```

### Scoped Q&A

```bash
# Query specific run's analyses
python3 tools/ask.py "Why do calls fail?" --run-dir runs/experiment_a

# Query latest run
python3 tools/ask.py "Main friction patterns?" --run-dir runs/latest
```

### Legacy Mode

```bash
# Uses flat sampled/, analyses/, reports/ directories
python3 tools/run_analysis.py --legacy
```

### Step-by-Step with --run-id (Recommended)

```bash
# --run-id auto-creates runs/<id>/ if it doesn't exist
python3 tools/sample_transcripts.py -n 50 --run-id my_run
python3 tools/batch_analyze.py --run-id my_run
python3 tools/compute_metrics.py --run-id my_run
python3 tools/extract_nl_fields.py --run-id my_run
python3 tools/generate_insights.py --run-id my_run
python3 tools/render_report.py --run-id my_run
```

### Step-by-Step with --run-dir (Explicit Path)

```bash
# --run-dir uses an existing directory
python3 tools/sample_transcripts.py -n 50 --run-dir runs/my_run
python3 tools/batch_analyze.py --run-dir runs/my_run
# etc.
```

## Why This Matters

### Before v4.1

- Each run overwrote previous state
- Impossible to compare different sampling strategies
- Interrupted runs required manual folder management
- Concurrent analyses would conflict
- No reproducibility without manual tracking

### After v4.1

- All runs preserved for comparison
- `config.json` enables exact reproduction
- `--resume` handles interruptions gracefully
- Multiple runs can execute simultaneously
- `runs/latest` always points to most recent

## Migration

- **No automatic migration**: Existing `sampled/`, `analyses/`, `reports/` directories remain untouched
- **Legacy mode preserved**: Use `--legacy` for old behavior
- **Manual archival optional**: Move old directories to `runs/legacy/` if desired

## Modified Files

| File | Changes |
|------|---------|
| `tools/run_utils.py` | **NEW**: Shared utilities for run isolation |
| `tools/run_analysis.py` | Run directory creation, `--run-dir`, `--run-id`, `--legacy` |
| `tools/sample_transcripts.py` | `--run-dir` and `--run-id` support |
| `tools/batch_analyze.py` | `--run-dir` and `--run-id` support |
| `tools/compute_metrics.py` | `--run-dir` and `--run-id` support |
| `tools/extract_nl_fields.py` | `--run-dir` and `--run-id` support |
| `tools/generate_insights.py` | `--run-dir` and `--run-id` support |
| `tools/render_report.py` | `--run-dir` and `--run-id` support |
| `tools/ask.py` | `--run-dir` and `--run-id` support, run_id in metadata |

## Arguments & Automatic Run Tracking

All tools support these options:

| Option | Description |
|--------|-------------|
| `--run-dir <path>` | Use existing directory at `<path>` (must exist) |
| `--run-id <name>` | Create/use `runs/<name>/` (auto-creates if needed) |
| `$LAST_RUN` | Environment variable override |
| `.last_run` | **Auto-updated file** with last created run |

**Priority**: `--run-dir` > `--run-id` > `$LAST_RUN` > `.last_run.<tty>` > `.last_run` > legacy mode

### Automatic Workflow (Zero Config)

When any tool creates a new run, it saves the run_id to a TTY-specific `.last_run` file. All subsequent tools in that terminal automatically use it:

```bash
# Creates run and saves to .last_run.ttys003 (for this terminal)
python3 tools/run_analysis.py -n 50
# Output: Creating isolated run: runs/run_20260121_143000
#         (saved to .last_run.ttys003 - tools in this terminal will use it)

# These automatically use runs/run_20260121_143000/
python3 tools/ask.py "Why do calls fail?"
python3 tools/batch_analyze.py --limit 10

# Override for a single command
python3 tools/ask.py "Compare" --run-id other_run
```

### Per-Terminal Isolation

Each terminal gets its own `.last_run` file based on its TTY:

```
Terminal A (ttys003):  .last_run.ttys003 → "experiment_a"
Terminal B (ttys004):  .last_run.ttys004 → "experiment_b"
```

This allows concurrent work in multiple terminals without conflicts:

```bash
# Terminal A
python3 tools/run_analysis.py -n 50 --run-id experiment_a
python3 tools/ask.py "Why do calls fail?"  # Uses experiment_a

# Terminal B (simultaneously)
python3 tools/run_analysis.py -n 100 --run-id experiment_b
python3 tools/ask.py "Main friction?"  # Uses experiment_b
```

For non-interactive contexts (cron, scripts), the global `.last_run` file is used.

### Directory Structure

Each run contains its own subdirectories:
```
runs/
├── run_20260121_143000/
│   ├── sampled/      # Transcript copies
│   ├── analyses/     # Analysis JSONs
│   └── reports/      # Metrics, insights, markdown
└── latest -> run_20260121_143000

.last_run.ttys003     # Terminal A's last run
.last_run.ttys004     # Terminal B's last run
.last_run             # Global fallback (non-interactive)
```

## Non-Interactive Mode (Scripts, CI/CD)

In non-interactive contexts, `--run-id` or `--run-dir` is **required**:

```bash
# In a script or cron job - MUST specify run explicitly
python3 tools/batch_analyze.py --run-id my_experiment

# Error if not specified:
# ERROR: Non-interactive mode requires explicit --run-id or --run-dir
```

This prevents silent fallback to legacy mode in automation, ensuring scripts always operate on the intended run.

## Backwards Compatibility

- All existing workflows work unchanged with `--legacy`
- Individual tools default to legacy paths when `--run-dir` not specified
- Schema version unchanged (v4.0 → v4.0), only pipeline infrastructure changed
