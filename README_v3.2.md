# Voice Agent Analytics - v3.2 Release Notes

**Released**: 2026-01-18

## What's New in v3.2

### Configurable Parallel Processing

The batch analyzer now supports parallel processing with configurable worker threads, significantly reducing analysis time for large batches.

**New flag**: `--workers` / `-w` (default: 3)

```bash
# Default: 3 parallel workers
python3 tools/batch_analyze.py

# Custom parallelization
python3 tools/batch_analyze.py --workers 5

# Sequential mode (v3.1 behavior)
python3 tools/batch_analyze.py --workers 1

# Aggressive parallelization (use with caution)
python3 tools/batch_analyze.py --workers 10 --rate-limit 0.5
```

### Performance Improvements

| Workers | 50 Transcripts | 200 Transcripts | 300 Transcripts |
|---------|----------------|-----------------|-----------------|
| 1 (sequential) | ~2-3 min | ~8-12 min | ~12-18 min |
| 3 (default) | ~1-2 min | ~3-5 min | ~5-7 min |
| 5 | ~45-90 sec | ~2-3 min | ~3-5 min |

*Actual times depend on API response latency and rate limiting.*

### Thread-Safe Progress Tracking

New `ProgressTracker` class provides:
- Real-time progress updates with completion percentage
- Throughput rate (transcripts/second)
- ETA estimation
- Thread-safe error collection

Example output:
```
[15/50] ✓ abc123.txt: resolved | effort=2 (0.8/s, ETA 44s)
[16/50] ✓ def456.txt: escalated | effort=4 (0.8/s, ETA 42s)
```

### Exponential Backoff on Retries

Failed API calls now use exponential backoff:
- Retry 1: `rate_limit * 2` seconds
- Retry 2: `rate_limit * 4` seconds
- Retry 3: `rate_limit * 8` seconds

This improves resilience against transient API errors and rate limiting.

### Pipeline Integration

`run_analysis.py` now accepts `--workers` flag:

```bash
# Full pipeline with 5 workers
python3 tools/run_analysis.py -n 200 --workers 5

# Quick test still works
python3 tools/run_analysis.py --quick
```

## Files Changed

| File | Change |
|------|--------|
| `tools/batch_analyze.py` | **Major** - Added parallel processing with ThreadPoolExecutor |
| `tools/run_analysis.py` | Added `--workers` flag, updated to v3.2 |

## Migration from v3.1

No breaking changes. Default behavior (3 workers) provides ~3x speedup over sequential processing.

To restore v3.1 sequential behavior:
```bash
python3 tools/batch_analyze.py --workers 1
```

## API Rate Limiting Considerations

When using multiple workers, be mindful of API rate limits:

- **Gemini API**: Generally handles 3-5 concurrent requests well
- **Conservative**: `--workers 3 --rate-limit 1.0` (default)
- **Moderate**: `--workers 5 --rate-limit 0.5`
- **Aggressive**: `--workers 10 --rate-limit 0.2` (may hit rate limits)

If you encounter `429 Too Many Requests` errors, reduce workers or increase rate-limit.

## Verification

```bash
# Verify batch_analyze parallel mode
python3 tools/batch_analyze.py --help | grep workers
# Expected: -w WORKERS, --workers WORKERS

# Test with existing test data (no API calls needed for metrics)
python3 tools/test_framework.py
# Expected: 17/17 tests passed
```
