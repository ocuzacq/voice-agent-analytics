#!/usr/bin/env python3
"""
Generic batch analyzer for v7.0 schema (impediment/agent_issue model).

Analyzes transcripts using Gemini structured output and writes per-call
JSON files to the specified output directory. Supports resume (skips
already-analyzed transcripts) and parallel workers.

Usage:
    source .env

    # Analyze a directory of transcripts
    python3 tools/batch_analyze_v7.py --input-dir transcripts/ --output-dir runs/v7_batch/

    # Analyze from a file list (one path per line)
    python3 tools/batch_analyze_v7.py --transcript-list /tmp/recent_500.txt --output-dir runs/v7_batch/

    # Parallel with 3 workers
    python3 tools/batch_analyze_v7.py --input-dir transcripts/ --output-dir runs/v7_batch/ --workers 3
"""

import json
import os
import sys
import time
import queue
import threading
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_transcript import preprocess_transcript, format_for_llm
from schema import CallAnalysis, CallRecord, SCHEMA_VERSION
from poc_structured_full import SYSTEM_PROMPT, _map_ended_reason, _override_queue_result


def analyze_one(client: genai.Client, transcript_path: Path, config: types.GenerateContentConfig, model_name: str) -> dict:
    """Analyze one transcript and return the result dict."""
    preprocessed = preprocess_transcript(transcript_path)
    call_id = preprocessed["call_id"]
    formatted = format_for_llm(preprocessed)

    t0 = time.time()
    response = client.models.generate_content(
        model=model_name,
        contents=f"Analyze this call transcript:\n\n{formatted}",
        config=config,
    )
    elapsed = time.time() - t0

    ended_reason_raw = preprocessed.get("metadata", {}).get("ended_reason")

    if hasattr(response, 'parsed') and response.parsed is not None:
        analysis: CallAnalysis = response.parsed
        _override_queue_result(analysis, ended_reason_raw)
        record = CallRecord(
            **analysis.model_dump(),
            call_id=call_id,
            schema_version=SCHEMA_VERSION,
            duration_seconds=round(preprocessed.get("metadata", {}).get("duration", 0), 1) or None,
            ended_by=_map_ended_reason(ended_reason_raw),
            ended_reason=ended_reason_raw,
        )
        return {
            "success": True,
            "call_id": call_id,
            "record": record.model_dump(),
            "elapsed": elapsed,
            "primary_scope": analysis.resolution.primary.scope,
            "primary_outcome": analysis.resolution.primary.outcome,
            "primary_request": analysis.resolution.primary.request,
            "request_category": analysis.resolution.primary.request_category,
            "has_secondary": analysis.resolution.secondary is not None,
            "impediment": analysis.impediment.type if analysis.impediment else None,
            "agent_issue": analysis.agent_issue is not None,
        }
    else:
        # Try manual parse
        raw = response.text if hasattr(response, 'text') else ""
        try:
            data = json.loads(raw)
            analysis = CallAnalysis.model_validate(data)
            _override_queue_result(analysis, ended_reason_raw)
            record = CallRecord(
                **analysis.model_dump(),
                call_id=call_id,
                schema_version=SCHEMA_VERSION,
                duration_seconds=round(preprocessed.get("metadata", {}).get("duration", 0), 1) or None,
                ended_by=_map_ended_reason(ended_reason_raw),
                ended_reason=ended_reason_raw,
            )
            return {
                "success": True,
                "call_id": call_id,
                "record": record.model_dump(),
                "elapsed": elapsed,
                "primary_scope": analysis.resolution.primary.scope,
                "primary_outcome": analysis.resolution.primary.outcome,
                "primary_request": analysis.resolution.primary.request,
                "request_category": analysis.resolution.primary.request_category,
                "has_secondary": analysis.resolution.secondary is not None,
                "impediment": analysis.impediment.type if analysis.impediment else None,
                "agent_issue": analysis.agent_issue is not None,
                "note": "manual_parse",
            }
        except Exception as e:
            return {
                "success": False,
                "call_id": call_id,
                "elapsed": elapsed,
                "error": str(e),
            }


def _is_rate_limit(e: Exception) -> bool:
    """Check if exception is a 429 / RESOURCE_EXHAUSTED error."""
    msg = str(e).lower()
    return "429" in msg or "resource_exhausted" in msg or "quota" in msg


def _analyze_with_retry(client, transcript_path, config, model_name, max_retries=5, rate_limit=0.5):
    """Call analyze_one with exponential backoff on errors."""
    call_id = transcript_path.stem
    for attempt in range(max_retries):
        try:
            return analyze_one(client, transcript_path, config, model_name)
        except Exception as e:
            if attempt < max_retries - 1:
                if _is_rate_limit(e):
                    wait = min(max(rate_limit, 2.0) * (2 ** (attempt + 1)), 120)
                else:
                    wait = rate_limit * (2 ** (attempt + 1))
                print(f"    {call_id}: {str(e)[:60]} — retry {attempt+1}/{max_retries-1} in {wait:.0f}s")
                time.sleep(wait)
            else:
                return {"success": False, "call_id": call_id, "elapsed": 0, "error": str(e)}
    return {"success": False, "call_id": call_id, "elapsed": 0, "error": "max retries exceeded"}


def _worker(task_queue, result_list, result_lock, output_dir, config, model_name, total, api_key, rate_limit=0.5, max_retries=5):
    """Worker thread: pull transcripts from queue, analyze, save."""
    client = genai.Client(api_key=api_key)
    while True:
        try:
            idx, tp = task_queue.get_nowait()
        except Exception:
            break
        call_id = tp.stem

        result = _analyze_with_retry(client, tp, config, model_name, max_retries=max_retries, rate_limit=rate_limit)

        if result["success"]:
            out_path = output_dir / f"{call_id}.json"
            with open(out_path, "w") as f:
                json.dump(result["record"], f, indent=2)

        with result_lock:
            result_list.append(result)
            done = len(result_list)
            if result["success"]:
                scope = result["primary_scope"]
                outcome = result["primary_outcome"]
                cat_tag = f" cat={result['request_category']}" if result.get("request_category") else ""
                secondary = " +2nd" if result["has_secondary"] else ""
                imp_tag = f" IMP:{result['impediment']}" if result.get("impediment") else ""
                ai_tag = " AGENT_ISSUE" if result.get("agent_issue") else ""
                print(f"  [{done}/{total}] {call_id}  OK ({result['elapsed']:.1f}s) [{scope}:{outcome}]{cat_tag}{secondary}{imp_tag}{ai_tag}")
            else:
                print(f"  [{done}/{total}] {call_id}  FAILED: {result.get('error', '?')[:80]}")

        task_queue.task_done()
        if rate_limit > 0:
            time.sleep(rate_limit)


def run_batch(
    transcripts: list[Path],
    output_dir: Path,
    *,
    workers: int = 1,
    delay: float = 0.5,
    max_retries: int = 5,
    model: str = "gemini-3-flash-preview",
    api_key: str | None = None,
    config: types.GenerateContentConfig | None = None,
) -> list[dict]:
    """Run batch analysis on a list of transcript paths.

    Args:
        transcripts: List of transcript file paths to analyze.
        output_dir: Directory to write per-call JSON results.
        workers: Number of parallel workers (1 = sequential).
        delay: Seconds between API calls (both sequential and parallel).
        max_retries: Max retries per transcript on transient/rate-limit errors.
        model: Gemini model name.
        api_key: Google API key. Falls back to env vars if None.
        config: Optional GenerateContentConfig override. If None, uses default.

    Returns:
        List of result dicts (one per transcript), each with 'success' bool.
    """
    api_key = api_key or os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("No API key: set GOOGLE_API_KEY or GEMINI_API_KEY")

    if config is None:
        config = types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=16384,
            system_instruction=SYSTEM_PROMPT,
            thinking_config=types.ThinkingConfig(thinking_level="LOW"),
            response_mime_type="application/json",
            response_schema=CallAnalysis,
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-analyzed transcripts
    existing = {p.stem for p in output_dir.glob("*.json")}
    before = len(transcripts)
    transcripts = [t for t in transcripts if t.stem not in existing]
    skipped = before - len(transcripts)

    print(f"Transcripts: {before} found, {skipped} already analyzed, {len(transcripts)} to process")
    print(f"Model: {model}  Workers: {workers}")
    print(f"Output: {output_dir}/")
    print(f"Schema: {SCHEMA_VERSION}")
    print("=" * 70)

    if not transcripts:
        print("Nothing to do.")
        return []

    total = len(transcripts)

    print(f"Rate limit: {delay}s per call, Max retries: {max_retries}")

    if workers <= 1:
        # Sequential mode
        client = genai.Client(api_key=api_key)
        results = []
        total_t0 = time.time()

        for i, tp in enumerate(transcripts):
            result = _analyze_with_retry(client, tp, config, model, max_retries=max_retries, rate_limit=delay)
            results.append(result)

            if result["success"]:
                out_path = output_dir / f"{tp.stem}.json"
                with open(out_path, "w") as f:
                    json.dump(result["record"], f, indent=2)

                scope = result["primary_scope"]
                outcome = result["primary_outcome"]
                cat_tag = f" cat={result['request_category']}" if result.get("request_category") else ""
                secondary = " +2nd" if result["has_secondary"] else ""
                imp_tag = f" IMP:{result['impediment']}" if result.get("impediment") else ""
                ai_tag = " AGENT_ISSUE" if result.get("agent_issue") else ""
                print(f"  [{i+1}/{total}] {tp.stem}  OK ({result['elapsed']:.1f}s) [{scope}:{outcome}]{cat_tag}{secondary}{imp_tag}{ai_tag}")
            else:
                print(f"  [{i+1}/{total}] {tp.stem}  FAILED: {result.get('error', '?')[:80]}")

            if delay > 0 and i < total - 1:
                time.sleep(delay)
    else:
        # Parallel mode
        task_queue = queue.Queue()
        for i, tp in enumerate(transcripts):
            task_queue.put((i, tp))

        results = []
        result_lock = threading.Lock()
        total_t0 = time.time()

        threads = []
        for _ in range(workers):
            t = threading.Thread(target=_worker,
                                 args=(task_queue, results, result_lock, output_dir,
                                       config, model, total, api_key, delay, max_retries))
            t.start()
            threads.append(t)

        for t in threads:
            t.join()

    total_elapsed = time.time() - total_t0
    ok = sum(1 for r in results if r["success"])
    fail = len(results) - ok

    # Summary
    print("\n" + "=" * 70)
    print(f"DONE: {ok}/{len(results)} succeeded, {fail} failed — {total_elapsed:.0f}s total")
    if skipped:
        print(f"  ({skipped} previously analyzed, {ok + skipped} total in {output_dir}/)")
    print()

    failures = [r for r in results if not r["success"]]
    if failures:
        print(f"FAILED ({len(failures)}):")
        for r in failures:
            print(f"  {r['call_id']}: {r.get('error', 'unknown')[:120]}")

    return results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Batch-analyze transcripts with v7.0 schema")
    parser.add_argument("--input-dir", default=None, help="Directory of transcript JSONs")
    parser.add_argument("--transcript-list", default=None, help="File with one transcript path per line")
    parser.add_argument("--output-dir", required=True, help="Output directory for analysis JSONs")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls per worker (seconds)")
    parser.add_argument("--max-retries", type=int, default=5, help="Max retries per transcript on errors (default: 5)")
    args = parser.parse_args()

    if not args.input_dir and not args.transcript_list:
        parser.error("Provide --input-dir or --transcript-list")

    # Resolve transcript list
    if args.transcript_list:
        with open(args.transcript_list) as f:
            transcripts = [Path(line.strip()) for line in f if line.strip()]
    else:
        transcripts = sorted(Path(args.input_dir).glob("*.json"))

    run_batch(
        transcripts=transcripts,
        output_dir=Path(args.output_dir),
        workers=args.workers,
        delay=args.delay,
        max_retries=args.max_retries,
        model=args.model,
    )


if __name__ == "__main__":
    main()
