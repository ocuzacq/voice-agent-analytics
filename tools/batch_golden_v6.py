#!/usr/bin/env python3
"""
Batch-analyze transcripts with the v6.0 per-intent schema.

Defaults to golden test set; supports arbitrary transcript sources
via --input-dir or --transcript-list.

Usage:
    source .env

    # Golden set (default)
    python3 tools/batch_golden_v6.py

    # Arbitrary directory
    python3 tools/batch_golden_v6.py --input-dir transcripts/ --output-dir runs/v6_batch/

    # File list (one path per line)
    python3 tools/batch_golden_v6.py --transcript-list /tmp/recent_500.txt --output-dir runs/v6_recent500/

    # Parallel (3 workers) with resume
    python3 tools/batch_golden_v6.py --transcript-list /tmp/recent_500.txt --output-dir runs/v6_recent500/ --workers 3
"""

import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

sys.path.insert(0, str(Path(__file__).parent))
from preprocess_transcript import preprocess_transcript, format_for_llm
from schema import CallAnalysis, CallRecord, SCHEMA_VERSION

# Reuse the system prompt from poc_structured_full.py
from poc_structured_full import SYSTEM_PROMPT, _map_ended_reason


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

    if hasattr(response, 'parsed') and response.parsed is not None:
        analysis: CallAnalysis = response.parsed
        record = CallRecord(
            **analysis.model_dump(),
            call_id=call_id,
            schema_version=SCHEMA_VERSION,
            duration_seconds=round(preprocessed.get("metadata", {}).get("duration", 0), 1) or None,
            ended_by=_map_ended_reason(preprocessed.get("metadata", {}).get("ended_reason")),
        )
        return {
            "success": True,
            "call_id": call_id,
            "record": record.model_dump(),
            "elapsed": elapsed,
            "primary_scope": analysis.resolution.primary.scope,
            "primary_outcome": analysis.resolution.primary.outcome,
            "primary_request": analysis.resolution.primary.request,
            "has_secondary": analysis.resolution.secondary is not None,
            "failure": analysis.failure.type if analysis.failure else None,
        }
    else:
        # Try manual parse
        raw = response.text if hasattr(response, 'text') else ""
        try:
            data = json.loads(raw)
            analysis = CallAnalysis.model_validate(data)
            record = CallRecord(
                **analysis.model_dump(),
                call_id=call_id,
                schema_version=SCHEMA_VERSION,
                duration_seconds=round(preprocessed.get("metadata", {}).get("duration", 0), 1) or None,
                ended_by=_map_ended_reason(preprocessed.get("metadata", {}).get("ended_reason")),
            )
            return {
                "success": True,
                "call_id": call_id,
                "record": record.model_dump(),
                "elapsed": elapsed,
                "primary_scope": analysis.resolution.primary.scope,
                "primary_outcome": analysis.resolution.primary.outcome,
                "primary_request": analysis.resolution.primary.request,
                "has_secondary": analysis.resolution.secondary is not None,
                "failure": analysis.failure.type if analysis.failure else None,
                "note": "manual_parse",
            }
        except Exception as e:
            return {
                "success": False,
                "call_id": call_id,
                "elapsed": elapsed,
                "error": str(e),
            }


def _worker(task_queue, result_list, result_lock, output_dir, config, model_name, total, api_key):
    """Worker thread: pull transcripts from queue, analyze, save."""
    client = genai.Client(api_key=api_key)
    while True:
        try:
            idx, tp = task_queue.get_nowait()
        except Exception:
            break
        call_id = tp.stem
        try:
            result = analyze_one(client, tp, config, model_name)
        except Exception as e:
            result = {"success": False, "call_id": call_id, "elapsed": 0, "error": str(e)}

        if result["success"]:
            out_path = output_dir / f"{call_id}.json"
            with open(out_path, "w") as f:
                json.dump(result["record"], f, indent=2)

        with result_lock:
            result_list.append(result)
            done = len(result_list)
            ok = sum(1 for r in result_list if r["success"])
            fail = done - ok
            if result["success"]:
                scope = result["primary_scope"]
                outcome = result["primary_outcome"]
                secondary = " +2nd" if result["has_secondary"] else ""
                failure_tag = f" FAIL:{result['failure']}" if result["failure"] else ""
                print(f"  [{done}/{total}] {call_id}  OK ({result['elapsed']:.1f}s) [{scope}:{outcome}]{secondary}{failure_tag}")
            else:
                print(f"  [{done}/{total}] {call_id}  FAILED: {result.get('error', '?')[:80]}")

        task_queue.task_done()


def main():
    import argparse
    import threading
    import queue

    parser = argparse.ArgumentParser(description="Batch-analyze transcripts with v6.0 schema")
    parser.add_argument("--model", default="gemini-3-flash-preview")
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between API calls per worker (seconds)")
    parser.add_argument("--output-dir", default="tests/golden/analyses_v6_review", help="Output directory")
    parser.add_argument("--input-dir", default=None, help="Directory of transcript JSONs (default: golden set)")
    parser.add_argument("--transcript-list", default=None, help="File with one transcript path per line")
    parser.add_argument("--workers", type=int, default=1, help="Parallel workers (default: 1)")
    args = parser.parse_args()

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("Error: Set GOOGLE_API_KEY or GEMINI_API_KEY", file=sys.stderr)
        sys.exit(1)

    config = types.GenerateContentConfig(
        temperature=0.2,
        max_output_tokens=16384,
        system_instruction=SYSTEM_PROMPT,
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
        response_mime_type="application/json",
        response_schema=CallAnalysis,
    )

    # Resolve transcript list
    if args.transcript_list:
        with open(args.transcript_list) as f:
            transcripts = [Path(line.strip()) for line in f if line.strip()]
    elif args.input_dir:
        transcripts = sorted(Path(args.input_dir).glob("*.json"))
    else:
        transcripts = sorted(Path("tests/golden/transcripts").glob("*.json"))

    # Output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resume: skip already-analyzed transcripts
    existing = {p.stem for p in output_dir.glob("*.json")}
    before = len(transcripts)
    transcripts = [t for t in transcripts if t.stem not in existing]
    skipped = before - len(transcripts)

    print(f"Transcripts: {before} found, {skipped} already analyzed, {len(transcripts)} to process")
    print(f"Model: {args.model}  Workers: {args.workers}")
    print(f"Output: {output_dir}/")
    print(f"Schema: {SCHEMA_VERSION}")
    print("=" * 70)

    if not transcripts:
        print("Nothing to do.")
        return

    total = len(transcripts)

    if args.workers <= 1:
        # Sequential mode
        client = genai.Client(api_key=api_key)
        results = []
        total_t0 = time.time()

        for i, tp in enumerate(transcripts):
            call_id = tp.stem
            try:
                result = analyze_one(client, tp, config, args.model)
            except Exception as e:
                result = {"success": False, "call_id": call_id, "elapsed": 0, "error": str(e)}

            results.append(result)

            if result["success"]:
                out_path = output_dir / f"{call_id}.json"
                with open(out_path, "w") as f:
                    json.dump(result["record"], f, indent=2)

                scope = result["primary_scope"]
                outcome = result["primary_outcome"]
                secondary = " +2nd" if result["has_secondary"] else ""
                failure_tag = f" FAIL:{result['failure']}" if result["failure"] else ""
                print(f"  [{i+1}/{total}] {call_id}  OK ({result['elapsed']:.1f}s) [{scope}:{outcome}]{secondary}{failure_tag}")
            else:
                print(f"  [{i+1}/{total}] {call_id}  FAILED: {result.get('error', '?')[:80]}")

            if i < total - 1:
                time.sleep(args.delay)
    else:
        # Parallel mode
        task_queue = queue.Queue()
        for i, tp in enumerate(transcripts):
            task_queue.put((i, tp))

        results = []
        result_lock = threading.Lock()
        total_t0 = time.time()

        threads = []
        for _ in range(args.workers):
            t = threading.Thread(target=_worker,
                                 args=(task_queue, results, result_lock, output_dir,
                                       config, args.model, total, api_key))
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


if __name__ == "__main__":
    main()
