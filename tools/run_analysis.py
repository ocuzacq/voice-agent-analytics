#!/usr/bin/env python3
"""
End-to-End Analysis Pipeline for Vacatia AI Voice Agent Analytics (v3.5.5)

Orchestrates the complete analysis workflow:
1. sample_transcripts.py → Sample transcripts from corpus
2. batch_analyze.py → Analyze transcripts with LLM (v3 schema, parallel in v3.2)
3. compute_metrics.py → Calculate Section A deterministic metrics
4. extract_nl_fields.py → Extract condensed NL data for LLM (v3.1)
5. generate_insights.py → Generate Section B LLM insights
6. render_report.py → Render Markdown executive summary
7. review_report.py → Editorial review and pipeline suggestions (v3.5.5)

v3.5.5 Features:
- Report review pass with Gemini 3 Pro for editorial refinement
- Pipeline improvement suggestions generated after each run
- --skip-review to bypass review step
- --no-suggestions to skip pipeline suggestions

v3.2 Features:
- Configurable parallelization (default 3 workers) for batch analysis
- Run isolation: sampled/ cleared by default between runs
- Scope coherence: batch_analyze respects manifest.csv
- Resume capability: --resume to continue from existing samples

Usage:
    # Full pipeline with 50 transcripts (3 parallel workers)
    python3 tools/run_analysis.py

    # Quick test with 5 transcripts
    python3 tools/run_analysis.py --quick

    # Custom sample size with more parallelization
    python3 tools/run_analysis.py -n 200 --workers 5

    # Skip sampling (use existing sampled directory)
    python3 tools/run_analysis.py --skip-sampling

    # Resume interrupted run (uses existing manifest)
    python3 tools/run_analysis.py --resume

    # Skip analysis (use existing analyses)
    python3 tools/run_analysis.py --skip-analysis
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def run_step(name: str, cmd: list[str], cwd: Path | None = None) -> bool:
    """Run a pipeline step and return success status."""
    print(f"\n{'='*60}")
    print(f"STEP: {name}")
    print(f"{'='*60}")
    print(f"Command: {' '.join(cmd)}")
    print()

    result = subprocess.run(cmd, cwd=cwd)

    if result.returncode != 0:
        print(f"\n❌ FAILED: {name} (exit code {result.returncode})")
        return False

    print(f"\n✅ COMPLETED: {name}")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end analysis pipeline (v3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with 50 transcripts
    python3 tools/run_analysis.py

    # Quick test with 5 transcripts
    python3 tools/run_analysis.py --quick

    # Custom sample size
    python3 tools/run_analysis.py -n 100

    # Skip sampling (use existing sampled directory)
    python3 tools/run_analysis.py --skip-sampling

    # Skip analysis (use existing analyses)
    python3 tools/run_analysis.py --skip-analysis
        """
    )

    parser.add_argument("-n", "--sample-size", type=int, default=50,
                        help="Number of transcripts to sample (default: 50)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with 5 transcripts")
    parser.add_argument("--skip-sampling", action="store_true",
                        help="Skip sampling step (use existing sampled/)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume from existing sampled/ (skip sampling, respect manifest)")
    parser.add_argument("--no-clear", action="store_true",
                        help="Don't clear sampled/ before new sampling (append mode)")
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip analysis step (use existing analyses/)")
    parser.add_argument("--no-clear-analyses", action="store_true",
                        help="Don't clear analyses/ before fresh run (default: clear for run isolation)")
    parser.add_argument("--skip-insights", action="store_true",
                        help="Skip LLM insights generation (metrics only)")
    parser.add_argument("--analysis-model", type=str, default="gemini-3-flash-preview",
                        help="Gemini model for transcript analysis (default: gemini-3-flash-preview)")
    parser.add_argument("--insights-model", type=str, default="gemini-3-pro-preview",
                        help="Gemini model for insights generation (default: gemini-3-pro-preview)")
    parser.add_argument("-w", "--workers", type=int, default=3,
                        help="Number of parallel workers for batch analysis (default: 3)")
    parser.add_argument("--rate-limit", type=float, default=1.0,
                        help="Delay between API calls per worker in seconds")
    parser.add_argument("--seed", type=int,
                        help="Random seed for reproducible sampling")
    parser.add_argument("--transcripts-dir", type=Path,
                        help="Custom transcripts directory")
    parser.add_argument("--output-dir", type=Path,
                        help="Custom output directory for all outputs")

    # v3.5.5: Report review options
    parser.add_argument("--skip-review", action="store_true",
                        help="Skip report review step (v3.5.5)")
    parser.add_argument("--review-model", type=str, default="gemini-3-pro-preview",
                        help="Gemini model for report review (default: gemini-3-pro-preview)")
    parser.add_argument("--no-suggestions", action="store_true",
                        help="Skip pipeline suggestions in review (review only)")

    args = parser.parse_args()

    # Determine paths
    tools_dir = Path(__file__).parent
    project_dir = tools_dir.parent

    transcripts_dir = args.transcripts_dir or project_dir / "transcripts"
    sampled_dir = project_dir / "sampled"
    analyses_dir = project_dir / "analyses"
    reports_dir = args.output_dir or project_dir / "reports"

    # Quick mode
    if args.quick:
        args.sample_size = 5

    # Resume mode: skip sampling and use existing manifest
    if args.resume:
        args.skip_sampling = True
        manifest_path = sampled_dir / "manifest.csv"
        if not manifest_path.exists():
            print(f"Error: Cannot resume - no manifest.csv found in {sampled_dir}")
            print("Run without --resume to create a new sample.")
            return 1
        # Count files in manifest for display
        import csv
        with open(manifest_path) as f:
            manifest_count = sum(1 for _ in csv.reader(f)) - 1  # minus header
        print(f"Resuming from existing manifest ({manifest_count} files)")

    # Generate run ID for logging
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 60)
    print("VACATIA AI VOICE AGENT ANALYTICS - v3.5.5 PIPELINE")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"\nConfiguration:")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Analysis model: {args.analysis_model}")
    print(f"  Insights model: {args.insights_model}")
    print(f"  Review model: {args.review_model}{' (skipped)' if args.skip_review else ''}")
    print(f"  Workers: {args.workers}")
    print(f"  Rate limit: {args.rate_limit}s per worker")
    print(f"  Transcripts: {transcripts_dir}")
    print(f"  Output: {reports_dir}")
    if args.resume:
        print(f"  Mode: RESUME (using existing manifest)")
    elif args.no_clear:
        print(f"  Mode: APPEND (keeping existing samples)")

    steps_completed = []
    steps_failed = []

    # Step 1: Sample transcripts
    if not args.skip_sampling:
        cmd = [
            sys.executable,
            str(tools_dir / "sample_transcripts.py"),
            "-n", str(args.sample_size),
            "-i", str(transcripts_dir),
            "-o", str(sampled_dir)
        ]
        if args.seed:
            cmd.extend(["--seed", str(args.seed)])
        if args.no_clear:
            cmd.append("--no-clear")

        step_name = "Sample Transcripts" + (" (append mode)" if args.no_clear else " (clearing existing)")
        if run_step(step_name, cmd):
            steps_completed.append("sampling")
        else:
            steps_failed.append("sampling")
            print("\n⚠️ Sampling failed. Attempting to continue with existing samples...")
    else:
        skip_reason = "--resume" if args.resume else "--skip-sampling"
        print(f"\n⏭️ Skipping sampling ({skip_reason})")
        steps_completed.append("sampling (skipped)")

    # Step 2: Batch analyze (parallel in v3.2)
    if not args.skip_analysis:
        # v3.3: Clear analyses/ for run isolation (unless --resume or --no-clear-analyses)
        if not args.resume and not args.no_clear_analyses:
            existing_analyses = list(analyses_dir.glob("*.json"))
            if existing_analyses:
                print(f"\n🧹 Clearing {len(existing_analyses)} existing analyses for run isolation...")
                for f in existing_analyses:
                    f.unlink()
                print("   (use --no-clear-analyses to preserve existing analyses)")

        cmd = [
            sys.executable,
            str(tools_dir / "batch_analyze.py"),
            "-i", str(sampled_dir),
            "-o", str(analyses_dir),
            "--model", args.analysis_model,
            "--workers", str(args.workers),
            "--rate-limit", str(args.rate_limit)
        ]

        if run_step(f"Batch Analyze Transcripts (v3.3, {args.workers} workers)", cmd):
            steps_completed.append("analysis")
        else:
            steps_failed.append("analysis")
            if not args.skip_insights:
                print("\n❌ Analysis failed. Cannot continue.")
                return 1
    else:
        print("\n⏭️ Skipping analysis (--skip-analysis)")
        steps_completed.append("analysis (skipped)")

    # Step 3: Compute metrics (v3.3: scope coherence via manifest)
    cmd = [
        sys.executable,
        str(tools_dir / "compute_metrics.py"),
        "-i", str(analyses_dir),
        "-o", str(reports_dir),
        "-s", str(sampled_dir)  # v3.3: Pass sampled dir for manifest filtering
    ]

    if run_step("Compute Deterministic Metrics (Section A)", cmd):
        steps_completed.append("metrics")
    else:
        steps_failed.append("metrics")
        print("\n❌ Metrics computation failed. Cannot continue.")
        return 1

    # Step 4: Extract NL fields (v3.3: scope coherence via manifest)
    cmd = [
        sys.executable,
        str(tools_dir / "extract_nl_fields.py"),
        "-i", str(analyses_dir),
        "-o", str(reports_dir),
        "-s", str(sampled_dir)  # v3.3: Pass sampled dir for manifest filtering
    ]

    if run_step("Extract NL Fields for LLM (v3.3)", cmd):
        steps_completed.append("nl_extraction")
    else:
        steps_failed.append("nl_extraction")
        print("\n⚠️ NL extraction failed. Insights will be based on metrics only.")

    # Step 5: Generate insights (v3.3: uses gemini-3-pro-preview by default)
    if not args.skip_insights:
        cmd = [
            sys.executable,
            str(tools_dir / "generate_insights.py"),
            "-o", str(reports_dir),
            "--model", args.insights_model
        ]

        if run_step("Generate LLM Insights (Section B)", cmd):
            steps_completed.append("insights")
        else:
            steps_failed.append("insights")
            print("\n⚠️ Insights generation failed. Continuing without insights...")
    else:
        print("\n⏭️ Skipping insights generation (--skip-insights)")
        steps_completed.append("insights (skipped)")

    # Step 6: Render report
    if "insights" in steps_completed:
        cmd = [
            sys.executable,
            str(tools_dir / "render_report.py"),
            "-o", str(reports_dir)
        ]

        if run_step("Render Markdown Report", cmd):
            steps_completed.append("report")
        else:
            steps_failed.append("report")
            print("\n⚠️ Report rendering failed.")
    else:
        print("\n⏭️ Skipping report rendering (no insights available)")

    # Step 7: Review and refine report (v3.5.5)
    if not args.skip_review and "report" in steps_completed:
        cmd = [
            sys.executable,
            str(tools_dir / "review_report.py"),
            "-o", str(reports_dir),
            "--model", args.review_model
        ]

        if args.no_suggestions:
            cmd.append("--no-suggestions")

        if run_step("Review & Refine Report (v3.5.5)", cmd):
            steps_completed.append("review")
        else:
            steps_failed.append("review")
            print("\n⚠️ Report review failed. Original report still available.")
    elif args.skip_review:
        print("\n⏭️ Skipping report review (--skip-review)")
        steps_completed.append("review (skipped)")
    elif "report" not in steps_completed:
        print("\n⏭️ Skipping report review (no report available)")

    # Summary
    print("\n" + "=" * 60)
    print("PIPELINE SUMMARY")
    print("=" * 60)
    print(f"Completed: {datetime.now().isoformat()}")
    print(f"\n✅ Steps completed: {', '.join(steps_completed)}")
    if steps_failed:
        print(f"❌ Steps failed: {', '.join(steps_failed)}")

    # List output files
    print(f"\n📁 Output directory: {reports_dir}")
    if reports_dir.exists():
        print("\nGenerated files:")
        for f in sorted(reports_dir.glob("*_v3_*")):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  - {f.name} ({size_kb:.1f} KB)")

    return 0 if not steps_failed else 1


if __name__ == "__main__":
    exit(main())
