#!/usr/bin/env python3
"""
End-to-End Analysis Pipeline for Vacatia AI Voice Agent Analytics (v4.3)

Orchestrates the complete analysis workflow with run-based isolation:
1. sample_transcripts.py → Sample transcripts from corpus
2. batch_analyze.py → Analyze transcripts with LLM (v3 schema, parallel in v3.2)
3. compute_metrics.py → Calculate Section A deterministic metrics
4. extract_nl_fields.py → Extract condensed NL data for LLM (v3.1)
5. generate_insights.py → Generate Section B LLM insights
6. render_report.py → Render Markdown executive summary
7. review_report.py → Editorial review and pipeline suggestions (v3.5.5) [optional]

v4.3 Features:
- Target-based augmentation: -n becomes a TARGET when run exists
- Day 1: -n 50 creates run with 50 transcripts
- Day 2: -n 200 grows to 200 total (adds 150)
- Day 3: -n 1000 grows to 1000 total (adds 800)
- System calculates delta automatically

v4.1 Features:
- Run-based isolation: Each run creates an isolated directory in runs/<run_id>/
- All outputs (sampled/, analyses/, reports/) contained within run directory
- config.json and status.json for reproducibility and progress tracking
- Symlink runs/latest points to most recent run
- --legacy flag to use flat directory mode (backwards compatible)
- --run-id for custom run naming

v3.9.2 Features:
- Lazy metrics/NL extraction: --skip-insights now stops after analysis step
- Enables fast workflow for ask.py (no metrics/NL files needed)
- Full pipeline auto-generates metrics/NL before insights as before

v3.9.1 Features:
- Custom questions: --questions flag to provide questions file for LLM to answer
- Questions answered in dedicated "Custom Analysis" section after Executive Summary
- Review step disabled by default (use --enable-review to run it)

Usage:
    # Full pipeline with 50 transcripts (creates isolated run)
    python3 tools/run_analysis.py -n 50

    # Quick test with 5 transcripts
    python3 tools/run_analysis.py --quick

    # Grow existing run to 200 total (system calculates delta)
    python3 tools/run_analysis.py -n 200

    # Use specific run directory
    python3 tools/run_analysis.py -n 300 --run-dir runs/run_xxx

    # Custom run ID for new run
    python3 tools/run_analysis.py -n 50 --run-id my_experiment

    # Legacy mode (flat directories, no isolation)
    python3 tools/run_analysis.py --legacy
"""

import argparse
import csv
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from run_utils import (
    create_run_directory,
    save_config,
    update_status,
    update_latest_symlink,
    get_run_paths,
    count_transcript_files,
    load_last_run,
)


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
        description="End-to-end analysis pipeline (v4.3 - target-based augmentation)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Full pipeline with 50 transcripts (creates isolated run)
    python3 tools/run_analysis.py -n 50

    # Quick test with 5 transcripts
    python3 tools/run_analysis.py --quick

    # Grow existing run to target size (system calculates delta)
    python3 tools/run_analysis.py -n 200    # has 50 → adds 150

    # Use specific run directory
    python3 tools/run_analysis.py -n 300 --run-dir runs/run_xxx

    # Custom run ID for new run
    python3 tools/run_analysis.py -n 50 --run-id experiment_1

    # Legacy mode (flat directories)
    python3 tools/run_analysis.py --legacy
        """
    )

    parser.add_argument("-n", "--sample-size", type=int, default=50,
                        help="Target number of transcripts (default: 50). For existing runs, this is the target total.")
    parser.add_argument("--quick", action="store_true",
                        help="Quick test with 5 transcripts")

    # Run isolation arguments
    parser.add_argument("--run-dir", type=Path,
                        help="Use existing run directory (resume or re-run)")
    parser.add_argument("--run-id", type=str,
                        help="Custom run ID (default: timestamp)")
    parser.add_argument("--legacy", action="store_true",
                        help="Use legacy flat directory mode (no isolation)")

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
                        help="Skip metrics, NL extraction, and insights (stop after analysis)")
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
                        help="Custom output directory for all outputs (legacy mode only)")

    # v3.5.5: Report review options (disabled by default in v3.9.1)
    parser.add_argument("--enable-review", action="store_true",
                        help="Enable report review step (disabled by default)")
    parser.add_argument("--skip-review", action="store_true",
                        help="[DEPRECATED] Skip report review step (now default behavior)")
    parser.add_argument("--review-model", type=str, default="gemini-3-pro-preview",
                        help="Gemini model for report review (default: gemini-3-pro-preview)")
    parser.add_argument("--no-suggestions", action="store_true",
                        help="Skip pipeline suggestions in review (review only)")

    # v3.9.1: Custom questions
    parser.add_argument("--questions", type=Path,
                        help="Path to questions file (one question per line) for custom analysis")

    args = parser.parse_args()

    # Determine paths
    tools_dir = Path(__file__).parent
    project_dir = tools_dir.parent

    transcripts_dir = args.transcripts_dir or project_dir / "transcripts"

    # Quick mode
    if args.quick:
        args.sample_size = 5

    # Determine run mode: isolated vs legacy
    run_dir = None
    using_isolation = not args.legacy

    if args.run_dir:
        # Use existing run directory
        run_dir = args.run_dir
        if not run_dir.exists():
            print(f"Error: Run directory not found: {run_dir}")
            return 1
        using_isolation = True
        print(f"Using existing run: {run_dir}")
    elif not args.legacy:
        # Check for existing run via .last_run
        run_id = load_last_run(project_dir)
        if run_id:
            runs_dir = project_dir / "runs"
            potential_run_dir = runs_dir / run_id
            if potential_run_dir.exists():
                run_dir = potential_run_dir
                using_isolation = True
                print(f"Found existing run: {run_dir}")

    if args.legacy:
        # Legacy flat directory mode
        using_isolation = False
        print("Using legacy flat directory mode")
    elif run_dir is None:
        # No existing run found, create new isolated run
        runs_dir = project_dir / "runs"
        run_dir, run_id_created = create_run_directory(runs_dir, args.run_id)
        using_isolation = True
        print(f"Creating isolated run: {run_dir}")
        print(f"  (saved to .last_run - all tools will use this automatically)")

    # Get paths based on mode
    if using_isolation:
        paths = get_run_paths(run_dir, project_dir)
        sampled_dir = paths["sampled_dir"]
        analyses_dir = paths["analyses_dir"]
        reports_dir = paths["reports_dir"]
    else:
        sampled_dir = project_dir / "sampled"
        analyses_dir = project_dir / "analyses"
        reports_dir = args.output_dir or project_dir / "reports"

    # v4.3: Target-based augmentation - calculate delta for existing runs
    existing_count = 0
    target = args.sample_size
    is_augmenting = False
    manifest_path = sampled_dir / "manifest.csv"

    if manifest_path.exists() and not args.skip_sampling:
        with open(manifest_path) as f:
            existing_count = sum(1 for _ in csv.reader(f)) - 1  # minus header

        if existing_count > 0:
            delta = target - existing_count
            if delta <= 0:
                print(f"\nRun already has {existing_count} transcripts (target: {target}). Nothing to add.")
                args.skip_sampling = True
            else:
                print(f"\nRun has {existing_count} transcripts. Adding {delta} to reach target of {target}.")
                args.sample_size = delta  # Sample only the delta
                args.no_clear = True      # Append mode
                is_augmenting = True

    # Resume mode: skip sampling and use existing manifest
    if args.resume:
        args.skip_sampling = True
        if not manifest_path.exists():
            print(f"Error: Cannot resume - no manifest.csv found in {sampled_dir}")
            print("Run without --resume to create a new sample.")
            return 1
        # Count files in manifest for display
        with open(manifest_path) as f:
            manifest_count = sum(1 for _ in csv.reader(f)) - 1  # minus header
        print(f"Resuming from existing manifest ({manifest_count} files)")

    # Save configuration for reproducibility (isolated mode only)
    if using_isolation and run_dir and not args.run_dir:
        # Only save config for NEW runs, not when using existing run_dir
        total_available = count_transcript_files(transcripts_dir)
        save_config(run_dir, args, extra={
            "source": {
                "transcripts_dir": str(transcripts_dir),
                "total_available": total_available
            }
        })

    # Generate run ID for logging
    run_id = run_dir.name if run_dir else datetime.now().strftime("%Y%m%d_%H%M%S")

    # v3.9.1: Review disabled by default
    run_review = args.enable_review and not args.skip_review

    print("=" * 60)
    print("VACATIA AI VOICE AGENT ANALYTICS - v4.3 PIPELINE")
    print("=" * 60)
    print(f"Run ID: {run_id}")
    print(f"Started: {datetime.now().isoformat()}")
    print(f"\nConfiguration:")
    print(f"  Mode: {'ISOLATED' if using_isolation else 'LEGACY'}")
    if using_isolation:
        print(f"  Run directory: {run_dir}")
    if is_augmenting:
        print(f"  Target: {target} transcripts ({existing_count} existing + {args.sample_size} new)")
    else:
        print(f"  Sample size: {args.sample_size}")
    print(f"  Analysis model: {args.analysis_model}")
    print(f"  Insights model: {args.insights_model}")
    print(f"  Review: {'enabled' if run_review else 'disabled'}{' (use --enable-review to enable)' if not run_review else ''}")
    print(f"  Workers: {args.workers}")
    print(f"  Rate limit: {args.rate_limit}s per worker")
    print(f"  Transcripts: {transcripts_dir}")
    if not using_isolation:
        print(f"  Output: {reports_dir}")
    if args.questions:
        print(f"  Custom questions: {args.questions}")
    if args.resume:
        print(f"  Mode: RESUME (using existing manifest)")
    elif args.no_clear and not is_augmenting:
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
        if using_isolation and run_dir:
            cmd.extend(["--run-dir", str(run_dir)])

        if is_augmenting:
            step_name = f"Augment Run (+{args.sample_size} → {target} total)"
        else:
            step_name = "Sample Transcripts" + (" (append mode)" if args.no_clear else "")
        if run_step(step_name, cmd):
            steps_completed.append("sampling")
            if using_isolation and run_dir:
                update_status(run_dir, "sampling", "completed", count=args.sample_size)
        else:
            steps_failed.append("sampling")
            if using_isolation and run_dir:
                update_status(run_dir, "sampling", "failed")
            print("\n⚠️ Sampling failed. Attempting to continue with existing samples...")
    else:
        skip_reason = "--resume" if args.resume else "--skip-sampling"
        print(f"\n⏭️ Skipping sampling ({skip_reason})")
        steps_completed.append("sampling (skipped)")

    # Step 2: Batch analyze (parallel in v3.2)
    if not args.skip_analysis:
        # v3.3: Clear analyses/ for run isolation (unless --resume or --no-clear-analyses)
        # In isolated mode, directory is fresh so no need to clear
        if not using_isolation and not args.resume and not args.no_clear_analyses:
            existing_analyses = list(analyses_dir.glob("*.json"))
            if existing_analyses:
                print(f"\n🧹 Clearing {len(existing_analyses)} existing analyses for run isolation...")
                for f in existing_analyses:
                    f.unlink()
                print("   (use --no-clear-analyses to preserve existing analyses)")

        if using_isolation and run_dir:
            update_status(run_dir, "analysis", "in_progress")

        cmd = [
            sys.executable,
            str(tools_dir / "batch_analyze.py"),
            "-i", str(sampled_dir),
            "-o", str(analyses_dir),
            "--model", args.analysis_model,
            "--workers", str(args.workers),
            "--rate-limit", str(args.rate_limit)
        ]
        if using_isolation and run_dir:
            cmd.extend(["--run-dir", str(run_dir)])

        if run_step(f"Batch Analyze Transcripts (v3.3, {args.workers} workers)", cmd):
            steps_completed.append("analysis")
            if using_isolation and run_dir:
                # Count successful analyses
                success_count = len(list(analyses_dir.glob("*.json")))
                update_status(run_dir, "analysis", "completed", count=success_count)
        else:
            steps_failed.append("analysis")
            if using_isolation and run_dir:
                update_status(run_dir, "analysis", "failed")
            print("\n" + "=" * 60)
            print("❌ PIPELINE STOPPED: Analysis step failed")
            print("=" * 60)
            print("\nThe batch analysis step is required for all downstream steps.")
            print("Common causes:")
            print("  • Missing API key: Set GOOGLE_API_KEY in .env file")
            print("  • Invalid API key: Check your Gemini API credentials")
            print("  • Network issues: Check your internet connection")
            print("\nTo retry, run the same command again.")
            return 1
    else:
        print("\n⏭️ Skipping analysis (--skip-analysis)")
        steps_completed.append("analysis (skipped)")

    # Steps 3-5: Metrics, NL extraction, and insights (skipped with --skip-insights)
    if not args.skip_insights:
        # Step 3: Compute metrics (v3.3: scope coherence via manifest)
        if using_isolation and run_dir:
            update_status(run_dir, "metrics", "in_progress")

        cmd = [
            sys.executable,
            str(tools_dir / "compute_metrics.py"),
            "-i", str(analyses_dir),
            "-o", str(reports_dir),
            "-s", str(sampled_dir)  # v3.3: Pass sampled dir for manifest filtering
        ]
        if using_isolation and run_dir:
            cmd.extend(["--run-dir", str(run_dir)])

        if run_step("Compute Deterministic Metrics (Section A)", cmd):
            steps_completed.append("metrics")
            if using_isolation and run_dir:
                update_status(run_dir, "metrics", "completed")
        else:
            steps_failed.append("metrics")
            if using_isolation and run_dir:
                update_status(run_dir, "metrics", "failed")
            print("\n❌ Metrics computation failed. Cannot continue.")
            return 1

        # Step 4: Extract NL fields (v3.3: scope coherence via manifest)
        if using_isolation and run_dir:
            update_status(run_dir, "nl_extraction", "in_progress")

        cmd = [
            sys.executable,
            str(tools_dir / "extract_nl_fields.py"),
            "-i", str(analyses_dir),
            "-o", str(reports_dir),
            "-s", str(sampled_dir)  # v3.3: Pass sampled dir for manifest filtering
        ]
        if using_isolation and run_dir:
            cmd.extend(["--run-dir", str(run_dir)])

        if run_step("Extract NL Fields for LLM (v3.3)", cmd):
            steps_completed.append("nl_extraction")
            if using_isolation and run_dir:
                update_status(run_dir, "nl_extraction", "completed")
        else:
            steps_failed.append("nl_extraction")
            if using_isolation and run_dir:
                update_status(run_dir, "nl_extraction", "failed")
            print("\n⚠️ NL extraction failed. Insights will be based on metrics only.")

        # Step 5: Generate insights (v3.3: uses gemini-3-pro-preview by default)
        if using_isolation and run_dir:
            update_status(run_dir, "insights", "in_progress")

        cmd = [
            sys.executable,
            str(tools_dir / "generate_insights.py"),
            "-o", str(reports_dir),
            "--model", args.insights_model
        ]
        if using_isolation and run_dir:
            cmd.extend(["--run-dir", str(run_dir)])

        # v3.9.1: Pass custom questions file if provided
        if args.questions:
            if not args.questions.exists():
                print(f"\n⚠️ Questions file not found: {args.questions}")
            else:
                cmd.extend(["--questions", str(args.questions)])

        step_name = "Generate LLM Insights (Section B)"
        if args.questions and args.questions.exists():
            step_name += " + Custom Questions"

        if run_step(step_name, cmd):
            steps_completed.append("insights")
            if using_isolation and run_dir:
                update_status(run_dir, "insights", "completed")
        else:
            steps_failed.append("insights")
            if using_isolation and run_dir:
                update_status(run_dir, "insights", "failed")
            print("\n⚠️ Insights generation failed. Continuing without insights...")
    else:
        print("\n⏭️ Skipping metrics, NL extraction, insights (--skip-insights)")
        steps_completed.append("metrics (skipped)")
        steps_completed.append("nl_extraction (skipped)")
        steps_completed.append("insights (skipped)")

    # Step 6: Render report
    if "insights" in steps_completed:
        if using_isolation and run_dir:
            update_status(run_dir, "report", "in_progress")

        cmd = [
            sys.executable,
            str(tools_dir / "render_report.py"),
            "-o", str(reports_dir)
        ]
        if using_isolation and run_dir:
            cmd.extend(["--run-dir", str(run_dir)])

        if run_step("Render Markdown Report", cmd):
            steps_completed.append("report")
            if using_isolation and run_dir:
                update_status(run_dir, "report", "completed")
        else:
            steps_failed.append("report")
            if using_isolation and run_dir:
                update_status(run_dir, "report", "failed")
            print("\n⚠️ Report rendering failed.")
    else:
        print("\n⏭️ Skipping report rendering (no insights available)")

    # Step 7: Review and refine report (v3.5.5) - disabled by default in v3.9.1
    if run_review and "report" in steps_completed:
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
    elif not run_review:
        print("\n⏭️ Skipping report review (disabled by default, use --enable-review)")
        steps_completed.append("review (skipped)")
    elif "report" not in steps_completed:
        print("\n⏭️ Skipping report review (no report available)")

    # Update latest symlink for isolated runs
    if using_isolation and run_dir:
        update_latest_symlink(run_dir)

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
        for f in sorted(reports_dir.glob("*_v*_*")):
            if f.is_file():
                size_kb = f.stat().st_size / 1024
                print(f"  - {f.name} ({size_kb:.1f} KB)")

    # Show run directory info for isolated runs
    if using_isolation and run_dir:
        print(f"\n📂 Run directory: {run_dir}")
        print(f"   Latest symlink: {run_dir.parent / 'latest'}")

    return 0 if not steps_failed else 1


if __name__ == "__main__":
    exit(main())
