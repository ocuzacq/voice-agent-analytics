#!/usr/bin/env python3
"""
End-to-End Analysis Pipeline for Vacatia AI Voice Agent Analytics (v3.2)

Orchestrates the complete analysis workflow:
1. sample_transcripts.py → Sample transcripts from corpus
2. batch_analyze.py → Analyze transcripts with LLM (v3 schema, parallel in v3.2)
3. compute_metrics.py → Calculate Section A deterministic metrics
4. extract_nl_fields.py → Extract condensed NL data for LLM (v3.1)
5. generate_insights.py → Generate Section B LLM insights
6. render_report.py → Render Markdown executive summary

v3.2: Added configurable parallelization (default 3 workers) for batch analysis.

Usage:
    # Full pipeline with 50 transcripts (3 parallel workers)
    python3 tools/run_analysis.py

    # Quick test with 5 transcripts
    python3 tools/run_analysis.py --quick

    # Custom sample size with more parallelization
    python3 tools/run_analysis.py -n 200 --workers 5

    # Skip sampling (use existing sampled directory)
    python3 tools/run_analysis.py --skip-sampling

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
    parser.add_argument("--skip-analysis", action="store_true",
                        help="Skip analysis step (use existing analyses/)")
    parser.add_argument("--skip-insights", action="store_true",
                        help="Skip LLM insights generation (metrics only)")
    parser.add_argument("--model", type=str, default="gemini-2.5-flash",
                        help="Gemini model to use")
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

    print("=" * 60)
    print("VACATIA AI VOICE AGENT ANALYTICS - v3.2 PIPELINE")
    print("=" * 60)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"\nConfiguration:")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Model: {args.model}")
    print(f"  Workers: {args.workers}")
    print(f"  Rate limit: {args.rate_limit}s per worker")
    print(f"  Transcripts: {transcripts_dir}")
    print(f"  Output: {reports_dir}")

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

        if run_step("Sample Transcripts", cmd):
            steps_completed.append("sampling")
        else:
            steps_failed.append("sampling")
            print("\n⚠️ Sampling failed. Attempting to continue with existing samples...")
    else:
        print("\n⏭️ Skipping sampling (--skip-sampling)")
        steps_completed.append("sampling (skipped)")

    # Step 2: Batch analyze (parallel in v3.2)
    if not args.skip_analysis:
        cmd = [
            sys.executable,
            str(tools_dir / "batch_analyze.py"),
            "-i", str(sampled_dir),
            "-o", str(analyses_dir),
            "--model", args.model,
            "--workers", str(args.workers),
            "--rate-limit", str(args.rate_limit)
        ]

        if run_step(f"Batch Analyze Transcripts (v3.2, {args.workers} workers)", cmd):
            steps_completed.append("analysis")
        else:
            steps_failed.append("analysis")
            if not args.skip_insights:
                print("\n❌ Analysis failed. Cannot continue.")
                return 1
    else:
        print("\n⏭️ Skipping analysis (--skip-analysis)")
        steps_completed.append("analysis (skipped)")

    # Step 3: Compute metrics
    cmd = [
        sys.executable,
        str(tools_dir / "compute_metrics.py"),
        "-i", str(analyses_dir),
        "-o", str(reports_dir)
    ]

    if run_step("Compute Deterministic Metrics (Section A)", cmd):
        steps_completed.append("metrics")
    else:
        steps_failed.append("metrics")
        print("\n❌ Metrics computation failed. Cannot continue.")
        return 1

    # Step 4: Extract NL fields (v3.1 - dedicated extraction step)
    cmd = [
        sys.executable,
        str(tools_dir / "extract_nl_fields.py"),
        "-i", str(analyses_dir),
        "-o", str(reports_dir)
    ]

    if run_step("Extract NL Fields for LLM (v3.1)", cmd):
        steps_completed.append("nl_extraction")
    else:
        steps_failed.append("nl_extraction")
        print("\n⚠️ NL extraction failed. Insights will be based on metrics only.")

    # Step 5: Generate insights
    if not args.skip_insights:
        cmd = [
            sys.executable,
            str(tools_dir / "generate_insights.py"),
            "-o", str(reports_dir),
            "--model", args.model
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
