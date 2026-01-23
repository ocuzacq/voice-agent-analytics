#!/usr/bin/env python3
"""
Run-Based Isolation Utilities for Voice Agent Analytics (v4.1)

Provides shared utilities for run-based directory isolation. Each analysis run
is self-contained in `runs/<run_id>/` with its own config, status, and outputs.

Usage:
    from run_utils import get_run_paths, add_run_dir_argument, create_run_directory

Directory Structure:
    runs/
    ├── run_20260121_143000/
    │   ├── config.json           # Run parameters (reproducibility)
    │   ├── status.json           # Pipeline progress tracking
    │   ├── manifest.csv          # Sample scope anchor
    │   ├── sampled/              # Transcript copies
    │   ├── analyses/             # Analysis JSONs
    │   └── reports/              # Metrics, insights, markdown
    └── latest -> run_20260121_143000  # Symlink to most recent
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def get_tty_suffix() -> str:
    """Get a TTY-based suffix for per-terminal isolation.

    Returns a sanitized TTY identifier (e.g., "ttys003" from "/dev/ttys003")
    or empty string if no TTY is available (non-interactive context).

    Returns:
        str: TTY suffix like "ttys003" or "" for non-interactive
    """
    try:
        # Try to get the TTY name from stdin
        if sys.stdin.isatty():
            tty_path = os.ttyname(sys.stdin.fileno())
            # Extract just the tty name: /dev/ttys003 -> ttys003
            return tty_path.split('/')[-1]
    except (OSError, AttributeError):
        pass
    return ""


def get_run_paths(run_dir: Optional[Path], project_dir: Path) -> dict:
    """Get all paths for a run (isolated or legacy mode).

    Args:
        run_dir: Run directory (if using run-based isolation) or None (legacy mode)
        project_dir: Project root directory

    Returns:
        dict with keys: sampled_dir, analyses_dir, reports_dir, manifest_path, run_dir

    Example:
        >>> paths = get_run_paths(Path("runs/run_20260121"), project_dir)
        >>> paths['sampled_dir']  # Path("runs/run_20260121/sampled")
        >>> paths['analyses_dir']  # Path("runs/run_20260121/analyses")
    """
    if run_dir is not None:
        # Run-based isolation: all paths under run directory
        return {
            "run_dir": run_dir,
            "sampled_dir": run_dir / "sampled",
            "analyses_dir": run_dir / "analyses",
            "reports_dir": run_dir / "reports",
            "manifest_path": run_dir / "manifest.csv",
            "config_path": run_dir / "config.json",
            "status_path": run_dir / "status.json",
        }
    else:
        # Legacy mode: flat directories in project root
        return {
            "run_dir": None,
            "sampled_dir": project_dir / "sampled",
            "analyses_dir": project_dir / "analyses",
            "reports_dir": project_dir / "reports",
            "manifest_path": project_dir / "sampled" / "manifest.csv",
            "config_path": None,
            "status_path": None,
        }


def add_run_dir_argument(parser: argparse.ArgumentParser) -> None:
    """Add --run-dir argument to an argument parser.

    Adds the standard --run-dir argument used by all pipeline tools
    for run-based isolation.

    Args:
        parser: ArgumentParser to add the argument to
    """
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Run directory for isolated execution (overrides -i/-o/-s paths)"
    )


def add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Add both --run-dir and --run-id arguments to an argument parser.

    Use this instead of add_run_dir_argument when you want tools to be able
    to create new runs (not just use existing ones).

    Args:
        parser: ArgumentParser to add the arguments to
    """
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Use existing run directory (overrides -i/-o/-s paths)"
    )
    parser.add_argument(
        "--run-id",
        type=str,
        default=None,
        help="Create/use run with this ID (creates runs/<run-id>/ if needed)"
    )


def get_last_run_path(project_dir: Path) -> Path:
    """Get the .last_run file path for the current terminal.

    Uses TTY-based suffix for per-terminal isolation:
    - Terminal with TTY: .last_run.ttys003
    - Non-interactive: .last_run (global fallback)

    Args:
        project_dir: Project root directory

    Returns:
        Path to the .last_run file for this terminal
    """
    tty_suffix = get_tty_suffix()
    if tty_suffix:
        return project_dir / f".last_run.{tty_suffix}"
    return project_dir / ".last_run"


def save_last_run(project_dir: Path, run_id: str) -> None:
    """Save run_id to .last_run file for automatic reuse.

    Uses TTY-based file naming for per-terminal isolation.
    """
    last_run_file = get_last_run_path(project_dir)
    last_run_file.write_text(run_id + "\n")


def load_last_run(project_dir: Path) -> str | None:
    """Load run_id from .last_run file if it exists.

    Checks TTY-specific file first, then falls back to global .last_run.
    """
    # First try TTY-specific file
    last_run_file = get_last_run_path(project_dir)
    if last_run_file.exists():
        run_id = last_run_file.read_text().strip()
        if run_id:
            return run_id

    # Fallback to global .last_run (for backwards compatibility)
    global_file = project_dir / ".last_run"
    if global_file != last_run_file and global_file.exists():
        run_id = global_file.read_text().strip()
        if run_id:
            return run_id

    return None


def resolve_run_from_args(args: argparse.Namespace, project_dir: Path) -> tuple[Path | None, str | None, str | None]:
    """Resolve run directory from --run-dir, --run-id, $LAST_RUN, or .last_run.

    Priority:
    1. --run-dir: Use existing directory (must exist)
    2. --run-id: Create/use runs/<run-id>/ directory
    3. $LAST_RUN: Use environment variable as run-id
    4. .last_run file: Use file contents as run-id
    5. Neither: Return None (legacy mode)

    Args:
        args: Parsed arguments (expects run_dir and/or run_id attributes)
        project_dir: Project root directory

    Returns:
        tuple: (run_dir Path or None, run_id string or None, source string or None)
        source is one of: "--run-dir", "--run-id", "$LAST_RUN", ".last_run", None

    Raises:
        ValueError: If --run-dir specified but doesn't exist
    """
    # --run-dir takes precedence (explicit path)
    if hasattr(args, 'run_dir') and args.run_dir is not None:
        if not args.run_dir.exists():
            raise ValueError(f"Run directory not found: {args.run_dir}")
        return args.run_dir, args.run_dir.name, "--run-dir"

    # --run-id creates/uses runs/<run-id>/
    run_id = None
    source = None
    if hasattr(args, 'run_id') and args.run_id is not None:
        run_id = args.run_id
        source = "--run-id"
    elif os.environ.get('LAST_RUN'):
        run_id = os.environ['LAST_RUN']
        source = "$LAST_RUN"
    else:
        run_id = load_last_run(project_dir)
        if run_id:
            source = ".last_run"

    if run_id:
        runs_dir = project_dir / "runs"
        run_dir = runs_dir / run_id

        # Create if doesn't exist and save as last run
        if not run_dir.exists():
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "sampled").mkdir(exist_ok=True)
            (run_dir / "analyses").mkdir(exist_ok=True)
            (run_dir / "reports").mkdir(exist_ok=True)
            save_last_run(project_dir, run_id)

        return run_dir, run_id, source

    # Neither specified - legacy mode
    return None, None, None


def list_existing_runs(project_dir: Path) -> list[tuple[str, Path]]:
    """List existing run directories sorted by modification time (newest first).

    Args:
        project_dir: Project root directory

    Returns:
        List of (run_id, run_dir) tuples, newest first
    """
    runs_dir = project_dir / "runs"
    if not runs_dir.exists():
        return []

    runs = []
    for d in runs_dir.iterdir():
        if d.is_dir() and not d.is_symlink() and d.name != "latest":
            runs.append((d.name, d))

    # Sort by modification time, newest first
    runs.sort(key=lambda x: x[1].stat().st_mtime, reverse=True)
    return runs


def require_explicit_run_noninteractive(source: str | None) -> None:
    """Raise error if non-interactive and no explicit run specified.

    In non-interactive contexts (scripts, cron, CI/CD), require explicit
    --run-id or --run-dir to avoid silent fallback to legacy mode.

    Args:
        source: Where the run was resolved from, or None

    Raises:
        SystemExit: If non-interactive and source is implicit or None
    """
    import sys

    if sys.stdin.isatty():
        return  # Interactive - prompts will handle it

    # Explicit sources are OK
    if source in ("--run-id", "--run-dir"):
        return

    # Non-interactive with implicit/no source - error
    print("\nERROR: Non-interactive mode requires explicit --run-id or --run-dir", file=sys.stderr)
    print("\nExamples:", file=sys.stderr)
    print("  python3 tools/sample_transcripts.py -n 50 --run-id my_experiment", file=sys.stderr)
    print("  python3 tools/batch_analyze.py --run-dir runs/my_experiment", file=sys.stderr)
    print("\nOr set environment variable:", file=sys.stderr)
    print("  export LAST_RUN=my_experiment", file=sys.stderr)
    sys.exit(1)


def confirm_or_select_run(
    project_dir: Path,
    current_run_dir: Path,
    current_run_id: str,
    source: str
) -> tuple[Path | None, str | None]:
    """Confirm current run or allow selecting a different one.

    Called when a run is resolved from .last_run or $LAST_RUN (implicit sources).
    Shows the resolved run and asks for confirmation.

    Args:
        project_dir: Project root directory
        current_run_dir: Currently resolved run directory
        current_run_id: Currently resolved run ID
        source: Where the run was resolved from (e.g., ".last_run", "$LAST_RUN")

    Returns:
        tuple: (run_dir Path, run_id string) or (None, None) for legacy mode
    """
    import sys

    # Non-interactive mode - use resolved run without confirmation
    # (already validated by require_explicit_run_noninteractive if needed)
    if not sys.stdin.isatty():
        return current_run_dir, current_run_id

    # Get run info
    mtime = datetime.fromtimestamp(current_run_dir.stat().st_mtime) if current_run_dir.exists() else None
    mtime_str = mtime.strftime('%Y-%m-%d %H:%M') if mtime else "new"

    # Count contents
    sampled_count = len(list((current_run_dir / "sampled").glob("*"))) if (current_run_dir / "sampled").exists() else 0
    analyses_count = len(list((current_run_dir / "analyses").glob("*.json"))) if (current_run_dir / "analyses").exists() else 0

    print("\n" + "=" * 60)
    print(f"CONTINUE WITH RUN? (from {source})")
    print("=" * 60)
    print(f"\n  Run ID:    {current_run_id}")
    print(f"  Path:      {current_run_dir}")
    print(f"  Modified:  {mtime_str}")
    print(f"  Contents:  {sampled_count} sampled, {analyses_count} analyses")
    print("\nOptions:")
    print("  [Enter] Continue with this run")
    print("  [c]     Choose a DIFFERENT run")
    print("  [q]     Quit")

    while True:
        try:
            choice = input("\nSelect option: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            sys.exit(0)

        if choice == '' or choice == 'y':
            print(f"Using run: {current_run_id}")
            return current_run_dir, current_run_id

        if choice == 'q':
            print("Aborted.")
            sys.exit(0)

        if choice == 'c':
            # Show full selection menu
            return prompt_for_run(project_dir)

        print("Invalid option. Press Enter to continue, 'c' to change, or 'q' to quit.")


def prompt_for_run(project_dir: Path, auto_create: bool = False) -> tuple[Path | None, str | None]:
    """Interactive prompt to select or create a run.

    Called when no run is specified via args, env, or .last_run file.
    Shows existing runs and offers to create a new one.

    Args:
        project_dir: Project root directory
        auto_create: If True, skip prompt and auto-create new run

    Returns:
        tuple: (run_dir Path, run_id string) or (None, None) for legacy mode
    """
    import sys

    # Non-interactive mode - can't prompt
    if not sys.stdin.isatty() or auto_create:
        if auto_create:
            runs_dir = project_dir / "runs"
            return create_run_directory(runs_dir)
        return None, None

    runs = list_existing_runs(project_dir)
    runs_dir = project_dir / "runs"

    print("\n" + "=" * 60)
    print("RUN SELECTION")
    print("=" * 60)

    if runs:
        print(f"\nExisting runs ({len(runs)}):")
        # Show up to 5 most recent
        for i, (run_id, run_dir) in enumerate(runs[:5], 1):
            mtime = datetime.fromtimestamp(run_dir.stat().st_mtime)
            print(f"  [{i}] {run_id}  ({mtime.strftime('%Y-%m-%d %H:%M')})")
        if len(runs) > 5:
            print(f"  ... and {len(runs) - 5} more")

    print("\nOptions:")
    print("  [n] Create NEW run (auto-generated timestamp ID)")
    print("  [c] Create NEW run with CUSTOM ID")
    if runs:
        print("  [1-5] Use existing run (number from list above)")
    print("  [l] Use LEGACY mode (flat directories, no isolation)")
    print("  [q] Quit")

    while True:
        try:
            choice = input("\nSelect option: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.")
            return None, None

        if choice == 'q':
            print("Aborted.")
            sys.exit(0)

        if choice == 'l':
            print("Using legacy mode (flat directories)")
            return None, None

        if choice == 'n':
            run_dir, run_id = create_run_directory(runs_dir)
            print(f"Created new run: {run_id}")
            return run_dir, run_id

        if choice == 'c':
            try:
                custom_id = input("Enter run ID: ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nAborted.")
                return None, None

            if not custom_id:
                print("Invalid ID. Try again.")
                continue

            # Sanitize: replace spaces/special chars
            custom_id = custom_id.replace(' ', '_').replace('/', '-')
            run_dir, run_id = create_run_directory(runs_dir, custom_id)
            print(f"Created new run: {run_id}")
            return run_dir, run_id

        # Check if it's a number for existing run
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < min(5, len(runs)):
                run_id, run_dir = runs[idx]
                save_last_run(project_dir, run_id)
                print(f"Using existing run: {run_id}")
                return run_dir, run_id
            else:
                print("Invalid selection. Try again.")
                continue

        print("Invalid option. Try again.")


def create_run_directory(
    runs_dir: Path,
    run_id: Optional[str] = None
) -> tuple[Path, str]:
    """Create a new run directory with timestamp-based ID.

    Also saves the run_id to .last_run for automatic reuse.

    Args:
        runs_dir: Parent directory for all runs (e.g., project_dir/runs)
        run_id: Custom run ID (default: timestamp like "run_20260121_143000")

    Returns:
        tuple: (run_dir Path, run_id string)

    Example:
        >>> run_dir, run_id = create_run_directory(Path("runs"))
        >>> run_dir  # Path("runs/run_20260121_143000")
        >>> run_id  # "run_20260121_143000"
    """
    if run_id is None:
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    run_dir = runs_dir / run_id

    # Create run directory and subdirectories
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "sampled").mkdir(exist_ok=True)
    (run_dir / "analyses").mkdir(exist_ok=True)
    (run_dir / "reports").mkdir(exist_ok=True)

    # Save as last run (project_dir is parent of runs_dir)
    project_dir = runs_dir.parent
    save_last_run(project_dir, run_id)

    return run_dir, run_id


def save_config(
    run_dir: Path,
    args: argparse.Namespace,
    extra: Optional[dict] = None
) -> Path:
    """Save run configuration to config.json for reproducibility.

    Args:
        run_dir: Run directory path
        args: Parsed command-line arguments
        extra: Additional configuration data (e.g., source stats)

    Returns:
        Path to the saved config.json file
    """
    config = {
        "run_id": run_dir.name,
        "created_at": datetime.now().isoformat(),
        "version": "4.1",
        "args": {
            k: str(v) if isinstance(v, Path) else v
            for k, v in vars(args).items()
            if not k.startswith('_')
        }
    }

    if extra:
        config.update(extra)

    config_path = run_dir / "config.json"
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=2)

    return config_path


def load_config(run_dir: Path) -> Optional[dict]:
    """Load run configuration from config.json.

    Args:
        run_dir: Run directory path

    Returns:
        Configuration dict or None if not found
    """
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return None

    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def update_status(
    run_dir: Path,
    stage: str,
    status: str,
    **kwargs
) -> Path:
    """Update pipeline status in status.json.

    Args:
        run_dir: Run directory path
        stage: Pipeline stage name (e.g., "sampling", "analysis", "metrics")
        status: Status value (e.g., "in_progress", "completed", "failed")
        **kwargs: Additional stage-specific data (e.g., count=50, success=48, failed=2)

    Returns:
        Path to the status.json file

    Example:
        >>> update_status(run_dir, "analysis", "completed", success=48, failed=2)
    """
    status_path = run_dir / "status.json"

    # Load existing status or create new
    if status_path.exists():
        with open(status_path, 'r', encoding='utf-8') as f:
            status_data = json.load(f)
    else:
        status_data = {
            "run_id": run_dir.name,
            "created_at": datetime.now().isoformat(),
            "stages": {}
        }

    # Update timestamp and stage
    status_data["updated_at"] = datetime.now().isoformat()
    status_data["stages"][stage] = {"status": status, **kwargs}

    # Compute overall status
    stages = status_data.get("stages", {})
    if any(s.get("status") == "failed" for s in stages.values()):
        status_data["overall_status"] = "failed"
    elif all(s.get("status") == "completed" for s in stages.values()):
        status_data["overall_status"] = "completed"
    else:
        status_data["overall_status"] = "in_progress"

    with open(status_path, 'w', encoding='utf-8') as f:
        json.dump(status_data, f, indent=2)

    return status_path


def load_status(run_dir: Path) -> Optional[dict]:
    """Load pipeline status from status.json.

    Args:
        run_dir: Run directory path

    Returns:
        Status dict or None if not found
    """
    status_path = run_dir / "status.json"
    if not status_path.exists():
        return None

    with open(status_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def update_latest_symlink(run_dir: Path) -> None:
    """Update the 'latest' symlink to point to this run.

    Args:
        run_dir: Run directory path (e.g., runs/run_20260121_143000)
    """
    runs_dir = run_dir.parent
    latest_link = runs_dir / "latest"

    # Remove existing symlink if present
    if latest_link.is_symlink():
        latest_link.unlink()
    elif latest_link.exists():
        # It's a real file/directory, don't overwrite
        return

    # Create relative symlink (works better across systems)
    try:
        latest_link.symlink_to(run_dir.name)
    except OSError:
        # Symlinks may fail on some systems (e.g., Windows without admin)
        pass


def resolve_run_dir(args: argparse.Namespace, project_dir: Path) -> Optional[Path]:
    """Resolve the run directory from command-line arguments.

    This helper determines whether to use run-based isolation based on
    the presence of --run-dir, --legacy, or auto-creation settings.

    Args:
        args: Parsed command-line arguments (expects run_dir and legacy attrs)
        project_dir: Project root directory

    Returns:
        Path to run directory, or None for legacy mode
    """
    # Explicit run directory specified
    if hasattr(args, 'run_dir') and args.run_dir is not None:
        return args.run_dir

    # Legacy mode explicitly requested
    if hasattr(args, 'legacy') and args.legacy:
        return None

    # Auto-create new run (default behavior when not in legacy mode)
    # This should be handled by the orchestrator, not individual tools
    return None


def get_project_dir() -> Path:
    """Get the project root directory.

    Returns:
        Path to project root (parent of tools/ directory)
    """
    return Path(__file__).parent.parent


def count_transcript_files(transcript_dir: Path) -> int:
    """Count available transcript files in a directory.

    Args:
        transcript_dir: Directory to count transcripts in

    Returns:
        Count of .txt and .json transcript files
    """
    if not transcript_dir.exists():
        return 0

    count = 0
    seen_stems = set()

    for ext in ['.json', '.txt']:
        for f in transcript_dir.iterdir():
            if f.is_file() and f.suffix == ext and f.stem not in seen_stems:
                count += 1
                seen_stems.add(f.stem)

    return count
