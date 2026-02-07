#!/usr/bin/env python3
"""
DuckDB Analytics Loader for Voice Agent Analytics (v5.0)

Loads analysis JSONs into a DuckDB database for flexible SQL-based analytics.
Sits alongside compute_metrics.py (which continues to drive the pipeline).

Creates:
- calls table: 1 row per call analysis (flattened)
- call_actions view: UNNEST'd actions array
- call_clarifications view: UNNEST'd clarifications array
- call_corrections view: UNNEST'd corrections array
- call_loops view: UNNEST'd loops array

Usage:
    python3 tools/load_duckdb.py runs/run_XXXX/analyses/

    # Creates: runs/run_XXXX/analytics.duckdb
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("Error: duckdb not installed. Run: pip install duckdb", file=sys.stderr)
    sys.exit(1)


def load_analyses(analyses_dir: Path) -> list[dict]:
    """Load all analysis JSON files from directory."""
    analyses = []
    for f in sorted(analyses_dir.glob("*.json")):
        try:
            with open(f, 'r', encoding='utf-8') as fp:
                data = json.load(fp)
                if isinstance(data, dict) and "call_id" in data:
                    analyses.append(data)
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Could not load {f}: {e}", file=sys.stderr)
    return analyses


def create_database(analyses: list[dict], db_path: Path) -> None:
    """Create DuckDB database from analysis data."""
    con = duckdb.connect(str(db_path))

    # Write analyses as JSON for DuckDB to read
    # DuckDB can read JSON natively, but for complex nested structures
    # it's easier to insert via Python
    con.execute("DROP TABLE IF EXISTS calls")
    con.execute("DROP VIEW IF EXISTS call_actions")
    con.execute("DROP VIEW IF EXISTS call_clarifications")
    con.execute("DROP VIEW IF EXISTS call_corrections")
    con.execute("DROP VIEW IF EXISTS call_loops")

    # Create table with explicit schema (v5.0 + v4.x compat)
    con.execute("""
        CREATE TABLE calls (
            call_id VARCHAR,
            schema_version VARCHAR,
            turns INTEGER,
            ended_by VARCHAR,
            duration_seconds DOUBLE,
            -- Intent
            intent VARCHAR,
            intent_context VARCHAR,
            secondary_intent VARCHAR,
            -- v5.0: Orthogonal disposition model
            call_scope VARCHAR,
            call_outcome VARCHAR,
            -- v4.x: Legacy disposition (populated via bridge for v5.0)
            disposition VARCHAR,
            resolution VARCHAR,
            -- Quality
            effectiveness INTEGER,
            quality INTEGER,
            effort INTEGER,
            sentiment_start VARCHAR,
            sentiment_end VARCHAR,
            -- Failure
            failure_type VARCHAR,
            failure_detail VARCHAR,
            -- Friction
            derailed_at INTEGER,
            -- Insights
            summary VARCHAR,
            verbatim VARCHAR,
            coaching VARCHAR,
            -- Flags
            repeat_caller BOOLEAN,
            -- v5.0: Conditional qualifiers
            escalation_trigger VARCHAR,
            abandon_stage VARCHAR,
            resolution_confirmed BOOLEAN,
            -- v4.5 legacy fields (populated for v4.x analyses)
            pre_intent_subtype VARCHAR,
            escalation_initiator VARCHAR,
            -- Shared dashboard fields
            transfer_destination VARCHAR,
            transfer_queue_detected BOOLEAN,
            -- Arrays as JSON strings for UNNEST
            actions_json VARCHAR,
            clarifications_json VARCHAR,
            corrections_json VARCHAR,
            loops_json VARCHAR,
            steps_json VARCHAR
        )
    """)

    # Insert data — synthesize legacy disposition for v5.0 analyses
    for a in analyses:
        # Bridge: synthesize legacy disposition from v5.0 orthogonal fields
        disposition = a.get("disposition")
        if not disposition and "call_scope" in a and "call_outcome" in a:
            scope, outcome = a["call_scope"], a["call_outcome"]
            if scope == "no_request":
                disposition = "pre_intent"
            elif scope == "out_of_scope":
                disposition = "out_of_scope_handled" if outcome == "completed" else "out_of_scope_failed"
            elif scope in ("in_scope", "mixed"):
                if outcome == "completed":
                    disposition = "in_scope_success"
                elif outcome == "escalated":
                    disposition = "escalated"
                elif outcome == "abandoned":
                    disposition = "in_scope_failed"

        con.execute("""
            INSERT INTO calls VALUES (
                ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?,
                ?, ?, ?,
                ?,
                ?, ?, ?,
                ?, ?,
                ?, ?,
                ?, ?, ?, ?, ?
            )
        """, [
            a.get("call_id"),
            a.get("schema_version"),
            a.get("turns"),
            a.get("ended_by"),
            a.get("duration_seconds"),
            # Intent
            a.get("intent"),
            a.get("intent_context"),
            a.get("secondary_intent"),
            # v5.0: Orthogonal disposition
            a.get("call_scope"),
            a.get("call_outcome"),
            # Legacy disposition (direct or synthesized)
            disposition,
            a.get("resolution"),
            # Quality
            a.get("effectiveness"),
            a.get("quality"),
            a.get("effort"),
            a.get("sentiment_start"),
            a.get("sentiment_end"),
            # Failure
            a.get("failure_type"),
            a.get("failure_detail"),
            # Friction
            a.get("derailed_at"),
            # Insights
            a.get("summary"),
            a.get("verbatim"),
            a.get("coaching"),
            # Flags
            a.get("repeat_caller"),
            # v5.0: Conditional qualifiers
            a.get("escalation_trigger"),
            a.get("abandon_stage"),
            a.get("resolution_confirmed"),
            # v4.5 legacy fields
            a.get("pre_intent_subtype"),
            a.get("escalation_initiator"),
            # Shared dashboard fields
            a.get("transfer_destination"),
            a.get("transfer_queue_detected"),
            # Arrays as JSON
            json.dumps(a.get("actions", [])),
            json.dumps(a.get("clarifications", [])),
            json.dumps(a.get("corrections", [])),
            json.dumps(a.get("loops", [])),
            json.dumps(a.get("steps", [])),
        ])

    # Create UNNEST views using JSON extraction
    con.execute("""
        CREATE VIEW call_actions AS
        SELECT
            c.call_id,
            c.call_scope,
            c.call_outcome,
            c.disposition,
            a.action,
            a.outcome,
            a.detail
        FROM calls c,
        LATERAL (
            SELECT
                json_extract_string(value, '$.action') as action,
                json_extract_string(value, '$.outcome') as outcome,
                json_extract_string(value, '$.detail') as detail
            FROM json_each(c.actions_json)
        ) a
        WHERE c.actions_json != '[]'
    """)

    con.execute("""
        CREATE VIEW call_clarifications AS
        SELECT
            c.call_id,
            cl.turn,
            cl.type,
            cl.cause,
            cl.note
        FROM calls c,
        LATERAL (
            SELECT
                CAST(json_extract(value, '$.turn') AS INTEGER) as turn,
                json_extract_string(value, '$.type') as type,
                json_extract_string(value, '$.cause') as cause,
                json_extract_string(value, '$.note') as note
            FROM json_each(c.clarifications_json)
        ) cl
        WHERE c.clarifications_json != '[]'
    """)

    con.execute("""
        CREATE VIEW call_corrections AS
        SELECT
            c.call_id,
            cr.turn,
            cr.severity,
            cr.note
        FROM calls c,
        LATERAL (
            SELECT
                CAST(json_extract(value, '$.turn') AS INTEGER) as turn,
                json_extract_string(value, '$.severity') as severity,
                json_extract_string(value, '$.note') as note
            FROM json_each(c.corrections_json)
        ) cr
        WHERE c.corrections_json != '[]'
    """)

    con.execute("""
        CREATE VIEW call_loops AS
        SELECT
            c.call_id,
            l.type,
            l.subject,
            l.note
        FROM calls c,
        LATERAL (
            SELECT
                json_extract_string(value, '$.type') as type,
                json_extract_string(value, '$.subject') as subject,
                json_extract_string(value, '$.note') as note
            FROM json_each(c.loops_json)
        ) l
        WHERE c.loops_json != '[]'
    """)

    con.close()


def print_summary(db_path: Path) -> None:
    """Print summary of loaded database."""
    con = duckdb.connect(str(db_path), read_only=True)

    # Row count
    n = con.execute("SELECT COUNT(*) FROM calls").fetchone()[0]
    print(f"\nLoaded {n} analyses into {db_path}")

    # Check if v5.0 data exists
    v5_count = con.execute("SELECT COUNT(*) FROM calls WHERE call_scope IS NOT NULL").fetchone()[0]

    if v5_count > 0:
        # v5.0: Scope x Outcome cross-tab
        print("\nScope x Outcome distribution:")
        rows = con.execute("""
            SELECT call_scope, call_outcome, COUNT(*) as n,
                   ROUND(COUNT(*)::FLOAT / (SELECT COUNT(*) FROM calls), 3) as rate
            FROM calls
            WHERE call_scope IS NOT NULL
            GROUP BY call_scope, call_outcome
            ORDER BY n DESC
        """).fetchall()
        for scope, outcome, count, rate in rows:
            print(f"  {scope} / {outcome}: {count} ({rate*100:.1f}%)")

        # v5.0 field completeness
        print("\nv5.0 field completeness:")
        fields = [
            ("call_scope", "call_scope IS NOT NULL"),
            ("call_outcome", "call_outcome IS NOT NULL"),
            ("resolution_confirmed", "resolution_confirmed IS NOT NULL"),
            ("escalation_trigger", "escalation_trigger IS NOT NULL"),
            ("abandon_stage", "abandon_stage IS NOT NULL"),
            ("actions (non-empty)", "actions_json != '[]'"),
            ("transfer_destination", "transfer_destination IS NOT NULL"),
        ]
        for label, condition in fields:
            count = con.execute(f"SELECT COUNT(*) FROM calls WHERE {condition}").fetchone()[0]
            print(f"  {label}: {count}/{n}")
    else:
        # v4.x fallback: Legacy disposition
        print("\nDisposition distribution:")
        rows = con.execute("""
            SELECT disposition, COUNT(*) as n,
                   ROUND(COUNT(*)::FLOAT / (SELECT COUNT(*) FROM calls), 3) as rate
            FROM calls
            GROUP BY disposition
            ORDER BY n DESC
        """).fetchall()
        for disp, count, rate in rows:
            print(f"  {disp}: {count} ({rate*100:.1f}%)")

    # Action counts
    action_count = con.execute("SELECT COUNT(*) FROM call_actions").fetchone()[0]
    if action_count > 0:
        print(f"\nActions: {action_count} total")
        rows = con.execute("""
            SELECT action, COUNT(*) as n FROM call_actions GROUP BY action ORDER BY n DESC LIMIT 5
        """).fetchall()
        for action, count in rows:
            print(f"  {action}: {count}")

    con.close()


def main():
    parser = argparse.ArgumentParser(description="Load analysis JSONs into DuckDB for SQL analytics")
    parser.add_argument("analyses_dir", type=Path, help="Path to analyses directory")
    parser.add_argument("-o", "--output", type=Path, help="Output .duckdb path (default: parent/analytics.duckdb)")
    args = parser.parse_args()

    if not args.analyses_dir.exists():
        print(f"Error: Directory not found: {args.analyses_dir}", file=sys.stderr)
        return 1

    # Default output: sibling of analyses dir
    db_path = args.output or (args.analyses_dir.parent / "analytics.duckdb")

    # Load analyses
    analyses = load_analyses(args.analyses_dir)
    if not analyses:
        print("Error: No valid analysis JSONs found", file=sys.stderr)
        return 1

    print(f"Loading {len(analyses)} analyses into DuckDB...", file=sys.stderr)

    # Create database
    create_database(analyses, db_path)

    # Print summary
    print_summary(db_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
