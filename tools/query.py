#!/usr/bin/env python3
"""
SQL Query CLI for Voice Agent Analytics (v5.0)

Run SQL queries against a run's DuckDB analytics database.

Usage:
    # Interactive mode (opens DuckDB CLI-like prompt)
    python3 tools/query.py runs/run_XXXX/

    # One-shot query
    python3 tools/query.py runs/run_XXXX/ -q "SELECT disposition, COUNT(*) FROM calls GROUP BY 1"

    # Predefined dashboard queries
    python3 tools/query.py runs/run_XXXX/ --dashboard
"""

import argparse
import sys
from pathlib import Path

try:
    import duckdb
except ImportError:
    print("Error: duckdb not installed. Run: pip install duckdb", file=sys.stderr)
    sys.exit(1)


DASHBOARD_QUERIES = {
    "1. Header KPIs": """
SELECT
  COUNT(*) as total_calls,
  COUNT(*) FILTER (WHERE call_scope IN ('in_scope','mixed')) as in_scope,
  COUNT(*) FILTER (WHERE call_scope IN ('in_scope','mixed') AND call_outcome = 'completed') as contained,
  ROUND(COUNT(*) FILTER (WHERE call_scope IN ('in_scope','mixed') AND call_outcome = 'completed')::DOUBLE
    / NULLIF(COUNT(*) FILTER (WHERE call_scope IN ('in_scope','mixed')), 0), 2) as containment_rate,
  ROUND(PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY duration_seconds)
    FILTER (WHERE call_outcome = 'completed'), 1) as p80_completed_sec,
  ROUND(COUNT(*) FILTER (WHERE resolution_confirmed = true)::DOUBLE
    / NULLIF(COUNT(*) FILTER (WHERE call_outcome = 'completed'), 0), 2) as confirmed_rate
FROM calls;
""",

    "2. Call Funnel (Scope)": """
SELECT
  COUNT(*) as total,
  COUNT(*) FILTER (WHERE call_scope = 'no_request') as no_request,
  COUNT(*) FILTER (WHERE call_scope != 'no_request') as request_made,
  COUNT(*) FILTER (WHERE call_scope = 'out_of_scope') as out_of_scope,
  COUNT(*) FILTER (WHERE call_scope = 'in_scope') as in_scope,
  COUNT(*) FILTER (WHERE call_scope = 'mixed') as mixed
FROM calls;
""",

    "3. No-Request Breakdown": """
SELECT abandon_stage, COUNT(*) as n
FROM calls WHERE call_scope = 'no_request'
GROUP BY 1
ORDER BY n DESC;
""",

    "4. In-Scope Outcomes": """
SELECT
  COUNT(*) FILTER (WHERE call_outcome = 'completed') as completed,
  COUNT(*) FILTER (WHERE resolution_confirmed = true) as confirmed,
  COUNT(*) FILTER (WHERE resolution_confirmed = false) as unconfirmed,
  COUNT(*) FILTER (WHERE call_outcome = 'escalated') as escalated,
  COUNT(*) FILTER (WHERE escalation_trigger = 'customer_requested') as esc_customer,
  COUNT(*) FILTER (WHERE escalation_trigger = 'scope_limit') as esc_scope_limit,
  COUNT(*) FILTER (WHERE escalation_trigger = 'task_failure') as esc_task_failure,
  COUNT(*) FILTER (WHERE call_outcome = 'abandoned') as abandoned,
  COUNT(*) FILTER (WHERE abandon_stage = 'mid_task') as abandon_mid_task,
  COUNT(*) FILTER (WHERE abandon_stage = 'post_delivery') as abandon_post_delivery
FROM calls
WHERE call_scope IN ('in_scope', 'mixed');
""",

    "5. Scope x Outcome Cross-Tab": """
SELECT call_scope, call_outcome, COUNT(*) as n,
  ROUND(COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER(), 2) as rate
FROM calls
GROUP BY call_scope, call_outcome
ORDER BY n DESC;
""",

    "6. Action Performance": """
SELECT action,
  COUNT(*) as attempted,
  COUNT(*) FILTER (WHERE outcome='success') as success,
  COUNT(*) FILTER (WHERE outcome='retry') as retry,
  COUNT(*) FILTER (WHERE outcome='failed') as failed,
  ROUND(COUNT(*) FILTER (WHERE outcome='success')::DOUBLE / COUNT(*), 2) as success_rate
FROM call_actions
GROUP BY action ORDER BY attempted DESC;
""",

    "7. Transfer Quality": """
SELECT transfer_destination, COUNT(*) as n,
  ROUND(COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER(), 2) as rate
FROM calls WHERE transfer_destination IS NOT NULL
GROUP BY 1 ORDER BY n DESC;
""",

    "8. Escalation Trigger Breakdown": """
SELECT escalation_trigger, COUNT(*) as n,
  ROUND(COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER(), 2) as rate,
  ROUND(AVG(duration_seconds), 1) as avg_duration_sec,
  ROUND(AVG(effectiveness), 1) as avg_effectiveness
FROM calls
WHERE call_outcome = 'escalated' AND escalation_trigger IS NOT NULL
GROUP BY escalation_trigger ORDER BY n DESC;
""",

    "9. Abandon Stage Analysis": """
SELECT abandon_stage, COUNT(*) as n,
  ROUND(COUNT(*)::DOUBLE / SUM(COUNT(*)) OVER(), 2) as rate,
  ROUND(AVG(duration_seconds), 1) as avg_duration_sec,
  ROUND(AVG(turns), 1) as avg_turns
FROM calls
WHERE call_outcome = 'abandoned' AND abandon_stage IS NOT NULL
GROUP BY abandon_stage ORDER BY n DESC;
""",

    "10. Sentiment Journey": """
WITH journeys AS (
  SELECT *,
    CASE
      WHEN sentiment_end IN ('satisfied') AND sentiment_start IN ('frustrated','angry','neutral') THEN 'improved'
      WHEN sentiment_end IN ('frustrated','angry') AND sentiment_start IN ('satisfied','neutral') THEN 'degraded'
      WHEN sentiment_start = sentiment_end THEN 'stable'
      WHEN sentiment_start = 'angry' AND sentiment_end = 'neutral' THEN 'improved'
      WHEN sentiment_start = 'frustrated' AND sentiment_end = 'neutral' THEN 'improved'
      ELSE 'mixed'
    END as direction
  FROM calls WHERE sentiment_start IS NOT NULL AND sentiment_end IS NOT NULL
)
SELECT sentiment_start || ' -> ' || sentiment_end as journey, direction,
  COUNT(*) as n,
  ROUND(COUNT(*)::DOUBLE / (SELECT COUNT(*) FROM journeys), 2) as rate
FROM journeys GROUP BY journey, direction ORDER BY n DESC;
""",

    "11. Duration by Scope x Outcome": """
SELECT call_scope, call_outcome, COUNT(*) as n,
  ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_seconds), 1) as p50_sec,
  ROUND(PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY duration_seconds), 1) as p80_sec,
  ROUND(AVG(duration_seconds), 1) as mean_sec,
  ROUND(MAX(duration_seconds), 1) as max_sec
FROM calls WHERE duration_seconds IS NOT NULL
GROUP BY call_scope, call_outcome ORDER BY call_scope, call_outcome;
""",

    "12. Friction x Outcome Correlation": """
SELECT call_outcome, COUNT(*) as n,
  ROUND(COUNT(*) FILTER (WHERE clarifications_json != '[]')::DOUBLE / COUNT(*), 2) as clarification_rate,
  ROUND(COUNT(*) FILTER (WHERE corrections_json != '[]')::DOUBLE / COUNT(*), 2) as correction_rate,
  ROUND(COUNT(*) FILTER (WHERE loops_json != '[]')::DOUBLE / COUNT(*), 2) as loop_rate,
  ROUND(AVG(effort), 1) as avg_effort
FROM calls GROUP BY call_outcome ORDER BY call_outcome;
""",

    "13. Transfer-Then-Abandon (Failed Transfers)": """
SELECT transfer_reason, transfer_destination,
  COUNT(*) as n,
  ROUND(AVG(duration_seconds), 0) as avg_wait_sec
FROM calls
WHERE call_outcome IN ('abandoned', 'escalated')
  AND transfer_destination IS NOT NULL
GROUP BY 1, 2 ORDER BY n DESC;
""",
}


def run_query(con: duckdb.DuckDBPyConnection, sql: str) -> None:
    """Execute a query and print results."""
    try:
        result = con.execute(sql)
        columns = [desc[0] for desc in result.description]
        rows = result.fetchall()

        if not rows:
            print("(no results)")
            return

        # Calculate column widths
        widths = [len(c) for c in columns]
        for row in rows:
            for i, val in enumerate(row):
                widths[i] = max(widths[i], len(str(val)))

        # Print header
        header = " | ".join(c.ljust(widths[i]) for i, c in enumerate(columns))
        print(header)
        print("-+-".join("-" * w for w in widths))

        # Print rows
        for row in rows:
            line = " | ".join(str(val).ljust(widths[i]) for i, val in enumerate(row))
            print(line)

        print(f"\n({len(rows)} rows)")
    except duckdb.Error as e:
        print(f"SQL Error: {e}", file=sys.stderr)


def run_dashboard(con: duckdb.DuckDBPyConnection) -> None:
    """Run all predefined dashboard queries."""
    for title, sql in DASHBOARD_QUERIES.items():
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        run_query(con, sql)


def interactive_mode(con: duckdb.DuckDBPyConnection) -> None:
    """Simple interactive SQL prompt."""
    print("DuckDB query shell. Type SQL queries, 'dashboard' for predefined queries, or 'quit' to exit.")
    print("Tables: calls | Views: call_actions, call_clarifications, call_corrections, call_loops\n")

    while True:
        try:
            sql = input("sql> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not sql:
            continue
        if sql.lower() in ("quit", "exit", "q"):
            break
        if sql.lower() == "dashboard":
            run_dashboard(con)
            continue
        if sql.lower() == "tables":
            run_query(con, "SHOW TABLES")
            continue

        run_query(con, sql)


def main():
    parser = argparse.ArgumentParser(description="SQL query CLI for DuckDB analytics")
    parser.add_argument("run_dir", type=Path, help="Run directory (containing analytics.duckdb)")
    parser.add_argument("-q", "--query", type=str, help="One-shot SQL query")
    parser.add_argument("--dashboard", action="store_true", help="Run predefined dashboard queries")
    parser.add_argument("--db", type=str, default="analytics.duckdb", help="Database filename (default: analytics.duckdb)")
    args = parser.parse_args()

    # Find database
    db_path = args.run_dir / args.db
    if not db_path.exists():
        # Maybe they passed the analyses dir
        parent_db = args.run_dir.parent / args.db
        if parent_db.exists():
            db_path = parent_db
        else:
            print(f"Error: Database not found: {db_path}", file=sys.stderr)
            print(f"Run 'python3 tools/load_duckdb.py {args.run_dir}/analyses/' first.", file=sys.stderr)
            return 1

    con = duckdb.connect(str(db_path), read_only=True)

    if args.dashboard:
        run_dashboard(con)
    elif args.query:
        run_query(con, args.query)
    else:
        interactive_mode(con)

    con.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
