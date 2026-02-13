#!/usr/bin/env python3
"""
V7 Dashboard — AI Voice Agent Performance Analytics

Reads v7.0 analysis JSONs via DuckDB read_json_auto() and prints a 4-act
narrative dashboard to stdout. No ETL — DuckDB dot notation reads nested
JSON fields directly.

Acts:
  1. The Big Picture     — scope split, containment, top requests
  2. Human Requests      — human_requested/department_requested analysis
  3. Quality & Impediments — impediments, agent issues, preventable escalations, scores, sentiment
  4. Operational Details  — duration, actions, transfers, queue performance, friction, abandons

Usage:
    python3 tools/dashboard_v7.py tests/golden/analyses_v7/
    python3 tools/dashboard_v7.py runs/v7_batch/
"""

import sys
from pathlib import Path

import duckdb

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

W = 72  # standard output width


def src(directory: str) -> str:
    """Return the read_json_auto expression for a directory of JSONs."""
    return f"read_json_auto('{directory}/*.json', union_by_name=true)"


def pct(n: int, total: int) -> str:
    """Format n/total as percentage, or '—' if total is 0."""
    return f"{n/total:.0%}" if total else "—"


def pct1(n: int, total: int) -> str:
    """Format n/total as percentage with 1 decimal, or '—' if total is 0."""
    return f"{n/total:.1%}" if total else "—"


def bar(n: int, total: int, width: int = 30) -> str:
    """Simple ASCII bar chart segment."""
    if total == 0:
        return ""
    filled = round(n / total * width)
    return "█" * filled + "░" * (width - filled)


def section(number: str, title: str) -> None:
    """Print a section header."""
    print()
    print(f"  {number} {title}")
    print("  " + "─" * (W - 4))


def act_header(title: str) -> None:
    """Print an act header."""
    print()
    print("  " + "═" * (W - 4))
    print(f"  {title}")
    print("  " + "═" * (W - 4))


def trunc(s: str, maxlen: int = 50) -> str:
    """Truncate string with ellipsis."""
    if s and len(s) > maxlen:
        return s[:maxlen - 1] + "…"
    return s or ""


def has_column(con: duckdb.DuckDBPyConnection, json_src: str, path: str) -> bool:
    """Check if a nested JSON field exists in the data."""
    try:
        con.sql(f"SELECT {path} FROM {json_src} LIMIT 1").fetchone()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Act 1: The Big Picture
# ---------------------------------------------------------------------------

def render_header_kpis(con, json_src, directory):
    """1.1 Header KPIs — one-line summary."""
    kpis = con.sql(f"""
        SELECT
            COUNT(*)                                                    AS total,
            COUNT(*) FILTER (resolution.primary.scope = 'in_scope'
                AND resolution.primary.outcome = 'fulfilled')           AS in_scope_fulfilled,
            COUNT(*) FILTER (resolution.primary.scope = 'in_scope')    AS in_scope_total,
            ROUND(AVG(duration_seconds), 0)                            AS avg_dur,
            ROUND(AVG(scores.effectiveness), 1)                        AS avg_eff,
            ROUND(AVG(scores.quality), 1)                              AS avg_qual,
            ROUND(AVG(scores.effort), 1)                               AS avg_effort
        FROM {json_src}
    """).fetchone()

    total, in_f, in_t, avg_dur, avg_eff, avg_qual, avg_effort = kpis
    contain = pct(in_f, in_t)

    path_label = Path(directory).name
    print(f"\n  V7 DASHBOARD: {path_label}")
    print("  " + "=" * (W - 4))
    print(f"  {total} calls  |  fulfillment: {contain} ({in_f}/{in_t} in-scope)"
          f"  |  avg {avg_dur:.0f}s  |  eff={avg_eff} qual={avg_qual} effort={avg_effort}")

    return total


def render_call_funnel(con, json_src, total):
    """1.2 Call Funnel — scope split with bar chart."""
    section("1.2", "CALL FUNNEL")

    rows = con.sql(f"""
        SELECT
            resolution.primary.scope AS scope,
            COUNT(*) AS n
        FROM {json_src}
        GROUP BY scope
        ORDER BY n DESC
    """).fetchall()

    for scope, n in rows:
        print(f"  {scope:<16s} {n:>4d}  {pct(n, total):>5s}  {bar(n, total)}")


def render_scope_outcome(con, json_src, total):
    """1.3 Scope x Outcome Cross-Tab."""
    section("1.3", "SCOPE × OUTCOME")

    rows = con.sql(f"""
        SELECT
            resolution.primary.scope AS scope,
            resolution.primary.outcome AS outcome,
            COUNT(*) AS n
        FROM {json_src}
        GROUP BY ALL
    """).fetchall()

    outcomes = ["fulfilled", "transferred", "abandoned"]
    scopes = ["in_scope", "out_of_scope", "no_request"]
    grid = {s: {o: 0 for o in outcomes} for s in scopes}
    for scope, outcome, n in rows:
        grid.setdefault(scope, {o: 0 for o in outcomes})[outcome] = n

    # Header
    hdr = f"  {'':16s}"
    for o in outcomes:
        hdr += f"{o:>14s}"
    hdr += f"  {'TOTAL':>14s}"
    print(hdr)
    print("  " + "─" * (W - 4))

    for s in scopes:
        row_total = sum(grid[s].values())
        if row_total == 0:
            continue
        line = f"  {s:<16s}"
        for o in outcomes:
            n = grid[s][o]
            if n == 0:
                line += f"{'—':>14s}"
            else:
                cell = f"{n:>3d} ({pct(n, total):>4s})"
                line += f"{cell:>14s}"
        line += f"  {row_total:>4d} ({pct(row_total, total):>4s})"
        print(line)

    # Totals row
    col_totals = [sum(grid[s][o] for s in scopes) for o in outcomes]
    print("  " + "─" * (W - 4))
    line = f"  {'TOTAL':<16s}"
    for ct in col_totals:
        cell = f"{ct:>3d} ({pct(ct, total):>4s})"
        line += f"{cell:>14s}"
    line += f"  {sum(col_totals):>4d} ({pct(sum(col_totals), total):>4s})"
    print(line)

    # Containment breakdown by scope (in_scope + out_of_scope, excluding no_request)
    request_total = sum(grid.get("in_scope", {}).values()) + sum(grid.get("out_of_scope", {}).values())
    if request_total > 0:
        try:
            buckets = con.sql(f"""
                SELECT
                    resolution.primary.scope AS scope,
                    CASE
                        WHEN resolution.primary.outcome = 'fulfilled'
                            THEN 'fulfilled'
                        WHEN ended_reason = 'assistant-forwarded-call'
                            THEN 'escalated'
                        WHEN resolution.primary.transfer IS NOT NULL
                            THEN 'transfer_abandoned'
                        ELSE 'unresolved'
                    END AS bucket,
                    COUNT(*) AS n
                FROM {json_src}
                WHERE resolution.primary.scope IN ('in_scope', 'out_of_scope')
                GROUP BY 1, 2
            """).fetchall()

            # Build per-scope dicts
            data = {}
            for scope, bucket, n in buckets:
                data.setdefault(scope, {})[bucket] = n

            print(f"\n  Containment breakdown ({request_total} calls with a request):\n")
            print(f"    {'':<26s} {'in_scope':>10s} {'out_scope':>10s} {'total':>10s}")
            print("    " + "─" * 58)

            for label, key in [("Contained", None), ("  Fulfilled", "fulfilled"),
                               ("  Unresolved", "unresolved"),
                               ("Not contained", None), ("  Escalated to human", "escalated"),
                               ("  Transfer abandoned", "transfer_abandoned")]:
                if key:
                    ins = data.get("in_scope", {}).get(key, 0)
                    oos = data.get("out_of_scope", {}).get(key, 0)
                    tot = ins + oos
                    print(f"    {label:<26s} {ins:>4d} {pct(ins, sum(data.get('in_scope', {}).values())):>5s}"
                          f" {oos:>4d} {pct(oos, sum(data.get('out_of_scope', {}).values())):>5s}"
                          f" {tot:>4d} {pct(tot, request_total):>5s}")
                else:
                    # Header row for contained / not contained
                    if label == "Contained":
                        keys = ["fulfilled", "unresolved"]
                    else:
                        keys = ["escalated", "transfer_abandoned"]
                    ins = sum(data.get("in_scope", {}).get(k, 0) for k in keys)
                    oos = sum(data.get("out_of_scope", {}).get(k, 0) for k in keys)
                    tot = ins + oos
                    ins_t = sum(data.get("in_scope", {}).values())
                    oos_t = sum(data.get("out_of_scope", {}).values())
                    print(f"    {label:<26s} {ins:>4d} {pct(ins, ins_t):>5s}"
                          f" {oos:>4d} {pct(oos, oos_t):>5s}"
                          f" {tot:>4d} {pct(tot, request_total):>5s}  {bar(tot, request_total, 16)}")
        except Exception:
            pass


def render_top_requests(con, json_src):
    """1.4 Top Requests — in-scope (fulfilled / unfulfilled) + out-of-scope.

    When request_category is present (v7.1+), groups by the enum for clean
    deterministic rows. Falls back to LOWER(request) for older data.
    """
    has_cat = has_column(con, json_src, "resolution.primary.request_category")
    if has_cat:
        # COALESCE handles mixed data where some analyses predate request_category
        group_col = "COALESCE(resolution.primary.request_category, '(uncategorized)')"
    else:
        group_col = "LOWER(resolution.primary.request)"
    col_label = "category" if has_cat else "request"

    # --- In-scope: fulfilled ---
    ful_rows = con.sql(f"""
        SELECT {group_col} AS grp, COUNT(*) AS n
        FROM {json_src}
        WHERE resolution.primary.scope = 'in_scope'
          AND resolution.primary.outcome = 'fulfilled'
        GROUP BY grp ORDER BY n DESC LIMIT 20
    """).fetchall()
    ful_total = sum(n for _, n in ful_rows)

    section("1.4a", f"IN-SCOPE REQUESTS — FULFILLED ({ful_total} calls)")
    print(f"  {col_label:<40s} {'n':>4s}  {'pct':>5s}")
    print("  " + "─" * (W - 4))
    for grp, n in ful_rows:
        print(f"  {trunc(grp, 40):<40s} {n:>4d}  {pct(n, ful_total):>5s}")

    # --- In-scope: unfulfilled (abandoned + transferred) ---
    unf_rows = con.sql(f"""
        SELECT
            {group_col} AS grp,
            COUNT(*) AS n,
            SUM(CASE WHEN resolution.primary.outcome = 'abandoned' THEN 1 ELSE 0 END) AS abandoned,
            SUM(CASE WHEN resolution.primary.outcome = 'transferred' THEN 1 ELSE 0 END) AS transferred
        FROM {json_src}
        WHERE resolution.primary.scope = 'in_scope'
          AND resolution.primary.outcome != 'fulfilled'
        GROUP BY grp ORDER BY n DESC LIMIT 20
    """).fetchall()
    unf_total = sum(n for _, n, _, _ in unf_rows)

    section("1.4b", f"IN-SCOPE REQUESTS — UNFULFILLED ({unf_total} calls)")
    print(f"  {col_label:<35s} {'n':>4s}  {'abandoned':>9s} {'transferred':>11s}")
    print("  " + "─" * (W - 4))
    for grp, n, abn, xfr in unf_rows:
        print(f"  {trunc(grp, 35):<35s} {n:>4d}  {abn:>9d} {xfr:>11d}")

    if ful_total + unf_total > 0:
        print(f"\n  → In-scope fulfillment: {ful_total}/{ful_total + unf_total}"
              f" ({pct(ful_total, ful_total + unf_total)})")

    # --- Out-of-scope ---
    oos_rows = con.sql(f"""
        SELECT
            {group_col} AS grp,
            COUNT(*) AS n,
            SUM(CASE WHEN resolution.primary.outcome = 'abandoned' THEN 1 ELSE 0 END) AS abandoned,
            SUM(CASE WHEN resolution.primary.outcome = 'transferred' THEN 1 ELSE 0 END) AS transferred
        FROM {json_src}
        WHERE resolution.primary.scope = 'out_of_scope'
        GROUP BY grp ORDER BY n DESC LIMIT 20
    """).fetchall()
    oos_total = sum(n for _, n, _, _ in oos_rows)

    section("1.4c", f"OUT-OF-SCOPE REQUESTS ({oos_total} calls)")
    print(f"  {col_label:<35s} {'n':>4s}  {'abandoned':>9s} {'transferred':>11s}")
    print("  " + "─" * (W - 4))
    for grp, n, abn, xfr in oos_rows:
        print(f"  {trunc(grp, 35):<35s} {n:>4d}  {abn:>9d} {xfr:>11d}")

    # --- no_request summary ---
    nr = con.sql(f"""
        SELECT COUNT(*) FROM {json_src}
        WHERE resolution.primary.scope = 'no_request'
    """).fetchone()[0]
    if nr > 0:
        print(f"\n  no_request: {nr} calls (abandoned before articulating intent)")

    if not has_cat:
        print(f"\n  [Using free-text LOWER(request) grouping — upgrade to v7.1+ for request_category enum]")


# ---------------------------------------------------------------------------
# Act 2: The Human-Request Phenomenon
# ---------------------------------------------------------------------------

def render_human_overview(con, json_src, total):
    """2.1 Human Request Overview."""
    section("2.1", "HUMAN REQUEST OVERVIEW")

    rows = con.sql(f"""
        SELECT
            resolution.primary.human_requested AS hr,
            COUNT(*) AS n
        FROM {json_src}
        GROUP BY hr
        ORDER BY n DESC
    """).fetchall()

    hr_total = sum(n for hr, n in rows if hr is not None)
    initial = sum(n for hr, n in rows if hr == "initial")
    after = sum(n for hr, n in rows if hr == "after_service")

    print(f"  Human requested:    {hr_total:>4d} / {total}  ({pct(hr_total, total)})")
    print(f"    initial:          {initial:>4d}  ({pct(initial, total)})")
    print(f"    after_service:    {after:>4d}  ({pct(after, total)})")
    print(f"    no request:       {total - hr_total:>4d}  ({pct(total - hr_total, total)})")

    # AI conversion rate: human_requested but still fulfilled
    converted = con.sql(f"""
        SELECT COUNT(*) FROM {json_src}
        WHERE resolution.primary.human_requested IS NOT NULL
          AND resolution.primary.outcome = 'fulfilled'
    """).fetchone()[0]

    if hr_total > 0:
        print(f"\n  AI conversion rate: {converted}/{hr_total} callers who asked for a human"
              f" were fulfilled by AI ({pct(converted, hr_total)})")

    return hr_total, initial, after


def render_organic_containment(con, json_src):
    """2.2 Organic vs Standard Fulfillment."""
    section("2.2", "ORGANIC vs STANDARD FULFILLMENT")

    stats = con.sql(f"""
        SELECT
            -- Standard fulfillment
            COUNT(*) FILTER (resolution.primary.scope = 'in_scope'
                AND resolution.primary.outcome = 'fulfilled')           AS std_fulfilled,
            COUNT(*) FILTER (resolution.primary.scope = 'in_scope')    AS std_total,
            -- Organic: exclude initial human requests
            COUNT(*) FILTER (resolution.primary.scope = 'in_scope'
                AND resolution.primary.outcome = 'fulfilled'
                AND (resolution.primary.human_requested IS NULL
                     OR resolution.primary.human_requested != 'initial')) AS org_fulfilled,
            COUNT(*) FILTER (resolution.primary.scope = 'in_scope'
                AND (resolution.primary.human_requested IS NULL
                     OR resolution.primary.human_requested != 'initial')) AS org_total
        FROM {json_src}
    """).fetchone()

    std_f, std_t, org_f, org_t = stats

    print(f"  Standard fulfillment:  {pct(std_f, std_t):>5s}  ({std_f}/{std_t} in-scope)")
    print(f"  Organic fulfillment:   {pct(org_f, org_t):>5s}  ({org_f}/{org_t} in-scope, excl initial human requests)")

    if std_t > 0 and org_t > 0 and std_t != org_t:
        gap = (org_f / org_t if org_t else 0) - (std_f / std_t if std_t else 0)
        print(f"\n  Gap: {gap:+.0%} — {'organic higher (initial requests drag down fulfillment)' if gap > 0 else 'standard higher (some initial human requests still get fulfilled)'}")


def render_scope_x_human(con, json_src):
    """2.3 Scope x Human Request Cross-Tab."""
    section("2.3", "SCOPE × HUMAN REQUEST")

    rows = con.sql(f"""
        SELECT
            resolution.primary.scope AS scope,
            COALESCE(resolution.primary.human_requested, 'none') AS hr,
            COUNT(*) AS n
        FROM {json_src}
        GROUP BY scope, hr
        ORDER BY scope, hr
    """).fetchall()

    hr_vals = ["none", "initial", "after_service"]
    scopes = ["in_scope", "out_of_scope", "no_request"]
    grid = {s: {h: 0 for h in hr_vals} for s in scopes}
    for scope, hr, n in rows:
        grid.setdefault(scope, {h: 0 for h in hr_vals})[hr] = n

    hdr = f"  {'':16s}"
    for h in hr_vals:
        hdr += f"{h:>16s}"
    hdr += f"  {'TOTAL':>6s}"
    print(hdr)
    print("  " + "─" * (W - 4))

    for s in scopes:
        row_total = sum(grid[s].values())
        if row_total == 0:
            continue
        line = f"  {s:<16s}"
        for h in hr_vals:
            n = grid[s][h]
            if n == 0:
                line += f"{'—':>16s}"
            else:
                line += f"{n:>16d}"
        line += f"  {row_total:>6d}"
        print(line)

    in_initial = grid.get("in_scope", {}).get("initial", 0)
    in_after = grid.get("in_scope", {}).get("after_service", 0)
    if in_initial > 0:
        print(f"\n  → {in_initial} in-scope + initial = missed containment (caller chose human before AI tried)")
    if in_after > 0:
        print(f"  → {in_after} in-scope + after_service = preventable escalation (AI started but lost them)")


def render_department_requests(con, json_src):
    """2.4 Department Requests — generic vs specific by scope."""
    section("2.4", "DEPARTMENT REQUESTS")

    rows = con.sql(f"""
        SELECT
            resolution.primary.scope AS scope,
            CASE
                WHEN resolution.primary.human_requested IS NULL THEN NULL
                WHEN resolution.primary.department_requested IS NOT NULL
                    THEN resolution.primary.department_requested
                ELSE '(generic rep)'
            END AS dept,
            COUNT(*) AS n
        FROM {json_src}
        WHERE resolution.primary.human_requested IS NOT NULL
        GROUP BY scope, dept
        ORDER BY n DESC
    """).fetchall()

    if not rows:
        print("  No human requests in dataset.")
        return

    print(f"  {'department':<30s} {'scope':<16s} {'n':>4s}")
    print("  " + "─" * (W - 4))
    for scope, dept, n in rows:
        print(f"  {trunc(dept or '(generic rep)', 30):<30s} {scope:<16s} {n:>4d}")

    generic = sum(n for _, dept, n in rows if dept == "(generic rep)")
    specific = sum(n for _, dept, n in rows if dept and dept != "(generic rep)")
    if generic > 0 or specific > 0:
        print(f"\n  Generic ('a representative'): {generic}  |  Specific department: {specific}")
        print(f"  → Generic = AI aversion; Specific = genuinely needs another team")


# ---------------------------------------------------------------------------
# Act 3: Quality & Impediments
# ---------------------------------------------------------------------------

def render_impediments(con, json_src):
    """3.1 Impediments by Scope + Agent Issues."""
    imp_total = con.sql(f"""
        SELECT COUNT(*) FROM {json_src} WHERE impediment IS NOT NULL
    """).fetchone()[0]

    ai_total = con.sql(f"""
        SELECT COUNT(*) FROM {json_src} WHERE agent_issue IS NOT NULL
    """).fetchone()[0]

    section("3.1", f"IMPEDIMENTS BY SCOPE ({imp_total} calls) + AGENT ISSUES ({ai_total} calls)")

    if imp_total == 0 and ai_total == 0:
        print("  No impediments or agent issues detected.")
        return imp_total

    if imp_total > 0:
        rows = con.sql(f"""
            SELECT
                impediment.type AS imp_type,
                resolution.primary.scope AS scope,
                COUNT(*) AS n,
                LIST(insights.coaching) FILTER (insights.coaching IS NOT NULL) AS coaching
            FROM {json_src}
            WHERE impediment IS NOT NULL
            GROUP BY imp_type, scope
            ORDER BY n DESC
        """).fetchall()

        # Aggregate
        by_type = {}
        for itype, scope, n, coaching in rows:
            if itype not in by_type:
                by_type[itype] = {"total": 0, "scopes": {}, "coaching": []}
            by_type[itype]["total"] += n
            by_type[itype]["scopes"][scope] = n
            if coaching:
                by_type[itype]["coaching"].extend(coaching)

        scopes = ["in_scope", "out_of_scope", "no_request"]
        active_scopes = [s for s in scopes if any(d["scopes"].get(s, 0) for d in by_type.values())]

        hdr = f"  {'impediment_type':<20s}"
        for s in active_scopes:
            hdr += f"{s:>14s}"
        hdr += f"  {'TOTAL':>5s}  {'pct':>4s}"
        print(hdr)
        print("  " + "─" * (W - 4))

        for itype, data in sorted(by_type.items(), key=lambda x: -x[1]["total"]):
            line = f"  {itype:<20s}"
            for s in active_scopes:
                v = data["scopes"].get(s, 0)
                line += f"{v:>14d}" if v else f"{'—':>14s}"
            line += f"  {data['total']:>5d}  {pct(data['total'], imp_total):>4s}"
            print(line)
            # Show one coaching sample
            if data["coaching"]:
                sample = trunc(data["coaching"][0].replace("\n", " "), 65)
                print(f"    ↳ {sample}")

    if ai_total > 0:
        print(f"\n  Agent issues ({ai_total} calls):")
        ai_rows = con.sql(f"""
            SELECT
                resolution.primary.scope AS scope,
                COUNT(*) AS n,
                LIST(agent_issue.detail) AS details
            FROM {json_src}
            WHERE agent_issue IS NOT NULL
            GROUP BY scope
            ORDER BY n DESC
        """).fetchall()

        for scope, n, details in ai_rows:
            print(f"    {scope}: {n}")
            if details:
                sample = trunc(details[0].replace("\n", " "), 65)
                print(f"      ↳ {sample}")

    # Overlap
    both = con.sql(f"""
        SELECT COUNT(*) FROM {json_src}
        WHERE impediment IS NOT NULL AND agent_issue IS NOT NULL
    """).fetchone()[0]
    if both > 0:
        print(f"\n  → {both} calls have BOTH impediment + agent issue")

    return imp_total


def render_failure_modes_compat(con, json_src):
    """3.1 (compat) Failure modes for v6.0 schema data (has failure, not impediment)."""
    fail_total = con.sql(f"""
        SELECT COUNT(*) FROM {json_src} WHERE failure IS NOT NULL
    """).fetchone()[0]

    section("3.1", f"FAILURE MODES BY SCOPE ({fail_total} calls) [v6.0 compat]")

    if fail_total == 0:
        print("  No failures detected.")
        return

    rows = con.sql(f"""
        SELECT
            failure.type AS ftype,
            resolution.primary.scope AS scope,
            COUNT(*) AS n
        FROM {json_src}
        WHERE failure IS NOT NULL
        GROUP BY ftype, scope
        ORDER BY n DESC
    """).fetchall()

    by_type = {}
    for ftype, scope, n in rows:
        if ftype not in by_type:
            by_type[ftype] = {"total": 0, "scopes": {}}
        by_type[ftype]["total"] += n
        by_type[ftype]["scopes"][scope] = n

    scopes = ["in_scope", "out_of_scope", "no_request"]
    active_scopes = [s for s in scopes if any(d["scopes"].get(s, 0) for d in by_type.values())]

    hdr = f"  {'failure_type':<20s}"
    for s in active_scopes:
        hdr += f"{s:>14s}"
    hdr += f"  {'TOTAL':>5s}  {'pct':>4s}"
    print(hdr)
    print("  " + "─" * (W - 4))

    for ftype, data in sorted(by_type.items(), key=lambda x: -x[1]["total"]):
        line = f"  {ftype:<20s}"
        for s in active_scopes:
            v = data["scopes"].get(s, 0)
            line += f"{v:>14d}" if v else f"{'—':>14s}"
        line += f"  {data['total']:>5d}  {pct(data['total'], fail_total):>4s}"
        print(line)

    print(f"\n  [Using v6.0 failure model — upgrade to v7.0 for impediment/agent_issue detail]")


def render_preventable_escalations(con, json_src):
    """3.2 Preventable Escalations — in-scope + after_service detail."""
    section("3.2", "PREVENTABLE ESCALATIONS (in-scope + after_service)")

    rows = con.sql(f"""
        SELECT
            resolution.primary.request AS request,
            resolution.primary.outcome AS outcome,
            COALESCE(impediment.type, '—') AS imp_type,
            CASE WHEN agent_issue IS NOT NULL THEN 'yes' ELSE '—' END AS has_ai,
            insights.coaching AS coaching,
            call_id
        FROM {json_src}
        WHERE resolution.primary.scope = 'in_scope'
          AND resolution.primary.human_requested = 'after_service'
        ORDER BY outcome, request
    """).fetchall()

    if not rows:
        print("  No preventable escalations found.")
        return

    print(f"  {len(rows)} calls where AI started in-scope service but caller asked for a human:\n")
    print(f"  {'request':<32s} {'outcome':<12s} {'impediment':14s} {'agent_issue':>11s}  {'coaching'}")
    print("  " + "─" * (W - 4))
    for request, outcome, imp_type, has_ai, coaching, cid in rows:
        c = trunc(coaching or "—", 35) if coaching else "—"
        print(f"  {trunc(request, 32):<32s} {outcome:<12s} {imp_type:14s} {has_ai:>11s}  {c}")


def render_quality_scores(con, json_src):
    """3.3 Quality Scores by Scope x Outcome."""
    section("3.3", "QUALITY SCORES BY SCOPE × OUTCOME")

    rows = con.sql(f"""
        SELECT
            resolution.primary.scope AS scope,
            resolution.primary.outcome AS outcome,
            COUNT(*) AS n,
            ROUND(AVG(scores.effectiveness), 1) AS eff,
            ROUND(AVG(scores.quality), 1) AS qual,
            ROUND(AVG(scores.effort), 1) AS effort
        FROM {json_src}
        GROUP BY scope, outcome
        ORDER BY scope, outcome
    """).fetchall()

    print(f"  {'scope':<16s} {'outcome':<13s} {'n':>4s}  {'eff':>4s} {'qual':>4s} {'effort':>6s}")
    print("  " + "─" * (W - 4))
    for scope, outcome, n, eff, qual, effort in rows:
        effort_flag = " ⚠" if effort and effort >= 3.0 else ""
        print(f"  {scope:<16s} {outcome:<13s} {n:>4d}  {eff:>4.1f} {qual:>4.1f} {effort:>5.1f}{effort_flag}")

    print("\n  (effort: 1=effortless, 5=painful; flag at >=3.0)")


def render_sentiment(con, json_src, total):
    """3.4 Sentiment Journey — start → end."""
    section("3.4", "SENTIMENT JOURNEY")

    rows = con.sql(f"""
        SELECT
            sentiment.start AS s_start,
            sentiment.end AS s_end,
            COUNT(*) AS n
        FROM {json_src}
        GROUP BY s_start, s_end
        ORDER BY n DESC
    """).fetchall()

    print(f"  {'journey':<30s} {'n':>4s}  {'pct':>5s}  {'bar'}")
    print("  " + "─" * (W - 4))
    for s_start, s_end, n in rows:
        journey = f"{s_start} → {s_end}"
        print(f"  {journey:<30s} {n:>4d}  {pct(n, total):>5s}  {bar(n, total, 20)}")

    # Highlight negative trajectories
    worse = sum(n for s, e, n in rows
                if s in ("positive", "neutral") and e in ("frustrated", "angry"))
    if worse > 0:
        print(f"\n  → {worse} calls ({pct(worse, total)}) ended worse than they started")


# ---------------------------------------------------------------------------
# Act 4: Operational Details
# ---------------------------------------------------------------------------

def render_duration(con, json_src):
    """4.1 Duration by Scope x Outcome."""
    section("4.1", "DURATION BY SCOPE × OUTCOME (seconds)")

    rows = con.sql(f"""
        SELECT
            resolution.primary.scope AS scope,
            resolution.primary.outcome AS outcome,
            COUNT(*) AS n,
            ROUND(PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY duration_seconds), 0) AS p50,
            ROUND(PERCENTILE_CONT(0.8) WITHIN GROUP (ORDER BY duration_seconds), 0) AS p80,
            ROUND(AVG(duration_seconds), 0) AS mean,
            ROUND(MAX(duration_seconds), 0) AS max_dur
        FROM {json_src}
        WHERE duration_seconds IS NOT NULL
        GROUP BY scope, outcome
        ORDER BY scope, outcome
    """).fetchall()

    print(f"  {'scope':<16s} {'outcome':<13s} {'n':>4s}  {'p50':>5s} {'p80':>5s} {'mean':>5s} {'max':>6s}")
    print("  " + "─" * (W - 4))
    for scope, outcome, n, p50, p80, mean, max_d in rows:
        print(f"  {scope:<16s} {outcome:<13s} {n:>4d}  {p50:>5.0f} {p80:>5.0f} {mean:>5.0f} {max_d:>6.0f}")


def render_action_performance(con, json_src):
    """4.2 Action Performance — success rate by type."""
    section("4.2", "ACTION PERFORMANCE")

    rows = con.sql(f"""
        WITH acts AS (
            SELECT UNNEST(actions) AS a FROM {json_src}
        )
        SELECT
            a.type AS action_type,
            COUNT(*) AS total,
            COUNT(*) FILTER (a.outcome = 'success') AS success,
            COUNT(*) FILTER (a.outcome = 'failed') AS failed,
            COUNT(*) FILTER (a.outcome = 'retry') AS retry
        FROM acts
        GROUP BY action_type
        ORDER BY total DESC
    """).fetchall()

    if not rows:
        print("  No actions recorded.")
        return

    print(f"  {'action':<24s} {'total':>5s}  {'success':>7s}  {'rate':>5s}  {'failed':>6s}  {'retry':>5s}")
    print("  " + "─" * (W - 4))
    for atype, total, success, failed, retry in rows:
        rate = pct(success, total)
        flag = " ⚠" if total > 0 and success / total < 0.90 else ""
        print(f"  {atype:<24s} {total:>5d}  {success:>7d}  {rate:>5s}{flag}  {failed:>6d}  {retry:>5d}")

    print("\n  (flag at <90% success rate)")


def render_transfer_analysis(con, json_src):
    """4.3 Transfer Analysis — reason x destination x queue."""
    section("4.3", "TRANSFER ANALYSIS")

    rows = con.sql(f"""
        SELECT
            resolution.primary.transfer.reason AS reason,
            resolution.primary.transfer.destination AS destination,
            resolution.primary.transfer.queue_detected AS queue,
            resolution.primary.outcome AS outcome,
            COUNT(*) AS n
        FROM {json_src}
        WHERE resolution.primary.transfer IS NOT NULL
        GROUP BY reason, destination, queue, outcome
        ORDER BY n DESC
    """).fetchall()

    if not rows:
        print("  No transfers recorded.")
        return

    print(f"  {'reason':<20s} {'destination':<20s} {'queue':>5s} {'outcome':<13s} {'n':>3s}")
    print("  " + "─" * (W - 4))
    for reason, dest, queue, outcome, n in rows:
        q = "yes" if queue else "no"
        print(f"  {reason or '—':<20s} {dest or '—':<20s} {q:>5s} {outcome:<13s} {n:>3d}")

    # Also check secondary transfers
    sec_rows = con.sql(f"""
        SELECT COUNT(*) FROM {json_src}
        WHERE resolution.secondary IS NOT NULL
          AND resolution.secondary.transfer IS NOT NULL
    """).fetchone()[0]
    if sec_rows > 0:
        print(f"\n  + {sec_rows} secondary intent transfers (not shown above)")

    # Queue + abandoned = staffing problem
    queue_abandoned = sum(n for reason, dest, queue, outcome, n in rows
                         if queue and outcome == "abandoned")
    if queue_abandoned > 0:
        print(f"\n  → {queue_abandoned} transfers with queue detected + abandoned = staffing/wait problem")


def render_queue_performance(con, json_src):
    """4.4 Queue Performance — queue_result distribution."""
    section("4.4", "QUEUE PERFORMANCE")

    # Check if queue_result field exists in data
    if not has_column(con, json_src, "resolution.primary.transfer.queue_result"):
        print("  queue_result field not present in data (pre-v6.1 schema).")
        return

    queue_total = con.sql(f"""
        SELECT COUNT(*) FROM {json_src}
        WHERE resolution.primary.transfer IS NOT NULL
          AND resolution.primary.transfer.queue_detected = true
    """).fetchone()[0]

    if queue_total == 0:
        print("  No transfers with queue detected.")
        return

    rows = con.sql(f"""
        SELECT
            COALESCE(resolution.primary.transfer.queue_result, 'unknown') AS qr,
            COUNT(*) AS n
        FROM {json_src}
        WHERE resolution.primary.transfer IS NOT NULL
          AND resolution.primary.transfer.queue_detected = true
        GROUP BY qr
        ORDER BY n DESC
    """).fetchall()

    print(f"  {queue_total} transfers entered a queue:\n")
    print(f"  {'queue_result':<22s} {'n':>4s}  {'pct':>5s}  {'bar'}")
    print("  " + "─" * (W - 4))
    for qr, n in rows:
        print(f"  {qr:<22s} {n:>4d}  {pct(n, queue_total):>5s}  {bar(n, queue_total, 20)}")

    unavail = sum(n for qr, n in rows if qr == "unavailable")
    abandoned = sum(n for qr, n in rows if qr == "caller_abandoned")
    if unavail + abandoned > 0:
        print(f"\n  → {unavail + abandoned}/{queue_total} queue entries did NOT reach a human"
              f" ({pct(unavail + abandoned, queue_total)})")


def render_friction(con, json_src):
    """4.5 Friction by Scope — clarifications, corrections, loops."""
    section("4.5", "FRICTION BY SCOPE")

    rows = con.sql(f"""
        SELECT
            resolution.primary.scope AS scope,
            COUNT(*) AS n,
            ROUND(AVG(LEN(friction.clarifications)), 1) AS avg_clar,
            ROUND(AVG(LEN(friction.corrections)), 1) AS avg_corr,
            ROUND(AVG(LEN(friction.loops)), 1) AS avg_loops,
            ROUND(AVG(scores.effort), 1) AS avg_effort,
            SUM(LEN(friction.clarifications)) AS total_clar,
            SUM(LEN(friction.corrections)) AS total_corr,
            SUM(LEN(friction.loops)) AS total_loops
        FROM {json_src}
        GROUP BY scope
        ORDER BY scope
    """).fetchall()

    print(f"  {'scope':<16s} {'n':>4s}  {'clar':>5s} {'corr':>5s} {'loops':>5s}  {'avg_effort':>10s}")
    print("  " + "─" * (W - 4))
    for scope, n, avg_c, avg_co, avg_l, avg_e, tot_c, tot_co, tot_l in rows:
        print(f"  {scope:<16s} {n:>4d}  {tot_c:>5.0f} {tot_co:>5.0f} {tot_l:>5.0f}  {avg_e:>10.1f}")

    total_loops = sum(tot_l for _, _, _, _, _, _, _, _, tot_l in rows)
    if total_loops > 0:
        print(f"\n  → {total_loops:.0f} loops detected — most severe friction (repeated failed attempts)")


def render_abandon_stages(con, json_src):
    """4.6 Abandon Stage Analysis."""
    section("4.6", "ABANDON STAGE ANALYSIS")

    rows = con.sql(f"""
        SELECT
            resolution.primary.scope AS scope,
            resolution.primary.abandon_stage AS stage,
            COUNT(*) AS n
        FROM {json_src}
        WHERE resolution.primary.outcome = 'abandoned'
          AND resolution.primary.abandon_stage IS NOT NULL
        GROUP BY scope, stage
        ORDER BY scope, n DESC
    """).fetchall()

    if not rows:
        print("  No abandoned calls with stage data.")
        return

    print(f"  {'scope':<16s} {'stage':<16s} {'n':>4s}")
    print("  " + "─" * (W - 4))
    for scope, stage, n in rows:
        flag = " ← CRITICAL" if scope == "in_scope" and stage == "mid_task" else ""
        print(f"  {scope:<16s} {stage:<16s} {n:>4d}{flag}")


def render_secondary_intents(con, json_src, total):
    """4.7 Secondary Intents."""
    section("4.7", "SECONDARY INTENTS")

    sec_count = con.sql(f"""
        SELECT COUNT(*) FROM {json_src}
        WHERE resolution.secondary IS NOT NULL
    """).fetchone()[0]

    print(f"  Calls with secondary intent: {sec_count} / {total} ({pct(sec_count, total)})")

    if sec_count == 0:
        return

    rows = con.sql(f"""
        SELECT
            resolution.secondary.scope AS scope,
            resolution.secondary.outcome AS outcome,
            COUNT(*) AS n
        FROM {json_src}
        WHERE resolution.secondary IS NOT NULL
        GROUP BY scope, outcome
        ORDER BY n DESC
    """).fetchall()

    print(f"\n  {'scope':<16s} {'outcome':<13s} {'n':>4s}")
    print("  " + "─" * (W - 4))
    for scope, outcome, n in rows:
        print(f"  {scope:<16s} {outcome:<13s} {n:>4d}")


def render_operational_flags(con, json_src, total):
    """4.8 Operational Flags — repeat callers, derailment."""
    section("4.8", "OPERATIONAL FLAGS")

    flags = con.sql(f"""
        SELECT
            COUNT(*) FILTER (repeat_caller = true) AS repeats,
            COUNT(*) FILTER (derailed_at IS NOT NULL) AS derailed,
            ROUND(AVG(derailed_at) FILTER (derailed_at IS NOT NULL), 0) AS avg_derail_turn,
            ROUND(AVG(turns), 1) AS avg_turns
        FROM {json_src}
    """).fetchone()

    repeats, derailed, avg_derail, avg_turns = flags

    print(f"  Repeat callers:    {repeats:>4d} / {total}  ({pct(repeats, total)})")
    print(f"  Derailed calls:    {derailed:>4d} / {total}  ({pct(derailed, total)})")
    if derailed > 0:
        print(f"  Avg derailment at: turn {avg_derail:.0f}")
    print(f"  Avg turns/call:    {avg_turns:.1f}")


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run_dashboard(directory: str) -> None:
    con = duckdb.connect()
    # Load data ONCE into a temp table so all sections see the same snapshot.
    # Re-scanning the glob per query can produce slightly different row counts.
    raw_src = src(directory)
    con.sql(f"CREATE TEMP TABLE calls AS SELECT * FROM {raw_src}")
    json_src = "calls"

    # Detect field availability for backward compatibility
    has_hr = has_column(con, json_src, "resolution.primary.human_requested")
    has_imp = has_column(con, json_src, "impediment")

    # === ACT 1 ===
    act_header("ACT 1: THE BIG PICTURE")
    total = render_header_kpis(con, json_src, directory)
    render_call_funnel(con, json_src, total)
    render_scope_outcome(con, json_src, total)
    render_top_requests(con, json_src)

    # === ACT 2 (v5+ only) ===
    if has_hr:
        act_header("ACT 2: THE HUMAN-REQUEST PHENOMENON")
        render_human_overview(con, json_src, total)
        render_organic_containment(con, json_src)
        render_scope_x_human(con, json_src)
        render_department_requests(con, json_src)
    else:
        print("\n  [Act 2 skipped — human_requested field not present in data]")

    # === ACT 3 ===
    if has_imp:
        act_header("ACT 3: QUALITY & IMPEDIMENTS")
        render_impediments(con, json_src)
        if has_hr:
            render_preventable_escalations(con, json_src)
    else:
        act_header("ACT 3: QUALITY & FAILURE (v6.0 compat)")
        render_failure_modes_compat(con, json_src)
    render_quality_scores(con, json_src)
    render_sentiment(con, json_src, total)

    # === ACT 4 ===
    act_header("ACT 4: OPERATIONAL DETAILS")
    render_duration(con, json_src)
    render_action_performance(con, json_src)
    render_transfer_analysis(con, json_src)
    render_queue_performance(con, json_src)
    render_friction(con, json_src)
    render_abandon_stages(con, json_src)
    render_secondary_intents(con, json_src, total)
    render_operational_flags(con, json_src, total)

    print()
    print("  " + "=" * (W - 4))
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 tools/dashboard_v7.py <directory-of-analysis-jsons>")
        print("  e.g. python3 tools/dashboard_v7.py tests/golden/analyses_v7/")
        sys.exit(1)

    directory = sys.argv[1].rstrip("/")
    path = Path(directory)
    if not path.is_dir():
        print(f"Error: {directory} is not a directory")
        sys.exit(1)

    json_count = len(list(path.glob("*.json")))
    if json_count == 0:
        print(f"Error: no .json files found in {directory}")
        sys.exit(1)

    run_dashboard(directory)


if __name__ == "__main__":
    main()
