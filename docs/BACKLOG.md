# Analytics Backlog — Deferred Items

Items considered during v4.5 planning but deferred due to platform telemetry requirements, complexity, or lower priority.

## Needs Platform Telemetry

These require data not available in call transcripts alone:

- **Action latency (P50/P90)** — Time between agent initiating action and result appearing in transcript. Requires platform-level timestamps.
- **Transfer timeliness** — Average wait time after transfer, % of transfers exceeding 60s, failed transfers. Requires telephony platform data or fine-grained timestamp analysis.
- **Action error root causes** — Distinguishing API timeout vs invalid data vs system error. Requires platform error codes.

## Needs Cross-Session Linking

- **First Contact Resolution (FCR)** — Requires customer ID linking across sessions to detect repeat contacts for the same issue.

## Analytics Tooling

- **Time-series trending** — Track metric changes over time (daily/weekly). Requires multiple runs with comparable parameters.
- **Segment dimensions** — Break down metrics by time-of-day, property, customer type. Partially available from transcript metadata.

## Schema Refinements

- **Action turn numbers** — Track which conversation turn each action occurred at. Would enable correlation between action timing and call outcomes.
- **Action timestamps** — When available from platform, tie actions to wall-clock time for latency analysis.
