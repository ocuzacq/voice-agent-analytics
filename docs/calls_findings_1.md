# Call Findings Report #1 — v5.0 Validation Run (200 calls)

**Date**: 2026-02-06
**Schema**: v5.0 (orthogonal disposition model)
**Sample**: 200 random transcripts from ~5,800 corpus
**Model**: gemini-3-flash-preview (analysis), v5.0 prompt

---

## 1. Escalation Dominance

**68.5% of calls escalate.** Only 16% complete successfully.

| Outcome | Count | Rate |
|---------|-------|------|
| Escalated | 137 | 68.5% |
| Completed | 32 | 16.0% |
| Abandoned | 31 | 15.5% |

### Escalation Trigger Breakdown (n=137)

| Trigger | Count | % of Escalations |
|---------|-------|-------------------|
| scope_limit | 61 | 44.5% |
| customer_requested | 49 | 35.8% |
| task_failure | 22 | 16.1% |
| policy_routing | 5 | 3.6% |

**Interpretation**: Nearly half of escalations occur because the AI agent hits the boundary of its capabilities (`scope_limit`). Another third are customer-driven — callers who want a human regardless of what the AI can do. Only 16% are genuine task failures where the agent attempted something and couldn't deliver.

**Actionable**: The `scope_limit` escalations represent the highest-leverage improvement opportunity. Expanding the agent's capability envelope (even for a few common request types) would directly reduce escalation volume.

---

## 2. Containment Rate: 20.5%

Of the 122 calls classified as in-scope or mixed, only 25 completed without escalation.

```
In-Scope + Mixed: 122 calls
  Completed: 25 (20.5%)
  Escalated: 79 (64.8%)
  Abandoned: 18 (14.8%)
```

This is the core operational KPI. The agent resolves roughly 1 in 5 calls where it theoretically could help.

**Why so low?** The escalation trigger data tells the story:
- 45 of 79 in-scope escalations are `customer_requested` — callers who don't want AI help
- 22 are `task_failure` — agent tried and failed (verification issues, link delivery problems)
- 7 are `scope_limit` — sub-tasks within an in-scope call that the agent can't handle
- 5 are `policy_routing` — business rules requiring human involvement

**Structural ceiling**: Even if every `task_failure` and `scope_limit` escalation were fixed, containment would only reach ~44% because `customer_requested` escalations are demand-side (the caller's choice, not a system limitation).

---

## 3. Scope Distribution

| Scope | Count | Rate |
|-------|-------|------|
| in_scope | 100 | 50.0% |
| out_of_scope | 72 | 36.0% |
| mixed | 22 | 11.0% |
| no_request | 6 | 3.0% |

**36% of calls are out-of-scope.** These are callers asking for things the agent was never designed to handle (reservations, ownership transfers, membership renewals, etc.). Combined with the 11% mixed calls (partially in-scope), nearly half the call volume involves requests outside the agent's defined capabilities.

**Out-of-scope outcome breakdown:**
- 58 escalated (81%) — correctly routed to humans
- 7 completed (10%) — agent provided general info even though request was out-of-scope
- 7 abandoned (10%) — caller gave up

---

## 4. Resolution Confirmation Gap

Of 32 completed calls, only 13 (40.6%) have confirmed resolution.

| resolution_confirmed | Count | Rate |
|---------------------|-------|------|
| true | 13 | 40.6% |
| false | 19 | 59.4% |

This means in 59% of "completed" calls, the agent took action (sent a link, provided info) but the customer never explicitly acknowledged receipt or satisfaction on the call. These could be genuine completions where the customer simply hung up satisfied, or they could be premature closures.

**Implication**: The unconfirmed completions are a quality risk. A follow-up analysis correlating `resolution_confirmed=false` with `sentiment_end` and `effort` scores could reveal whether these are actually successful.

---

## 5. Transfer Quality

**153 transfers occurred** across 200 calls (76.5% transfer rate).

| Destination | Count | Rate |
|-------------|-------|------|
| concierge | 150 | 98.0% |
| specific_department | 3 | 2.0% |

**No routing intelligence.** Virtually every transfer goes to the same generic concierge queue. The agent doesn't differentiate between a billing question, a maintenance request, or a reservation inquiry when transferring.

### Queue Detection

**124 of 153 transfers (81%) show queue evidence** in the transcript — hold messages, "all agents are currently busy" loops, silent assistant entries during wait.

This flag (`transfer_queue_detected`) is an LLM observation from transcript content. It's not derivable from structured fields — only the raw conversation reveals whether a queue was encountered. Correlated signals:
- Escalated calls average **1,087s** handle time vs **446s** for completed calls — the delta is largely queue wait
- Calls with `ended_by=customer` and `call_outcome=escalated` likely represent queue abandonment (customer hung up waiting)

**Implication**: 4 out of 5 customers who get transferred sit in a queue. Combined with the 98% single-destination routing, this suggests the transfer experience is undifferentiated and often involves significant wait time.

---

## 6. Action Performance

403 actions tracked across 200 calls (2.0 actions/call average, 187/200 calls have at least one action).

| Action | Attempted | Success Rate |
|--------|-----------|-------------|
| transfer | 163 | 82% |
| account_lookup | 98 | 82% |
| verification | 82 | 90% |
| send_payment_link | 25 | 56% |
| other | 21 | 76% |
| send_clubhouse_link | 6 | 50% |
| send_rental_link | 4 | 100% |
| send_portal_link | 3 | 67% |
| send_rci_link | 1 | 100% |

**Payment link delivery is unreliable at 56% success.** This is the most common in-scope action after account lookup and verification. Failed payment links directly cause `task_failure` escalations and `mid_task` abandonments.

**Transfer success at 82%** means 18% of transfer attempts fail — the call either drops, the customer hangs up during transfer, or the routing doesn't connect.

---

## 7. Abandonment Patterns

31 calls abandoned (15.5%), distributed across stages:

| Stage | Count | Rate |
|-------|-------|------|
| mid_task | 20 | 64.5% |
| pre_intent | 8 | 25.8% |
| post_delivery | 3 | 9.7% |

**Most abandonments happen mid-task** — the customer was engaged, the agent was working on their request, but the caller gave up. This correlates with friction (verification loops, slow processes, failed actions).

**Pre-intent abandonments** (8) are callers who disconnected before stating any request — silence, hangups during greeting, or misdials.

**Post-delivery abandonments** (3) are the most interesting — the agent completed an action but the customer still hung up. These may be cases where the "delivery" wasn't actually what the customer needed.

---

## 8. Sentiment Trajectory

| Sentiment | Start | End |
|-----------|-------|-----|
| neutral | 95.5% | 78.5% |
| frustrated | 4.5% | 15.5% |
| satisfied | 0% | 5.0% |
| angry | 0% | 1.0% |

**14.5% of calls end worse than they started.** Frustration triples from 4.5% to 15.5%. Only 5% end satisfied — a low bar for any customer service channel.

The frustration increase correlates with escalation and queue waits. Callers who start neutral and get transferred to a queue tend to end frustrated.

---

## Summary of Actionable Priorities

1. **Expand agent capabilities** to reduce `scope_limit` escalations (44.5% of all escalations). Even handling 2-3 more common request types (reservations, ownership inquiries) could meaningfully reduce transfer volume.

2. **Fix payment link delivery** (56% success rate). This is the primary in-scope action that fails and directly drives task_failure escalations.

3. **Add transfer routing intelligence**. 98% of transfers go to one queue. Routing to specialized queues based on intent would reduce customer wait and improve resolution speed.

4. **Investigate unconfirmed completions**. 59% of "completed" calls lack customer confirmation. Correlate with sentiment and effort to determine if these are genuine successes or false positives.

5. **Reduce mid-task abandonment**. 64.5% of abandonments happen during active task work. Root causes likely include friction (verification loops, slow responses) and failed actions.
