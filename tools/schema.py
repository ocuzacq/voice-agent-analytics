"""
Voice Agent Call Analysis Schema — v6.0

Canonical Pydantic models for structured LLM output via Gemini response_schema.

Design principles:
  - Per-intent resolution: each intent gets its own scope + outcome + transfer
  - Explicit primary/secondary: no implicit list ordering
  - Transfer consolidated: reason + destination live on the intent, not scattered
  - "fulfilled" not "completed": unambiguous — the customer's need was met
  - "transferred" not "escalated": neutral — correct routing, not failure
  - Optional = nullable: None means "not applicable", not "missing"
  - Enums via Literal: API-enforced, no free-text leakage

Architecture:
  CallAnalysis  — what the LLM produces (used as Gemini response_schema)
  CallRecord    — complete record = CallAnalysis + deterministic metadata
                  (call_id, schema_version, duration_seconds, ended_by)

Serialization:
  CallRecord.model_dump() → JSON on disk (what gets saved to analyses/)
  CallAnalysis alone     → Gemini response_schema parameter
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


SCHEMA_VERSION = "v6.0"


# ---------------------------------------------------------------------------
# Enums as type aliases (for readability + reuse)
# ---------------------------------------------------------------------------

IntentScope = Literal["in_scope", "out_of_scope", "no_request"]
IntentOutcome = Literal["fulfilled", "transferred", "abandoned"]

HumanRequestTiming = Literal["initial", "after_service"]

TransferReason = Literal["customer_requested", "out_of_scope", "task_failure"]
TransferDestination = Literal["concierge", "specific_department", "ivr", "unknown"]

AbandonStage = Literal["pre_greeting", "pre_intent", "mid_task", "post_delivery"]

FailureType = Literal[
    "nlu_miss", "wrong_action", "policy_gap",
    "customer_confusion", "tech_issue", "other",
]

# Sentiment values are intentionally asymmetric:
#   start: "positive" = customer enters in good mood (before any service)
#   end:   "satisfied" = explicitly pleased with outcome (earned, not innate)
SentimentStart = Literal["positive", "neutral", "frustrated", "angry"]
SentimentEnd = Literal["satisfied", "neutral", "frustrated", "angry"]

ActionType = Literal[
    "account_lookup", "verification",
    "send_payment_link", "send_portal_link", "send_autopay_link",
    "send_rental_link", "send_clubhouse_link", "send_rci_link",
    "transfer", "other",
]
ActionOutcome = Literal["success", "failed", "retry", "unknown"]

ClarificationType = Literal["name", "phone", "intent", "repeat", "verify"]
ClarificationCause = Literal["misheard", "unclear", "refused", "tech", "proactive"]
CorrectionSeverity = Literal["minor", "moderate", "major"]
LoopType = Literal[
    "info_retry", "intent_retry", "deflection", "comprehension", "action_retry"
]

PolicyGapCategory = Literal[
    "capability_limit", "data_access", "auth_restriction",
    "business_rule", "integration_missing",
]

EndedBy = Literal["agent", "customer", "error", "unknown"]


# ---------------------------------------------------------------------------
# Resolution sub-models (per-intent)
# ---------------------------------------------------------------------------

class TransferDetail(BaseModel):
    """Where and why a transfer was attempted. Present for both completed and failed transfers."""
    reason: TransferReason = Field(
        description="customer_requested = caller asked for human; "
                    "out_of_scope = request beyond AI capability; "
                    "task_failure = AI tried but failed"
    )
    destination: TransferDestination
    queue_detected: bool = Field(
        default=False, description="Post-transfer queue content visible in transcript"
    )


class IntentResolution(BaseModel):
    """A customer intent and how it was resolved.

    Each intent independently tracks scope (could the AI handle it?),
    outcome (what happened?), and conditional details (transfer/abandon/confirm).
    """
    request: str = Field(description="What the customer wanted (3-8 words, action verb)")
    context: Optional[str] = Field(None, description="Underlying reason or situation")

    human_requested: Optional[HumanRequestTiming] = Field(
        None,
        description="null = caller did not ask for a human agent; "
                    "initial = caller asked for a human/representative/department BEFORE "
                    "the AI attempted any substantive service (account lookup, link send, "
                    "info delivery — verification/name collection alone is NOT substantive service); "
                    "after_service = caller asked for a human AFTER the AI had begun "
                    "substantive service (indicates preventable escalation)"
    )
    department_requested: Optional[str] = Field(
        None,
        description="Specific department or team the caller asked for by name "
                    "(e.g., 'finance', 'billing', 'accounting', 'sales', 'reservations', 'management'); "
                    "null when caller asked for a generic rep / 'customer service' / 'concierge' "
                    "or did not request a human at all"
    )

    scope: IntentScope = Field(
        description="in_scope = AI has a capability pathway for the UNDERLYING request "
                    "(not the transfer action itself); "
                    "out_of_scope = needs human, OR caller requests a human without stating "
                    "a need the AI can address; "
                    "no_request = caller never articulated a request"
    )
    outcome: IntentOutcome = Field(
        description="fulfilled = customer's need was met (action completed, not just prepared); "
                    "transferred = caller reached or remained in human agent queue (no AI re-engagement after queue); "
                    "abandoned = caller disconnected without resolution"
    )
    detail: str = Field(description="One sentence: what happened for this intent")

    # Conditional qualifiers (transfer can coexist with abandon_stage for failed transfers)
    resolution_confirmed: Optional[bool] = Field(
        None,
        description="Customer explicitly confirmed receipt/completion (fulfilled only); "
                    "true = confirmed, false = action taken but unconfirmed",
    )
    transfer: Optional[TransferDetail] = Field(
        None, description="Where and why a transfer was attempted or completed"
    )
    abandon_stage: Optional[AbandonStage] = Field(
        None, description="When the caller dropped (abandoned only)"
    )


class Resolution(BaseModel):
    """How the call was resolved — per-intent structure.

    Primary intent is always present. Secondary is optional (most calls have one intent).
    This avoids collapsing multi-intent calls into a single misleading label.

    Example: customer wants to pay a fee (in_scope → fulfilled) AND update their
    phone number (out_of_scope → transferred). Both outcomes are independently visible.
    """
    primary: IntentResolution = Field(description="The customer's main reason for calling")
    secondary: Optional[IntentResolution] = Field(
        None, description="Additional request beyond the primary intent (if any)"
    )
    steps: list[str] = Field(
        default_factory=list,
        description="High-level narrative of the call flow (free text); "
                    "use actions[] for structured tool tracking",
    )


# ---------------------------------------------------------------------------
# Quality, sentiment, failure, friction sub-models
# ---------------------------------------------------------------------------

class Scores(BaseModel):
    """Agent and interaction quality ratings (1-5 scale).

    Note: effort uses CES (Customer Effort Score) convention where
    1=effortless and 5=painful — inverted relative to effectiveness/quality.
    """
    effectiveness: int = Field(ge=1, le=5, description="Did agent understand and respond appropriately? (5=best)")
    quality: int = Field(ge=1, le=5, description="Flow, tone, clarity combined (5=best)")
    effort: int = Field(ge=1, le=5, description="How hard did the customer work? (1=effortless, 5=painful)")


class Sentiment(BaseModel):
    """Customer emotional journey from start to end of call."""
    start: SentimentStart
    end: SentimentEnd


class PolicyGap(BaseModel):
    """Structured detail when failure type is 'policy_gap'."""
    category: PolicyGapCategory
    specific_gap: str = Field(description="What specifically couldn't be done")
    customer_ask: str = Field(description="What the customer wanted")
    blocker: str = Field(description="Why it couldn't be fulfilled")


class Failure(BaseModel):
    """What went wrong on this call.

    Present only when the call had a meaningful failure.
    Null for clean fulfilled calls and for correct out-of-scope transfers.

    Key distinction:
      tech_issue = a system component broke (API error, send failure, crash)
      wrong_action = agent took an inappropriate action for the context
    """
    type: FailureType
    detail: str = Field(description="One sentence: what went wrong")
    policy_gap: Optional[PolicyGap] = Field(
        None, description="Required when type='policy_gap'"
    )


class Clarification(BaseModel):
    """Agent asked customer to clarify something."""
    turn: int
    type: ClarificationType
    cause: ClarificationCause = Field(
        description="Why the clarification was needed: misheard/unclear/refused/tech = problem; "
                    "proactive = agent confirming preemptively"
    )
    note: str = Field(description="5-8 words, telegraph style")


class Correction(BaseModel):
    """Customer corrected the agent."""
    turn: int
    severity: CorrectionSeverity
    note: str = Field(description="5-8 words, telegraph style")


class Loop(BaseModel):
    """Agent friction loop (repeated failed attempts, NOT benign repetition)."""
    turns: list[int] = Field(min_length=2, description="Turn numbers involved in the loop (minimum 2)")
    type: LoopType
    subject: str = Field(description="What the loop was about")
    note: str = Field(description="5-8 words, telegraph style")


class Friction(BaseModel):
    """Interaction pain points: clarifications, corrections, and loops."""
    clarifications: list[Clarification] = Field(default_factory=list)
    corrections: list[Correction] = Field(default_factory=list)
    loops: list[Loop] = Field(default_factory=list)


class Action(BaseModel):
    """A distinct tool or action the agent attempted.

    Transfer remains an action type for the attempt log (including failed transfers).
    The canonical transfer outcome lives on IntentResolution.transfer.
    """
    type: ActionType = Field(description="Which tool/action was attempted")
    outcome: ActionOutcome
    detail: str = Field(description="5-10 words, telegraph style")


class Insights(BaseModel):
    """Qualitative takeaways from the call."""
    summary: str = Field(description="1-2 sentences: what customer wanted, what happened, outcome")
    verbatim: Optional[str] = Field(
        None,
        description="Exact customer quote (1-2 sentences) capturing core need or frustration; "
                    "null if none stands out",
    )
    coaching: Optional[str] = Field(
        None, description="What the agent should have done differently; null if agent performed well"
    )


# ---------------------------------------------------------------------------
# Main models
# ---------------------------------------------------------------------------

class CallAnalysis(BaseModel):
    """LLM-produced call analysis — used as Gemini response_schema.

    This model defines exactly what the LLM should output for each transcript.
    Deterministic metadata (call_id, schema_version, duration, ended_by) is
    injected afterward to produce the full CallRecord.
    """
    turns: int = Field(description="Count of customer (user) messages in the transcript")
    repeat_caller: bool = Field(description="Customer mentioned prior calls or ongoing frustration")
    derailed_at: Optional[int] = Field(
        None, description="Turn number where call started going off track; null if call stayed on course"
    )

    resolution: Resolution
    scores: Scores
    sentiment: Sentiment

    failure: Optional[Failure] = Field(
        None, description="What went wrong; null for clean successful calls and correct transfers"
    )
    friction: Friction = Field(default_factory=Friction)

    actions: list[Action] = Field(default_factory=list)

    insights: Insights


class CallRecord(CallAnalysis):
    """Complete analysis record = LLM output + deterministic metadata.

    This is what gets serialized to JSON on disk (analyses/*.json).
    Inherits all fields from CallAnalysis and adds:
      - call_id: from transcript filename
      - schema_version: hardcoded
      - duration_seconds: from transcript metadata
      - ended_by: mapped from transcript ended_reason
    """
    call_id: str = Field(description="UUID from transcript filename")
    schema_version: str = Field(default=SCHEMA_VERSION)
    duration_seconds: Optional[float] = Field(None, description="Call duration from transcript metadata")
    ended_by: EndedBy = Field(description="Who/what ended the call")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

__all__ = [
    "SCHEMA_VERSION",
    "HumanRequestTiming",
    "CallAnalysis", "CallRecord",
    "Resolution", "IntentResolution", "TransferDetail",
    "Scores", "Sentiment",
    "Failure", "PolicyGap", "Friction", "Clarification", "Correction", "Loop",
    "Action", "Insights",
]
