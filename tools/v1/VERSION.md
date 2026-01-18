# Version 1 (v1) - Enhanced Schema

**Date:** 2026-01-18
**Status:** Current

## Overview

Enhanced implementation with comprehensive analysis dimensions for drill-down analysis:
- All v0 capabilities (funnel, coverage, outcome, issues, actions)
- **NEW:** Performance scores (efficiency, resolution, satisfaction)
- **NEW:** Agent quality metrics (NLU, response quality, tone, transcription)
- **NEW:** Customer profile analysis (age, tech comfort, emotional states)

## Schema Additions (over v0)

### Performance Scores (1-5 scale)

```json
{
  "performance_scores": {
    "efficiency": {
      "score": 4,
      "rationale": "string",
      "factors": {
        "call_flow_smoothness": 4,
        "unnecessary_repetition": 3,
        "time_to_resolution": 4
      }
    },
    "resolution": {
      "score": 3,
      "rationale": "string",
      "customer_need_met": true,
      "partial_resolution": false,
      "next_steps_clear": true
    },
    "satisfaction_signals": {
      "overall_score": 3,
      "rationale": "string",
      "components": {
        "interaction_quality": 4,
        "outcome_satisfaction": 3,
        "effort_required": 3
      },
      "sentiment_indicators": ["gratitude", "slight_confusion"]
    }
  }
}
```

### Agent Quality Metrics

```json
{
  "agent_quality": {
    "nlu_accuracy": {
      "score": 4,
      "misunderstandings": [],
      "clarification_attempts": 1
    },
    "response_quality": {
      "relevance": 4,
      "helpfulness": 4,
      "verbosity": "appropriate"
    },
    "conversational_flow": {
      "naturalness": 4,
      "turn_taking": 4,
      "context_retention": 3
    },
    "tone_and_style": {
      "professionalism": 5,
      "empathy": 3,
      "patience": 4
    },
    "transcription_quality": {
      "apparent_asr_errors": 2,
      "impact_on_call": "minor"
    }
  }
}
```

### Customer Profile

```json
{
  "customer_profile": {
    "apparent_age_group": "senior",
    "tech_comfort_level": 3,
    "emotional_state_start": "calm",
    "emotional_state_end": "satisfied",
    "communication_clarity": 4
  }
}
```

## New Aggregate Metrics

- Performance score distributions (mean, median, stdev)
- Customer need met rate
- Sentiment indicator frequency
- NLU accuracy across calls
- Response quality trends
- Emotional state transition patterns
- Age group distribution
- Tech comfort correlation with outcomes

## Usage

```bash
# From project root
python3 tools/v1/sample_transcripts.py -n 20
python3 tools/v1/batch_analyze.py
python3 tools/v1/compute_metrics.py
```

## Use Cases

- **Root cause analysis**: Why are IN-SCOPE calls failing?
- **Agent improvement**: Which quality dimensions need work?
- **Customer segmentation**: How do seniors perform vs younger users?
- **Efficiency optimization**: What causes call flow breakdowns?
- **Satisfaction drivers**: What correlates with positive sentiment?
