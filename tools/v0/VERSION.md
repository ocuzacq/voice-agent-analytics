# Version 0 (v0) - Simple Schema

**Date:** 2026-01-18
**Status:** Archived

## Overview

Original implementation with basic analysis schema focused on:
- Funnel analysis (connect → intent → verification → resolution)
- Coverage classification (PRE-INTENT / IN-SCOPE / OUT-OF-SCOPE)
- Outcome tracking (SUCCESS / FAILURE / UNKNOWN)
- Issue detection and categorization
- Action claim vs completion tracking

## Schema Fields

```json
{
  "call_id": "uuid",
  "file_path": "path",
  "funnel": {
    "connect_greet": true,
    "intent_captured": true,
    "intent_description": "string",
    "verification_attempted": true,
    "verification_successful": false,
    "solution_path_entered": false,
    "closure_type": "timeout"
  },
  "coverage": "IN-SCOPE",
  "outcome": "FAILURE",
  "actions_claimed": [],
  "actions_with_completion_evidence": [],
  "issues": [],
  "human_escalation_requested": false,
  "human_escalation_honored": null,
  "call_duration_proxy": 100,
  "turn_count": 50,
  "summary": "..."
}
```

## Metrics

- **ICR** (Intent Capture Rate)
- **ISSR** (In-Scope Success Rate)
- **ACR** (Action Completion Rate)
- Funnel drop-off rates
- Issue frequency by type/severity
- Verification success rate
- Escalation honor rate

## Usage

```bash
# From project root
python3 tools/v0/sample_transcripts.py -n 20
python3 tools/v0/batch_analyze.py
python3 tools/v0/compute_metrics.py
```

## Limitations

- No performance scoring (efficiency, satisfaction, resolution)
- No agent quality metrics (NLU, tone, conversational flow)
- No customer profile analysis
- Basic binary/categorical metrics only
