# FinData Verifier Agent Report

## Dataset
**Name:** Loan Approval Leakage Demo
**Generated at:** 2026-06-21T22:31:12.842432 UTC

## Dataset Overview
- Rows: 20
- Columns: 6
- Duplicate rows: 0
- Duplicate ratio: 0.0

## Trust Score
**Score:** 49/100
**Grade:** High Risk

## Key Risks
- Possible target leakage: -16
- Safety gate: target leakage caps trust score at 49

## Missing Values
- No missing values detected.

## Class Imbalance
- Target column: Class
- Minority class ratio: 0.5
- Is imbalanced: False

## Possible Target Leakage
- approved_status: suspicious_column_name — Column name may reveal outcome-related information.
- approved_status: very_high_correlation_with_target — Feature is almost perfectly correlated with target.

## Verification Proof
- Dataset fingerprint: `b84d3de00031c3b804530c954e4a7a02e7e7ed1dd5d57c3aa7bbe9855091daf0`
- Canonical report hash: `70f941d6ff3284f70cb5069c45bc54a8f2d40250ae1f5c9748798ba7aa637484`
- Execution timestamp: `2026-06-21T22:31:12.842426Z`

## Recommendation
**BLOCKED — Target leakage detected.** Do not use this dataset for downstream model training. Remove or independently validate leakage-related columns, then rerun verification.
