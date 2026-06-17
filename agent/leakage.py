import pandas as pd
import numpy as np


def detect_possible_leakage(df: pd.DataFrame, target_column: str | None) -> dict:
    if not target_column or target_column not in df.columns:
        return {"status": "no_target_column_provided"}

    target = df[target_column]
    findings = []

    suspicious_keywords = [
        "target", "label", "result", "outcome", "status",
        "approved", "default", "fraud", "churn", "survived"
    ]

    for col in df.columns:
        if col == target_column:
            continue

        col_lower = col.lower()

        if any(word in col_lower for word in suspicious_keywords):
            findings.append({
                "column": col,
                "risk_type": "suspicious_column_name",
                "reason": "Column name may reveal outcome-related information."
            })

        if pd.api.types.is_numeric_dtype(df[col]) and pd.api.types.is_numeric_dtype(target):
            corr = df[[col, target_column]].dropna().corr().iloc[0, 1]
            if abs(corr) > 0.95:
                findings.append({
                    "column": col,
                    "risk_type": "very_high_correlation_with_target",
                    "correlation": round(float(corr), 4),
                    "reason": "Feature is almost perfectly correlated with target."
                })

    return {
        "target_column": target_column,
        "leakage_risks": findings,
        "risk_count": len(findings)
    }
