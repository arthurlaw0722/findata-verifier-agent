def calculate_trust_score(analysis: dict, leakage: dict) -> dict:
    score = 100
    penalties = []

    profile = analysis["profile"]

    duplicate_ratio = profile.get("duplicate_ratio", 0)
    if duplicate_ratio > 0.05:
        penalty = 10
        score -= penalty
        penalties.append(f"High duplicate ratio: -{penalty}")

    missing_values = profile.get("missing_values_ratio", {})
    if missing_values:
        max_missing = max(missing_values.values())
        if max_missing > 0.3:
            penalty = 15
            score -= penalty
            penalties.append(f"Severe missing values: -{penalty}")
        elif max_missing > 0.1:
            penalty = 8
            score -= penalty
            penalties.append(f"Moderate missing values: -{penalty}")

    outliers = analysis.get("outliers", {})
    high_outlier_cols = [
        col for col, info in outliers.items()
        if info["outlier_ratio"] > 0.05
    ]
    if high_outlier_cols:
        penalty = min(15, len(high_outlier_cols) * 3)
        score -= penalty
        penalties.append(f"Outlier-heavy columns: -{penalty}")

    imbalance = analysis.get("class_imbalance", {})
    if imbalance.get("is_imbalanced"):
        penalty = 10
        score -= penalty
        penalties.append(f"Class imbalance: -{penalty}")

    leakage_count = leakage.get("risk_count", 0)
    if leakage_count > 0:
        penalty = min(25, leakage_count * 8)
        score -= penalty
        penalties.append(f"Possible target leakage: -{penalty}")

    score = max(score, 0)

    if score >= 85:
        grade = "High Trust"
    elif score >= 70:
        grade = "Medium Trust"
    elif score >= 50:
        grade = "Low Trust"
    else:
        grade = "High Risk"

    return {
        "trust_score": score,
        "trust_grade": grade,
        "penalties": penalties
    }
