def assess_readiness(analysis: dict, leakage: dict, score: dict) -> dict:
    """
    Convert technical verification results into business-friendly readiness decisions.
    """

    trust_score = score.get("trust_score", 0)
    leakage_count = leakage.get("risk_count", 0)

    imbalance = analysis.get("class_imbalance", {})
    is_imbalanced = imbalance.get("is_imbalanced", False)

    profile = analysis.get("profile", {})
    missing_values = profile.get("missing_values_ratio", {})
    max_missing = max(missing_values.values()) if missing_values else 0

    reasons = []
    next_steps = []

    if trust_score >= 85:
        ml_readiness = "High"
    elif trust_score >= 70:
        ml_readiness = "Medium"
    elif trust_score >= 50:
        ml_readiness = "Low"
    else:
        ml_readiness = "Not Ready"

    if trust_score >= 85 and leakage_count == 0:
        business_decision_readiness = "Medium"
    else:
        business_decision_readiness = "Low"

    if is_imbalanced:
        reasons.append("Severe class imbalance detected.")
        next_steps.append("Use stratified split, resampling, class weights, and recall/F1-focused evaluation.")

    if leakage_count > 0:
        reasons.append("Possible target leakage detected.")
        next_steps.append("Review suspicious columns before training any downstream model.")

    if max_missing > 0.1:
        reasons.append("Meaningful missing values detected.")
        next_steps.append("Apply missing value treatment and document the imputation strategy.")

    if not reasons:
        reasons.append("No major blocking issue detected.")
        next_steps.append("Proceed with model experimentation, but validate before production use.")

    should_train_model = trust_score >= 70 and leakage_count == 0

    return {
        "ml_readiness": ml_readiness,
        "business_decision_readiness": business_decision_readiness,
        "should_train_model": should_train_model,
        "main_reasons": reasons,
        "recommended_next_steps": next_steps
    }
