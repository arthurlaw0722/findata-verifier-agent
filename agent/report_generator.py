from datetime import datetime


def generate_markdown_report(
    dataset_name: str,
    analysis: dict,
    leakage: dict,
    score: dict,
    proof: dict
) -> str:
    profile = analysis.get("profile", {})
    imbalance = analysis.get("class_imbalance", {})

    missing_values = profile.get("missing_values_ratio", {})
    penalties = score.get("penalties", [])
    leakage_risks = leakage.get("leakage_risks", [])
    leakage_count = leakage.get("risk_count", 0)

    if leakage_count > 0:
        recommendation = (
            "**BLOCKED — Target leakage detected.** Do not use this dataset "
            "for downstream model training. Remove or independently validate "
            "leakage-related columns, then rerun verification."
        )
    else:
        recommendation = (
            "This dataset can be used for experimentation if the listed risks are addressed. "
            "For production or financial decision-making, further validation is recommended."
        )

    lines = [
        "# FinData Verifier Agent Report",
        "",
        "## Dataset",
        f"**Name:** {dataset_name}",
        f"**Generated at:** {datetime.utcnow().isoformat()} UTC",
        "",
        "## Dataset Overview",
        f"- Rows: {profile.get('rows', 'Unknown')}",
        f"- Columns: {profile.get('columns', 'Unknown')}",
        f"- Duplicate rows: {profile.get('duplicate_rows', 0)}",
        f"- Duplicate ratio: {profile.get('duplicate_ratio', 0)}",
        "",
        "## Trust Score",
        f"**Score:** {score.get('trust_score', 0)}/100",
        f"**Grade:** {score.get('trust_grade', 'Unknown')}",
        "",
        "## Key Risks",
    ]

    if penalties:
        for penalty in penalties:
            lines.append(f"- {penalty}")
    else:
        lines.append("- No major penalties detected.")

    lines.extend([
        "",
        "## Missing Values",
    ])

    if missing_values:
        for col, ratio in missing_values.items():
            lines.append(f"- {col}: {ratio * 100:.2f}%")
    else:
        lines.append("- No missing values detected.")

    lines.extend([
        "",
        "## Class Imbalance",
    ])

    if imbalance.get("status") == "no_target_column_provided":
        lines.append("- No target column provided.")
    else:
        lines.extend([
            f"- Target column: {imbalance.get('target_column', 'Unknown')}",
            f"- Minority class ratio: {imbalance.get('minority_class_ratio', 'Unknown')}",
            f"- Is imbalanced: {imbalance.get('is_imbalanced', False)}",
        ])

    lines.extend([
        "",
        "## Possible Target Leakage",
    ])

    if leakage_count == 0:
        lines.append("- No obvious leakage risk detected.")
    else:
        for item in leakage_risks:
            lines.append(
                f"- {item.get('column', 'Unknown')}: "
                f"{item.get('risk_type', 'risk')} — "
                f"{item.get('reason', 'No reason provided.')}"
            )

    lines.extend([
        "",
        "## Verification Proof",
        f"- Dataset fingerprint: `{proof.get('dataset_fingerprint', 'pending')}`",
        f"- Canonical report hash: `{proof.get('report_hash', 'pending')}`",
        f"- Execution timestamp: `{proof.get('execution_timestamp', 'pending')}`",
        "",
        "## Recommendation",
        recommendation,
        "",
    ])

    return "\n".join(lines)
