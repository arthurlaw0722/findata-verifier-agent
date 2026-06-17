from datetime import datetime


def generate_markdown_report(
    dataset_name: str,
    analysis: dict,
    leakage: dict,
    score: dict,
    proof: dict
) -> str:
    profile = analysis["profile"]
    imbalance = analysis["class_imbalance"]

    report = f"""
# FinData Verifier Agent Report

## Dataset
**Name:** {dataset_name}  
**Generated at:** {datetime.utcnow().isoformat()} UTC

## Dataset Overview
- Rows: {profile["rows"]}
- Columns: {profile["columns"]}
- Duplicate rows: {profile["duplicate_rows"]}
- Duplicate ratio: {profile["duplicate_ratio"]}

## Trust Score
**Score:** {score["trust_score"]}/100  
**Grade:** {score["trust_grade"]}

## Key Risks
"""

    if score["penalties"]:
        for p in score["penalties"]:
            report += f"- {p}\n"
    else:
        report += "- No major risk detected.\n"

    report += "\n## Missing Values\n"
    if profile["missing_values_ratio"]:
        for col, ratio in profile["missing_values_ratio"].items():
            report += f"- {col}: {ratio * 100:.2f}%\n"
    else:
        report += "- No missing values detected.\n"

    report += "\n## Class Imbalance\n"
    if imbalance.get("status") == "no_target_column_provided":
        report += "- No target column provided.\n"
    else:
        report += f"- Target column: {imbalance['target_column']}\n"
        report += f"- Minority class ratio: {imbalance['minority_class_ratio']}\n"
        report += f"- Is imbalanced: {imbalance['is_imbalanced']}\n"

    report += "\n## Possible Target Leakage\n"
    if leakage.get("risk_count", 0) == 0:
        report += "- No obvious leakage risk detected.\n"
    else:
        for item in leakage["leakage_risks"]:
            report += f"- {item['column']}: {item['risk_type']} — {item['reason']}\n"

    report += f"""
## Verification Proof
- Dataset fingerprint: `{proof["dataset_fingerprint"]}`
- Report hash: `{proof["report_hash"]}`
- Execution timestamp: `{proof["execution_timestamp"]}`

## Recommendation
This dataset can be used for experimentation if the listed risks are addressed. 
For production or financial decision-making, further validation is recommended.
"""

    return report
