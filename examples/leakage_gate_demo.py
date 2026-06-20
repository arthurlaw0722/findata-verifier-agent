import sys
import os
import asyncio
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.cap_provider import run_verification_job


def model_builder_decision(verification_result: dict) -> dict:
    leakage = verification_result.get("leakage", {})
    readiness = verification_result.get("readiness", {})
    risk_count = leakage.get("risk_count", 0)
    risks = leakage.get("leakage_risks", [])

    suspicious_columns = sorted(
        {risk.get("column", "unknown") for risk in risks}
    )

    if risk_count > 0:
        return {
            "downstream_agent": "Model Builder Agent",
            "decision": "BLOCK_DOWNSTREAM_TRAINING",
            "reason": (
                "Target leakage risk detected. A downstream model may learn "
                "the answer instead of genuine predictive patterns."
            ),
            "suspicious_columns": suspicious_columns,
            "next_action": (
                "Remove or validate suspicious columns, then rerun "
                "FinData Verifier Agent before training."
            )
        }

    if readiness.get("should_train_model", False):
        return {
            "downstream_agent": "Model Builder Agent",
            "decision": "ALLOW_BASELINE_MODEL_EXPERIMENT",
            "reason": "No leakage blocker detected and dataset passed the safety gate.",
            "next_action": "Proceed to train a baseline logistic regression model."
        }

    return {
        "downstream_agent": "Model Builder Agent",
        "decision": "REJECT_DATASET",
        "reason": "Dataset does not meet the minimum readiness requirement.",
        "next_action": "Address the recommended data risks before training."
    }


async def main():
    print("Buyer Agent: requesting verification for a loan approval dataset...")

    request = {
        "csv_path": "examples/loan_leakage_demo.csv",
        "dataset_name": "Loan Approval Leakage Demo",
        "target_column": "Class",
        "task_type": "classification"
    }

    verification_json = await run_verification_job(json.dumps(request))
    verification_result = json.loads(verification_json)

    print("\nFinData Verifier Agent returned:")
    print(f"Trust Score: {verification_result['trust_score']}")
    print(f"Trust Grade: {verification_result['trust_grade']}")
    print(f"Leakage Risks: {verification_result['leakage']['risk_count']}")

    decision = model_builder_decision(verification_result)

    print("\nModel Builder Agent safety decision:")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
