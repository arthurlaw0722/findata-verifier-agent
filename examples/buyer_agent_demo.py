import sys
import os
import asyncio
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from agent.cap_provider import run_verification_job


def model_builder_decision(verification_result: dict) -> dict:
    trust_score = verification_result["trust_score"]
    trust_grade = verification_result["trust_grade"]

    if trust_score >= 70:
        decision = "ACCEPT_DATASET_FOR_MODEL_EXPERIMENT"
        action = "Proceed to train a baseline logistic regression model."
    else:
        decision = "REJECT_DATASET"
        action = "Do not train until data risks are fixed."

    return {
        "downstream_agent": "Model Builder Agent",
        "received_trust_score": trust_score,
        "received_trust_grade": trust_grade,
        "decision": decision,
        "next_action": action
    }


async def main():
    print("Buyer Agent: requesting dataset verification from FinData Verifier Agent...")

    request = {
        "csv_path": "data/creditcard.csv",
        "dataset_name": "Credit Card Fraud Detection",
        "target_column": "Class",
        "task_type": "classification"
    }

    verification_json = await run_verification_job(json.dumps(request))
    verification_result = json.loads(verification_json)

    print("\nFinData Verifier Agent returned:")
    print(f"Trust Score: {verification_result['trust_score']}")
    print(f"Trust Grade: {verification_result['trust_grade']}")

    decision = model_builder_decision(verification_result)

    print("\nModel Builder Agent decision:")
    print(json.dumps(decision, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
