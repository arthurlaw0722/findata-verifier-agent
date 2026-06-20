import sys
import os
import asyncio
import json
from pathlib import Path

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

from agent.cap_provider import run_verification_job


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "examples" / "safe_loan_demo.csv"


def create_safe_loan_dataset(path: Path) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    n = 240

    income = rng.integers(25000, 120001, n)
    age = rng.integers(21, 66, n)
    credit_score = rng.integers(520, 851, n)
    debt_to_income = rng.integers(5, 51, n)
    loan_amount = rng.integers(3000, 40001, n)
    years_employed = rng.integers(0, 26, n)

    # The target depends on several factors plus noise.
    # No single feature directly reveals the target.
    risk_signal = (
        0.015 * (credit_score - 650)
        + 0.000015 * (income - 55000)
        - 0.08 * (debt_to_income - 25)
        + 0.05 * (years_employed - 8)
        - 0.000015 * (loan_amount - 16000)
        + rng.normal(0, 1.0, n)
    )

    approval = (risk_signal >= np.median(risk_signal)).astype(int)

    df = pd.DataFrame(
        {
            "income": income,
            "age": age,
            "credit_score": credit_score,
            "debt_to_income": debt_to_income,
            "loan_amount": loan_amount,
            "years_employed": years_employed,
            "Class": approval,
        }
    )

    df.to_csv(path, index=False)
    return df


async def main():
    print("Buyer Agent: requesting verification for a safe loan dataset...")

    df = create_safe_loan_dataset(CSV_PATH)

    request = {
        "csv_path": str(CSV_PATH),
        "dataset_name": "Safe Loan Approval Demo",
        "target_column": "Class",
        "task_type": "classification",
    }

    verification_json = await run_verification_job(json.dumps(request))
    verification_result = json.loads(verification_json)

    print("\nFinData Verifier Agent returned:")
    print(f"Trust Score: {verification_result['trust_score']}")
    print(f"Trust Grade: {verification_result['trust_grade']}")
    print(f"Leakage Risks: {verification_result['leakage']['risk_count']}")

    readiness = verification_result["readiness"]

    if not readiness.get("should_train_model", False):
        print("\nModel Builder Agent safety decision:")
        print("REJECT_DATASET")
        print("The dataset did not pass the verification gate.")
        return

    print("\nModel Builder Agent safety decision:")
    print("ALLOW_BASELINE_MODEL_EXPERIMENT")

    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = LogisticRegression(max_iter=2000)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions)), 4),
        "recall": round(float(recall_score(y_test, predictions)), 4),
        "f1_score": round(float(f1_score(y_test, predictions)), 4),
    }

    print("\nBaseline Logistic Regression completed:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
