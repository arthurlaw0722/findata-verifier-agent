import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from agent.proof import verify_proof


CSV_PATH = PROJECT_ROOT / "examples" / "loan_leakage_demo.csv"
PROOF_PATH = PROJECT_ROOT / "outputs" / "proof.json"
REPORT_PATH = PROJECT_ROOT / "outputs" / "report.md"


def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"Missing demo dataset: {CSV_PATH}")

    if not PROOF_PATH.exists() or not REPORT_PATH.exists():
        raise FileNotFoundError(
            "Missing outputs. Run run_agent.py first to generate report.md and proof.json."
        )

    with open(PROOF_PATH) as file:
        proof = json.load(file)

    with open(REPORT_PATH) as file:
        report_text = file.read()

    result = verify_proof(
        str(CSV_PATH),
        report_text,
        proof,
    )

    print("FinData Verifier Agent — Proof Verification")
    print("-" * 48)
    print("Dataset fingerprint valid:", result["dataset_fingerprint_valid"])
    print("Canonical report hash valid:", result["canonical_report_hash_valid"])
    print("Overall proof valid:", result["valid"])

    if not result["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
