import argparse
import json
import os

from agent.analyzer import analyse_dataset
from agent.leakage import detect_possible_leakage
from agent.scoring import calculate_trust_score
from agent.report_generator import generate_markdown_report
from agent.proof import create_proof, save_json

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--dataset-name", default="Unknown Dataset")
    parser.add_argument("--target", default=None)
    args = parser.parse_args()

    df = pd.read_csv(args.csv)

    analysis = analyse_dataset(args.csv, args.target)
    leakage = detect_possible_leakage(df, args.target)
    score = calculate_trust_score(analysis, leakage)

    temp_report = generate_markdown_report(
        args.dataset_name,
        analysis,
        leakage,
        score,
        {
            "dataset_fingerprint": "pending",
            "report_hash": "pending",
            "execution_timestamp": "pending"
        }
    )

    proof = create_proof(args.csv, temp_report)

    final_report = generate_markdown_report(
        args.dataset_name,
        analysis,
        leakage,
        score,
        proof
    )

    os.makedirs("outputs", exist_ok=True)

    with open("outputs/report.md", "w") as f:
        f.write(final_report)

    save_json({
        "dataset_name": args.dataset_name,
        "analysis": analysis,
        "leakage": leakage,
        "score": score,
        "proof": proof
    }, "outputs/summary.json")

    save_json(proof, "outputs/proof.json")

    print("Report generated: outputs/report.md")
    print("Summary generated: outputs/summary.json")
    print("Proof generated: outputs/proof.json")


if __name__ == "__main__":
    main()
