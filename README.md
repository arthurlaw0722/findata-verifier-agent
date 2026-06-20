# FinData Verifier Agent

FinData Verifier Agent is a CROO CAP-powered financial dataset verification agent.

It checks CSV and Kaggle datasets for data quality risks before they are used by downstream AI agents, machine learning models, research agents, or financial decision-making workflows.

---

## Problem

AI agents increasingly use external datasets to train models, generate research, and support business decisions.

However, many datasets contain hidden risks such as missing values, duplicate rows, outliers, severe class imbalance, or possible target leakage.

In financial workflows, bad data can lead to bad models, unreliable reports, and poor business decisions.

---

## Solution

FinData Verifier Agent checks a dataset and produces an explainable verification report.

The agent analyses a CSV dataset and returns:

* Dataset profile
* Missing value analysis
* Duplicate row detection
* Outlier-heavy column detection
* Class imbalance detection
* Possible target leakage detection
* Dataset trust score
* Markdown verification report
* JSON summary
* SHA256 dataset fingerprint
* SHA256 report hash
* Execution timestamp

This allows another AI agent to decide whether to accept or reject a dataset before using it.

---

## Why CROO CAP

CROO CAP enables AI agents to be discovered, hired, paid, and verified through a decentralized agent marketplace.

FinData Verifier Agent fits this workflow because it acts as a specialist verification service provider.

```text
Buyer Agent
    |
    | hires verification service through CROO CAP
    v
FinData Verifier Agent
    |
    | analyses dataset
    v
Trust Report + Proof Hash
    |
    | verified output
    v
Downstream ML / Research Agent
```

Each verification job produces a report and proof hash, making the result easier to audit and verify.

---

## Key Features

* CSV dataset verification
* Missing value detection
* Duplicate row detection
* Outlier detection
* Class imbalance detection
* Possible target leakage detection
* Explainable dataset trust score
* Business-friendly readiness recommendation
* Markdown report generation
* JSON summary output
* SHA256 dataset fingerprint
* SHA256 report hash
* Streamlit demo UI
* CROO Python SDK provider integration
* Agent-to-agent composability demo

---

## Demo Dataset

The demo uses the Kaggle Credit Card Fraud Detection dataset.

Dataset reference:

```text
mlg-ulb/creditcardfraud
```

This dataset is suitable because it is finance-related and contains a strong class imbalance, which is common in real fraud detection problems.

---

## Example Output

```text
Trust Score: 75/100
Trust Grade: Medium Trust

Key Risks:
- Outlier-heavy columns
- Class imbalance

Class Imbalance:
- Target column: Class
- Minority class ratio: 0.0017
- Is imbalanced: True

Possible Target Leakage:
- No obvious leakage risk detected

Verification Proof:
- Dataset fingerprint: SHA256 hash
- Report hash: SHA256 hash
- Execution timestamp: UTC timestamp
```

---

## A2A Composability Demo

This project includes a simple buyer-agent simulation.

```bash
python3 examples/buyer_agent_demo.py
```

Demo flow:

```text
Buyer Agent
    |
    | requests dataset verification
    v
FinData Verifier Agent
    |
    | returns trust score and report
    v
Model Builder Agent
    |
    | accepts or rejects dataset
    v
Train model only if dataset is trusted enough
```

Decision rule:

* If trust score is greater than or equal to 70, the downstream agent can proceed with model experimentation.
* If trust score is below 70, the downstream agent rejects the dataset until risks are fixed.

This demonstrates how one AI agent can hire another specialist verification agent before executing a higher-risk financial or machine learning workflow.

---

## Target Leakage Safety Gate Demo

This demo shows how FinData Verifier Agent prevents a downstream model from training on a dataset with target leakage.

```bash
python3 examples/leakage_gate_demo.py
```

The sample loan dataset intentionally includes `approved_status`, a column that reveals the final approval outcome.

The agent detects the leakage and returns:

```text
Leakage Risks: 2
Decision: BLOCK_DOWNSTREAM_TRAINING
Suspicious column: approved_status
```

This demonstrates that FinData Verifier Agent is not only a data checker. It acts as a safety gate before higher-risk financial AI workflows begin.

## Production Readiness Layer

The agent translates technical data risks into business-friendly readiness guidance.

Example:

```text
ML Readiness: Medium
Business Decision Readiness: Low
Main reason: Severe class imbalance and possible data risks
Recommended next step: resampling, leakage review, and validation before production use
```

---

## CROO CAP Integration

The project includes a CROO Python SDK provider.

Run the provider:

```bash
python3 -m agent.cap_provider
```

The provider connects to CROO, listens for paid verification jobs, runs the FinData verification pipeline, and delivers a JSON-compatible result containing:

* Trust score
* Trust grade
* Markdown report
* Dataset fingerprint
* Report hash
* Execution timestamp

Required environment variables:

```bash
CROO_API_URL=https://api.croo.network
CROO_WS_URL=wss://api.croo.network/ws
CROO_SDK_KEY=your_croo_sdk_key
```

For safety, these should be stored in a local `.env` file. The `.env` file should never be committed to GitHub.

---

## Streamlit Demo

Run the local UI:

```bash
streamlit run app/streamlit_app.py
```

Then upload a CSV file, enter the target column, and run verification.

For the credit card fraud demo:

```text
Target column: Class
Dataset name: Credit Card Fraud Detection
```

The UI displays:

* Dataset preview
* Trust score
* Trust grade
* Full markdown report
* Download button for report.md

---

## Command Line Demo

Run the verifier from terminal:

```bash
python3 run_agent.py --csv data/creditcard.csv --dataset-name "Credit Card Fraud Detection" --target Class
```

Generated outputs:

```text
outputs/report.md
outputs/summary.json
outputs/proof.json
```

---

## Installation

Create a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Project Structure

```text
findata-verifier-agent/
|
├── agent/
│   ├── analyzer.py
│   ├── scoring.py
│   ├── leakage.py
│   ├── report_generator.py
│   ├── proof.py
│   ├── readiness.py
│   └── cap_provider.py
|
├── app/
│   └── streamlit_app.py
|
├── examples/
│   ├── sample_request.json
│   └── buyer_agent_demo.py
|
├── outputs/
│   ├── report.md
│   ├── summary.json
│   └── proof.json
|
├── run_agent.py
├── requirements.txt
├── README.md
├── LICENSE
└── demo_script.md
```

---

## Tech Stack

* Python
* pandas
* numpy
* scikit-learn
* Streamlit
* Kaggle API
* CROO Python SDK
* SHA256 proof hashing

---

## Track

Primary track:

```text
Data & Verification Agents
```

Secondary track:

```text
Research & Intelligence Agents
```

This project focuses on financial data verification, business analytics, and safe downstream AI workflows.

---

## License

MIT License
