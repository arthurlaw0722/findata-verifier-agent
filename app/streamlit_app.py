import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import tempfile

from agent.analyzer import analyse_dataset
from agent.leakage import detect_possible_leakage
from agent.scoring import calculate_trust_score
from agent.report_generator import generate_markdown_report
from agent.proof import create_proof


st.title("FinData Verifier Agent")
st.write("A CAP-powered data verification agent for Kaggle, business and financial datasets.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
target_column = st.text_input("Target column, optional")
dataset_name = st.text_input("Dataset name", value="Uploaded Dataset")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        csv_path = tmp.name

    df = pd.read_csv(csv_path)
    st.write("Preview")
    st.dataframe(df.head())

    if st.button("Run Verification"):
        analysis = analyse_dataset(csv_path, target_column if target_column else None)
        leakage = detect_possible_leakage(df, target_column if target_column else None)
        score = calculate_trust_score(analysis, leakage)

        temp_report = generate_markdown_report(
            dataset_name,
            analysis,
            leakage,
            score,
            {
                "dataset_fingerprint": "pending",
                "report_hash": "pending",
                "execution_timestamp": "pending"
            }
        )

        proof = create_proof(csv_path, temp_report)

        final_report = generate_markdown_report(
            dataset_name,
            analysis,
            leakage,
            score,
            proof
        )

        st.subheader("Trust Score")
        st.metric("Dataset Trust Score", score["trust_score"])
        st.write(score["trust_grade"])

        st.subheader("Report")
        st.markdown(final_report)

        st.download_button(
            "Download report.md",
            final_report,
            file_name="report.md"
        )
