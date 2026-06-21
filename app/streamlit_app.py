import sys
import os
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import streamlit as st

from agent.analyzer import analyse_dataset
from agent.leakage import detect_possible_leakage
from agent.scoring import calculate_trust_score
from agent.report_generator import generate_markdown_report
from agent.proof import create_proof
from agent.readiness import assess_readiness


st.set_page_config(
    page_title="FinData Verifier Agent",
    layout="wide"
)


st.markdown(
    """
    <style>
    .main-title {
        font-size: 2.7rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.15rem;
        color: #5f6368;
        margin-bottom: 1.2rem;
    }
    .badge {
        display: inline-block;
        padding: 0.35rem 0.7rem;
        border-radius: 999px;
        background: #e8f5ee;
        color: #0f7a4f;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.4rem;
        margin-bottom: 0.4rem;
    }
    .risk-box {
        padding: 1rem;
        border-radius: 0.8rem;
        background: #f8fafc;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }
    .small-muted {
        color: #6b7280;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)


st.markdown('<div class="main-title">FinData Verifier Agent</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Verify financial datasets before downstream AI agents and ML models use them.</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <span class="badge">CROO CAP-ready</span>
    <span class="badge">Financial Data Verification</span>
    <span class="badge">Target Leakage Detection</span>
    <span class="badge">SHA256 Proof</span>
    <span class="badge">A2A Workflow</span>
    """,
    unsafe_allow_html=True
)

st.divider()

left_col, right_col = st.columns([1.1, 1])

with left_col:
    st.subheader("Upload dataset")
    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    target_column = st.text_input("Target column", value="", placeholder="e.g. Class")
    dataset_name = st.text_input("Dataset name", value="Credit Card Fraud Detection")

with right_col:
    st.subheader("What this agent checks")
    st.markdown(
        """
        - Missing values  
        - Duplicate rows  
        - Outlier-heavy columns  
        - Class imbalance  
        - Possible target leakage  
        - Dataset trust score  
        - SHA256 dataset and report proof  
        """
    )
    st.info("Designed for Kaggle, business analytics, fraud detection, and financial AI workflows.")


if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(uploaded_file.read())
        csv_path = tmp.name

    df = pd.read_csv(csv_path)

    st.divider()
    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    preview_col1, preview_col2, preview_col3 = st.columns(3)
    preview_col1.metric("Rows", f"{df.shape[0]:,}")
    preview_col2.metric("Columns", f"{df.shape[1]:,}")
    preview_col3.metric("Target Column", target_column if target_column else "Not provided")

    run_button = st.button("Run Verification", type="primary")

    if run_button:
        with st.spinner("Running dataset verification..."):
            target = target_column if target_column.strip() else None

            analysis = analyse_dataset(csv_path, target)
            leakage = detect_possible_leakage(df, target)
            score = calculate_trust_score(analysis, leakage)
            readiness = assess_readiness(analysis, leakage, score)

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

        st.success("Verification completed")

        st.divider()
        st.subheader("Executive Summary")

        metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)

        metric_col1.metric("Trust Score", f"{score['trust_score']}/100")
        metric_col2.metric("Trust Grade", score["trust_grade"])
        metric_col3.metric("ML Readiness", readiness["ml_readiness"])
        metric_col4.metric("Business Readiness", readiness["business_decision_readiness"])

        st.markdown("### Key Findings")

        findings_col1, findings_col2 = st.columns([1, 1])

        with findings_col1:
            st.warning("**Main reasons**")
            for reason in readiness["main_reasons"]:
                st.markdown(f"- {reason}")

        with findings_col2:
            st.info("**Recommended next steps**")
            for step in readiness["recommended_next_steps"]:
                st.markdown(f"- {step}")

        st.markdown("### Dataset Risk Breakdown")

        risk_col1, risk_col2, risk_col3 = st.columns(3)

        profile = analysis["profile"]
        imbalance = analysis.get("class_imbalance", {})

        risk_col1.metric("Duplicate Rows", profile["duplicate_rows"])
        risk_col2.metric("Duplicate Ratio", profile["duplicate_ratio"])
        risk_col3.metric("Leakage Risks", leakage.get("risk_count", 0))

        st.markdown("### Score Penalties")

        if score["penalties"]:
            for penalty in score["penalties"]:
                st.markdown(f"- {penalty}")
        else:
            st.markdown("- No major penalties detected.")

        st.markdown("### Class Imbalance")

        if imbalance.get("status") == "no_target_column_provided":
            st.warning("No target column provided.")
        else:
            imb_col1, imb_col2, imb_col3 = st.columns(3)
            imb_col1.metric("Target Column", imbalance["target_column"])
            imb_col2.metric("Minority Class Ratio", imbalance["minority_class_ratio"])
            imb_col3.metric("Is Imbalanced", str(imbalance["is_imbalanced"]))

        st.markdown("### Verification Proof")

        st.code(f"Dataset fingerprint: {proof['dataset_fingerprint']}")
        st.code(f"Report hash: {proof['report_hash']}")
        st.code(f"Execution timestamp: {proof['execution_timestamp']}")

        with st.expander("View full markdown report"):
            st.markdown(final_report)

        with st.expander("View JSON-style summary"):
            st.json(
                {
                    "dataset_name": dataset_name,
                    "trust_score": score["trust_score"],
                    "trust_grade": score["trust_grade"],
                    "readiness": readiness,
                    "leakage": leakage,
                    "proof": proof
                }
            )

        st.download_button(
            "Download report.md",
            final_report,
            file_name="report.md"
        )
else:
    st.divider()
    st.markdown("### Demo flow")
    st.markdown(
        """
        1. Upload a CSV dataset  
        2. Enter the target column, for example `Class`  
        3. Run verification  
        4. Review trust score, risks, readiness, and SHA256 proof  
        5. Download the markdown report  
        """
    )
