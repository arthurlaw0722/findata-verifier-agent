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
from analytics.profiler import profile_dataset
from analytics.target_analysis import analyze_target
from analytics.statistics import (
    numeric_statistics,
    categorical_statistics,
)
from analytics.visualizations import (
    histogram_chart,
    box_chart,
    scatter_chart,
    correlation_heatmap,
    missing_values_chart,
)


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
    st.markdown(
        """
        <style>
        .st-key-run_verification button {
            min-height: 3rem;
            font-weight: 700;
            border-radius: 0.6rem;
            transition: all 0.15s ease;
        }

        .st-key-run_verification button:not(:disabled) {
            background-color: #D62828 !important;
            border-color: #D62828 !important;
            color: white !important;
        }

        .st-key-run_verification button:not(:disabled):hover {
            background-color: #B71C1C !important;
            border-color: #B71C1C !important;
            color: white !important;
        }

        .st-key-run_verification button:disabled {
            background-color: #FDECEC !important;
            border: 1px solid #DFA3A3 !important;
            color: #A63A3A !important;
            opacity: 1 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    run_button = st.button(
        "Run Verification",
        type="primary",
        disabled=uploaded_file is None,
        use_container_width=True,
        key="run_verification",
    )

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

    dataset_profile = profile_dataset(df)

    st.divider()
    st.subheader("Dataset Overview")

    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)

    overview_col1.metric("Rows", f"{dataset_profile['rows']:,}")
    overview_col2.metric("Columns", f"{dataset_profile['columns']:,}")
    overview_col3.metric("Numeric", dataset_profile["numeric_columns"])
    overview_col4.metric("Categorical", dataset_profile["categorical_columns"])

    overview_col5, overview_col6, overview_col7, overview_col8 = st.columns(4)

    overview_col5.metric("Missing Cells", f"{dataset_profile['missing_cells']:,}")
    overview_col6.metric("Missing %", f"{dataset_profile['missing_ratio'] * 100:.2f}%")
    overview_col7.metric("Duplicate Rows", f"{dataset_profile['duplicate_rows']:,}")
    overview_col8.metric("Memory", f"{dataset_profile['memory_mb']:.2f} MB")


    st.divider()
    if target_column and target_column in df.columns:
        target_summary = analyze_target(
            df,
            target_column,
        )

        st.divider()
        st.subheader("Target Analysis")
        st.caption(
            "Understand the prediction target, class balance, "
            "and strongest numeric associations."
        )

        target_col1, target_col2, target_col3, target_col4 = st.columns(4)

        target_col1.metric(
            "Target",
            target_summary["target_column"],
        )

        target_col2.metric(
            "Task",
            target_summary["task_type"],
        )

        target_col3.metric(
            "Unique Values",
            target_summary["unique_values"],
        )

        target_col4.metric(
            "Missing %",
            f'{target_summary["missing_pct"]:.2f}%',
        )

        if "Classification" in target_summary["task_type"]:
            class_col1, class_col2, class_col3 = st.columns(3)

            class_col1.metric(
                "Majority Class",
                target_summary["majority_class"],
                help=f'{target_summary["majority_count"]:,} rows',
            )

            class_col2.metric(
                "Minority Class",
                target_summary["minority_class"],
                help=f'{target_summary["minority_count"]:,} rows',
            )

            imbalance_value = target_summary["imbalance_ratio"]

            class_col3.metric(
                "Imbalance Ratio",
                (
                    f"{imbalance_value:.2f}:1"
                    if imbalance_value is not None
                    else "N/A"
                ),
            )

            st.markdown("#### Class Distribution")

            class_distribution_df = pd.DataFrame(
                target_summary["class_distribution"]
            )

            st.dataframe(
                class_distribution_df,
                use_container_width=True,
                hide_index=True,
            )

        associations = target_summary["top_numeric_associations"]

        if associations:
            st.markdown("#### Top Numeric Associations")

            association_df = pd.DataFrame(associations)

            st.dataframe(
                association_df,
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.subheader("Data Explorer")
    st.caption(
        "Explore descriptive statistics before moving to visual analysis "
        "and machine learning."
    )

    numeric_stats = numeric_statistics(df)
    categorical_stats = categorical_statistics(df)

    numeric_tab, categorical_tab, visual_tab = st.tabs(
        [
            "Numeric Statistics",
            "Categorical Statistics",
            "Visual Analysis",
        ]
    )

    with numeric_tab:
        if numeric_stats.empty:
            st.info("No numeric columns found.")
        else:
            st.dataframe(
                numeric_stats.round(4),
                use_container_width=True,
            )

    with categorical_tab:
        if categorical_stats.empty:
            st.info("No categorical columns found.")
        else:
            st.dataframe(
                categorical_stats,
                use_container_width=True,
            )

    with visual_tab:
        numeric_columns = dataset_profile["numeric_column_names"]

        group_columns = (
            dataset_profile["categorical_column_names"]
            + dataset_profile["low_cardinality_numeric_column_names"]
        )

        chart_type = st.selectbox(
            "Chart type",
            [
                "Histogram",
                "Box Plot",
                "Scatter Plot",
                "Correlation Heatmap",
                "Missing Values",
            ],
            key="visual_chart_type",
        )

        if chart_type == "Histogram":
            column = st.selectbox(
                "Column",
                numeric_columns,
                key="histogram_column",
            )

            group = st.selectbox(
                "Group / colour",
                ["None"] + group_columns,
                key="histogram_group",
            )

            figure = histogram_chart(
                df,
                column,
                None if group == "None" else group,
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

        elif chart_type == "Box Plot":
            column = st.selectbox(
                "Column",
                numeric_columns,
                key="box_column",
            )

            group = st.selectbox(
                "Group by",
                ["None"] + group_columns,
                key="box_group",
            )

            figure = box_chart(
                df,
                column,
                None if group == "None" else group,
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

        elif chart_type == "Scatter Plot":
            selector_col1, selector_col2 = st.columns(2)

            with selector_col1:
                x_column = st.selectbox(
                    "X-axis",
                    numeric_columns,
                    key="scatter_x",
                )

            with selector_col2:
                default_y_index = (
                    1 if len(numeric_columns) > 1 else 0
                )

                y_column = st.selectbox(
                    "Y-axis",
                    numeric_columns,
                    index=default_y_index,
                    key="scatter_y",
                )

            group = st.selectbox(
                "Group / colour",
                ["None"] + group_columns,
                key="scatter_group",
            )

            figure = scatter_chart(
                df,
                x_column,
                y_column,
                None if group == "None" else group,
            )

            st.caption(
                "Large datasets are sampled to a maximum of "
                "30,000 points for responsive visualisation. "
                "Statistical calculations still use the full dataset."
            )

            st.plotly_chart(
                figure,
                use_container_width=True,
            )

        elif chart_type == "Correlation Heatmap":
            figure = correlation_heatmap(df)

            if figure is None:
                st.info(
                    "No numeric columns are available "
                    "for correlation analysis."
                )
            else:
                st.plotly_chart(
                    figure,
                    use_container_width=True,
                )

        elif chart_type == "Missing Values":
            figure = missing_values_chart(df)

            if figure is None:
                st.success(
                    "No missing values were found in this dataset."
                )
            else:
                st.plotly_chart(
                    figure,
                    use_container_width=True,
                )

    st.subheader("Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    preview_col1, preview_col2, preview_col3 = st.columns(3)
    preview_col1.metric("Rows", f"{df.shape[0]:,}")
    preview_col2.metric("Columns", f"{df.shape[1]:,}")
    preview_col3.metric("Target Column", target_column if target_column else "Not provided")

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
        st.code(f"Canonical report hash: {proof['report_hash']}")
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
