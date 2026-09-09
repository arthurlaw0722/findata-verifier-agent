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
    layout="wide",
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
        font-size: 1.05rem;
        color: #6b7280;
        margin-bottom: 1.1rem;
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

    .section-note {
        color: #6b7280;
        font-size: 0.92rem;
        margin-bottom: 0.5rem;
    }

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


def render_header():
    st.markdown(
        '<div class="main-title">FinData Verifier Agent</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        (
            '<div class="subtitle">'
            "Verify financial datasets before downstream AI agents "
            "and ML models use them."
            "</div>"
        ),
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <span class="badge">CROO CAP-ready</span>
        <span class="badge">Financial Data Verification</span>
        <span class="badge">Target Leakage Detection</span>
        <span class="badge">SHA256 Proof</span>
        <span class="badge">A2A Workflow</span>
        """,
        unsafe_allow_html=True,
    )


def uploaded_file_signature(uploaded_file):
    return (
        uploaded_file.name,
        getattr(uploaded_file, "size", None),
        getattr(uploaded_file, "file_id", None),
    )


def load_dataset(uploaded_file):
    signature = uploaded_file_signature(uploaded_file)

    if st.session_state.get("loaded_file_signature") != signature:
        uploaded_file.seek(0)
        df = pd.read_csv(uploaded_file)
        uploaded_file.seek(0)

        st.session_state["dataset_df"] = df
        st.session_state["dataset_profile"] = profile_dataset(df)
        st.session_state["loaded_file_signature"] = signature

        st.session_state.pop("verification_result", None)
        st.session_state.pop("verification_signature", None)

    return (
        st.session_state["dataset_df"],
        st.session_state["dataset_profile"],
    )


def render_configuration():
    st.subheader("Dataset configuration")

    config_col1, config_col2 = st.columns([1.35, 1])

    with config_col1:
        uploaded_file = st.file_uploader(
            "Upload CSV",
            type=["csv"],
            key="dataset_uploader",
        )

    df = None
    dataset_profile = None

    if uploaded_file is not None:
        with st.spinner("Loading dataset..."):
            df, dataset_profile = load_dataset(uploaded_file)

    with config_col2:
        if df is None:
            target_column = ""
            st.selectbox(
                "Target column",
                ["Upload a CSV first"],
                disabled=True,
                key="target_column_disabled",
            )
        else:
            target_column = st.selectbox(
                "Target column",
                options=[""] + df.columns.tolist(),
                index=0,
                format_func=lambda value: (
                    "None / No target"
                    if value == ""
                    else value
                ),
                help="Select the column you want to predict or analyse.",
                key="target_column_selector",
            )

        dataset_name = st.text_input(
            "Dataset name",
            value="Credit Card Fraud Detection",
        )

    return uploaded_file, df, dataset_profile, target_column, dataset_name


def render_overview(df, dataset_profile, target_column):
    st.subheader("Dataset Overview")
    st.caption(
        "A compact view of dataset size, structure, quality, and a sample preview."
    )

    overview_col1, overview_col2, overview_col3, overview_col4 = st.columns(4)
    overview_col1.metric("Rows", f"{dataset_profile['rows']:,}")
    overview_col2.metric("Columns", f"{dataset_profile['columns']:,}")
    overview_col3.metric("Numeric", dataset_profile["numeric_columns"])
    overview_col4.metric("Categorical", dataset_profile["categorical_columns"])

    overview_col5, overview_col6, overview_col7, overview_col8 = st.columns(4)
    overview_col5.metric(
        "Missing Cells",
        f"{dataset_profile['missing_cells']:,}",
    )
    overview_col6.metric(
        "Missing %",
        f"{dataset_profile['missing_ratio'] * 100:.2f}%",
    )
    overview_col7.metric(
        "Duplicate Rows",
        f"{dataset_profile['duplicate_rows']:,}",
    )
    overview_col8.metric(
        "Memory",
        f"{dataset_profile['memory_mb']:.2f} MB",
    )

    st.divider()
    st.markdown("### Dataset Preview")
    st.dataframe(df.head(), use_container_width=True)

    preview_col1, preview_col2, preview_col3 = st.columns(3)
    preview_col1.metric("Rows", f"{df.shape[0]:,}")
    preview_col2.metric("Columns", f"{df.shape[1]:,}")
    preview_col3.metric(
        "Target Column",
        target_column if target_column else "Not provided",
    )


def render_target_analysis(df, target_column):
    st.subheader("Target Analysis")
    st.caption(
        "Understand the prediction target, class balance, and strongest "
        "numeric associations."
    )

    if not target_column:
        st.info(
            "Select a target column above to unlock target-aware analysis."
        )
        return

    if target_column not in df.columns:
        st.warning("The selected target column is not available in this dataset.")
        return

    target_summary = analyze_target(df, target_column)

    target_col1, target_col2, target_col3, target_col4 = st.columns(
        [1.0, 1.8, 0.8, 0.9]
    )
    target_col1.metric("Target", target_summary["target_column"])
    target_col2.metric("Task", target_summary["task_type"])
    target_col3.metric("Unique Values", target_summary["unique_values"])
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

        detail_col1, detail_col2 = st.columns([1, 1.25])

        with detail_col1:
            st.markdown("### Class Distribution")
            class_distribution_df = pd.DataFrame(
                target_summary["class_distribution"]
            )
            st.dataframe(
                class_distribution_df,
                use_container_width=True,
                hide_index=True,
            )

        with detail_col2:
            associations = target_summary["top_numeric_associations"]
            st.markdown("### Top Numeric Associations")
            if associations:
                association_df = pd.DataFrame(associations)
                st.dataframe(
                    association_df,
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.info("No numeric target associations are available.")
    else:
        associations = target_summary["top_numeric_associations"]
        st.markdown("### Top Numeric Associations")
        if associations:
            association_df = pd.DataFrame(associations)
            st.dataframe(
                association_df,
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.info("No numeric target associations are available.")


def render_explorer(df, dataset_profile):
    st.subheader("Data Explorer")
    st.caption(
        "Explore descriptive statistics and interactive visual analysis "
        "without leaving this workspace."
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

        if chart_type in {"Histogram", "Box Plot", "Scatter Plot"} and not numeric_columns:
            st.info("No numeric columns are available for this chart.")
            return

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
            st.plotly_chart(figure, use_container_width=True)

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
            st.plotly_chart(figure, use_container_width=True)

        elif chart_type == "Scatter Plot":
            selector_col1, selector_col2 = st.columns(2)

            with selector_col1:
                x_column = st.selectbox(
                    "X-axis",
                    numeric_columns,
                    key="scatter_x",
                )

            with selector_col2:
                default_y_index = 1 if len(numeric_columns) > 1 else 0
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
                "Large datasets are sampled to a maximum of 30,000 points "
                "for responsive visualisation. Statistical calculations "
                "still use the full dataset."
            )
            st.plotly_chart(figure, use_container_width=True)

        elif chart_type == "Correlation Heatmap":
            figure = correlation_heatmap(df)

            if figure is None:
                st.info(
                    "No numeric columns are available for correlation analysis."
                )
            else:
                st.plotly_chart(figure, use_container_width=True)

        elif chart_type == "Missing Values":
            figure = missing_values_chart(df)

            if figure is None:
                st.success("No missing values were found in this dataset.")
            else:
                st.plotly_chart(figure, use_container_width=True)


def build_verification_result(uploaded_file, df, target_column, dataset_name):
    target = target_column if target_column.strip() else None

    uploaded_file.seek(0)
    raw_bytes = uploaded_file.read()
    uploaded_file.seek(0)

    csv_path = None

    try:
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".csv",
        ) as tmp:
            tmp.write(raw_bytes)
            csv_path = tmp.name

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
                "execution_timestamp": "pending",
            },
        )

        proof = create_proof(csv_path, temp_report)

        final_report = generate_markdown_report(
            dataset_name,
            analysis,
            leakage,
            score,
            proof,
        )

        return {
            "analysis": analysis,
            "leakage": leakage,
            "score": score,
            "readiness": readiness,
            "proof": proof,
            "final_report": final_report,
            "dataset_name": dataset_name,
        }
    finally:
        if csv_path and os.path.exists(csv_path):
            os.remove(csv_path)


def render_verification_result(result):
    analysis = result["analysis"]
    leakage = result["leakage"]
    score = result["score"]
    readiness = result["readiness"]
    proof = result["proof"]
    final_report = result["final_report"]
    dataset_name = result["dataset_name"]

    st.success("Verification completed")

    metric_col1, metric_col2, metric_col3, metric_col4 = st.columns(4)
    metric_col1.metric("Trust Score", f"{score['trust_score']}/100")
    metric_col2.metric("Trust Grade", score["trust_grade"])
    metric_col3.metric("ML Readiness", readiness["ml_readiness"])
    metric_col4.metric(
        "Business Readiness",
        readiness["business_decision_readiness"],
    )

    summary_tab, risks_tab, proof_tab = st.tabs(
        [
            "Summary",
            "Risks & Readiness",
            "SHA256 Proof & Report",
        ]
    )

    with summary_tab:
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

    with risks_tab:
        st.markdown("### Dataset Risk Breakdown")

        risk_col1, risk_col2, risk_col3 = st.columns(3)
        profile = analysis["profile"]
        imbalance = analysis.get("class_imbalance", {})

        risk_col1.metric("Duplicate Rows", profile["duplicate_rows"])
        risk_col2.metric("Duplicate Ratio", profile["duplicate_ratio"])
        risk_col3.metric(
            "Leakage Risks",
            leakage.get("risk_count", 0),
        )

        risk_detail_col1, risk_detail_col2 = st.columns([1, 1])

        with risk_detail_col1:
            st.markdown("### Score Penalties")
            if score["penalties"]:
                for penalty in score["penalties"]:
                    st.markdown(f"- {penalty}")
            else:
                st.success("No major penalties detected.")

        with risk_detail_col2:
            st.markdown("### Class Imbalance")

            if imbalance.get("status") == "no_target_column_provided":
                st.warning("No target column provided.")
            else:
                st.metric(
                    "Target Column",
                    imbalance["target_column"],
                )
                imbalance_col1, imbalance_col2 = st.columns(2)
                imbalance_col1.metric(
                    "Minority Class Ratio",
                    imbalance["minority_class_ratio"],
                )
                imbalance_col2.metric(
                    "Is Imbalanced",
                    str(imbalance["is_imbalanced"]),
                )

    with proof_tab:
        st.markdown("### SHA256 Verification Proof")
        st.caption(
            "These SHA256 values let you verify the integrity of the "
            "uploaded dataset and the generated report."
        )

        st.markdown("**SHA256 Dataset Fingerprint**")
        st.code(proof["dataset_fingerprint"])

        st.markdown("**SHA256 Canonical Report Hash**")
        st.code(proof["report_hash"])

        st.markdown("**Execution Timestamp**")
        st.code(proof["execution_timestamp"])

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
                    "proof": proof,
                }
            )

        st.download_button(
            "Download report.md",
            final_report,
            file_name="report.md",
        )


def render_verification(
    uploaded_file,
    df,
    target_column,
    dataset_name,
):
    st.subheader("Verification")
    st.caption(
        "Run the trust, leakage, readiness, and SHA256 integrity checks "
        "from one place."
    )

    if uploaded_file is None or df is None:
        st.info("Upload a CSV dataset first.")
        return

    current_signature = (
        uploaded_file_signature(uploaded_file),
        target_column,
        dataset_name,
    )

    if (
        st.session_state.get("verification_signature") is not None
        and st.session_state.get("verification_signature") != current_signature
    ):
        st.session_state.pop("verification_result", None)
        st.session_state.pop("verification_signature", None)

    run_button = st.button(
        "Run Verification",
        type="primary",
        use_container_width=True,
        key="run_verification",
    )

    if run_button:
        with st.status(
            "Running dataset verification...",
            expanded=True,
        ) as status:
            st.write("Checking dataset quality and class imbalance...")
            st.write("Checking possible target leakage...")
            st.write("Calculating trust and readiness scores...")
            st.write("Generating SHA256 integrity proof and report...")

            result = build_verification_result(
                uploaded_file,
                df,
                target_column,
                dataset_name,
            )

            st.session_state["verification_result"] = result
            st.session_state["verification_signature"] = current_signature

            status.update(
                label="Verification completed",
                state="complete",
                expanded=False,
            )

    result = st.session_state.get("verification_result")

    if (
        result is not None
        and st.session_state.get("verification_signature") == current_signature
    ):
        render_verification_result(result)
    else:
        st.markdown(
            """
            **What this check includes**
            - Dataset quality and duplicate-row checks
            - Outlier-heavy column detection
            - Class imbalance analysis
            - Possible target leakage detection
            - Dataset trust score and readiness guidance
            - SHA256 dataset fingerprint and canonical report hash
            """
        )


def render_getting_started():
    st.subheader("Getting started")
    st.markdown(
        """
        1. Upload a CSV dataset.
        2. Select a target column if you are doing supervised analysis.
        3. Review **Overview**, **Explore**, and **Target Analysis**.
        4. Open **Verification** and run the trust and leakage checks.
        5. Review the **SHA256 Proof & Report** tab and download the report.
        """
    )


render_header()
st.divider()

uploaded_file, df, dataset_profile, target_column, dataset_name = (
    render_configuration()
)

st.divider()

if df is None:
    render_getting_started()
else:
    workspace = st.radio(
        "Workspace",
        [
            "Overview",
            "Explore",
            "Target Analysis",
            "Verification",
        ],
        horizontal=True,
        label_visibility="collapsed",
        key="workspace_nav",
    )

    st.divider()

    if workspace == "Overview":
        render_overview(
            df,
            dataset_profile,
            target_column,
        )
    elif workspace == "Explore":
        render_explorer(
            df,
            dataset_profile,
        )
    elif workspace == "Target Analysis":
        render_target_analysis(
            df,
            target_column,
        )
    elif workspace == "Verification":
        render_verification(
            uploaded_file,
            df,
            target_column,
            dataset_name,
        )
