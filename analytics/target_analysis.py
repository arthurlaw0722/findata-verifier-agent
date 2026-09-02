import pandas as pd


def infer_target_task(series: pd.Series) -> str:
    """Infer whether a target represents classification or regression."""

    clean = series.dropna()

    if clean.empty:
        return "Unknown"

    unique_count = clean.nunique()

    if (
        pd.api.types.is_bool_dtype(clean)
        or isinstance(clean.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(clean)
        or pd.api.types.is_string_dtype(clean)
    ):
        if unique_count == 2:
            return "Binary Classification"
        return "Multiclass Classification"

    if pd.api.types.is_numeric_dtype(clean):
        if unique_count == 2:
            return "Binary Classification"

        if unique_count <= 20:
            return "Multiclass Classification"

        return "Regression"

    return "Unknown"


def target_numeric_associations(
    df: pd.DataFrame,
    target_column: str,
    top_n: int = 10,
) -> list[dict]:
    """
    Rank numeric features by absolute Pearson correlation
    with a numeric target.
    """

    numeric_df = df.select_dtypes(include="number")

    if target_column not in numeric_df.columns:
        return []

    correlations = (
        numeric_df.corr()[target_column]
        .drop(labels=[target_column], errors="ignore")
        .dropna()
    )

    if correlations.empty:
        return []

    ranked = (
        correlations
        .to_frame("correlation")
        .assign(
            absolute_correlation=lambda x: x["correlation"].abs()
        )
        .sort_values(
            "absolute_correlation",
            ascending=False,
        )
        .head(top_n)
    )

    results = []

    for feature, row in ranked.iterrows():
        results.append(
            {
                "feature": feature,
                "correlation": round(
                    float(row["correlation"]),
                    4,
                ),
                "absolute_correlation": round(
                    float(row["absolute_correlation"]),
                    4,
                ),
            }
        )

    return results


def analyze_target(
    df: pd.DataFrame,
    target_column: str,
) -> dict:
    """Create a target-aware analytical summary."""

    if target_column not in df.columns:
        raise ValueError(
            f"Target column '{target_column}' does not exist."
        )

    series = df[target_column]

    row_count = len(df)
    missing_count = int(series.isna().sum())
    unique_count = int(series.nunique(dropna=True))

    missing_pct = (
        missing_count / row_count * 100
        if row_count > 0
        else 0.0
    )

    task_type = infer_target_task(series)

    result = {
        "target_column": target_column,
        "task_type": task_type,
        "rows": row_count,
        "missing": missing_count,
        "missing_pct": round(missing_pct, 4),
        "unique_values": unique_count,
        "class_distribution": [],
        "majority_class": None,
        "majority_count": None,
        "minority_class": None,
        "minority_count": None,
        "minority_pct": None,
        "imbalance_ratio": None,
        "top_numeric_associations": target_numeric_associations(
            df,
            target_column,
        ),
    }

    if "Classification" in task_type:
        counts = (
            series
            .dropna()
            .value_counts()
        )

        result["class_distribution"] = [
            {
                "class": str(label),
                "count": int(count),
                "percentage": round(
                    float(count / counts.sum() * 100),
                    4,
                ),
            }
            for label, count in counts.items()
        ]

        if not counts.empty:
            majority_class = counts.index[0]
            majority_count = int(counts.iloc[0])

            minority_class = counts.index[-1]
            minority_count = int(counts.iloc[-1])

            result["majority_class"] = str(
                majority_class
            )
            result["majority_count"] = majority_count

            result["minority_class"] = str(
                minority_class
            )
            result["minority_count"] = minority_count

            result["minority_pct"] = round(
                minority_count / counts.sum() * 100,
                4,
            )

            if minority_count > 0:
                result["imbalance_ratio"] = round(
                    majority_count / minority_count,
                    2,
                )

    return result
