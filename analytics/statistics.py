import pandas as pd


def numeric_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for numeric columns."""

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return pd.DataFrame()

    summary = numeric_df.describe().T

    summary["median"] = numeric_df.median()
    summary["skewness"] = numeric_df.skew()
    summary["missing"] = numeric_df.isna().sum()
    summary["missing_pct"] = (
        numeric_df.isna().mean() * 100
    )
    summary["unique"] = numeric_df.nunique()

    summary = summary.rename(
        columns={
            "count": "Count",
            "mean": "Mean",
            "std": "Std",
            "min": "Min",
            "25%": "25%",
            "50%": "50%",
            "75%": "75%",
            "max": "Max",
            "median": "Median",
            "skewness": "Skewness",
            "missing": "Missing",
            "missing_pct": "Missing %",
            "unique": "Unique",
        }
    )

    desired_order = [
        "Count",
        "Mean",
        "Median",
        "Std",
        "Min",
        "25%",
        "50%",
        "75%",
        "Max",
        "Skewness",
        "Missing",
        "Missing %",
        "Unique",
    ]

    return summary[desired_order]


def categorical_statistics(df: pd.DataFrame) -> pd.DataFrame:
    """Return descriptive statistics for non-numeric columns."""

    categorical_df = df.select_dtypes(exclude="number")

    if categorical_df.empty:
        return pd.DataFrame()

    rows = []

    for column in categorical_df.columns:
        series = categorical_df[column]

        non_null = series.dropna()
        mode = non_null.mode()

        rows.append(
            {
                "Column": column,
                "Count": int(series.count()),
                "Unique": int(series.nunique()),
                "Missing": int(series.isna().sum()),
                "Missing %": round(series.isna().mean() * 100, 2),
                "Most Frequent": (
                    str(mode.iloc[0])
                    if len(mode) == 1
                    else "Tie: " + ", ".join(map(str, mode.tolist()))
                    if len(mode) > 1
                    else "N/A"
                ),
            }
        )

    return pd.DataFrame(rows).set_index("Column")
