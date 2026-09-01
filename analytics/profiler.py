import pandas as pd


def profile_dataset(df: pd.DataFrame) -> dict:
    """Return a high-level profile of a pandas DataFrame."""

    rows, columns = df.shape

    numeric_columns = df.select_dtypes(include="number").columns.tolist()

    low_cardinality_numeric_columns = [
        column
        for column in numeric_columns
        if df[column].nunique(dropna=True) <= 20
    ]

    continuous_numeric_columns = [
        column
        for column in numeric_columns
        if column not in low_cardinality_numeric_columns
    ]

    datetime_columns = df.select_dtypes(
        include=["datetime", "datetimetz"]
    ).columns.tolist()

    categorical_columns = [
        column
        for column in df.columns
        if column not in numeric_columns
        and column not in datetime_columns
    ]

    missing_cells = int(df.isna().sum().sum())
    total_cells = rows * columns

    missing_ratio = (
        missing_cells / total_cells
        if total_cells > 0
        else 0.0
    )

    duplicate_rows = int(df.duplicated().sum())
    duplicate_ratio = (
        duplicate_rows / rows
        if rows > 0
        else 0.0
    )

    memory_mb = float(
        df.memory_usage(deep=True).sum() / (1024 ** 2)
    )

    return {
        "rows": rows,
        "columns": columns,
        "numeric_columns": len(numeric_columns),
        "continuous_numeric_columns": len(continuous_numeric_columns),
        "low_cardinality_numeric_columns": len(low_cardinality_numeric_columns),
        "categorical_columns": len(categorical_columns),
        "datetime_columns": len(datetime_columns),
        "missing_cells": missing_cells,
        "missing_ratio": round(missing_ratio, 4),
        "duplicate_rows": duplicate_rows,
        "duplicate_ratio": round(duplicate_ratio, 4),
        "memory_mb": round(memory_mb, 2),
        "numeric_column_names": numeric_columns,
        "continuous_numeric_column_names": continuous_numeric_columns,
        "low_cardinality_numeric_column_names": low_cardinality_numeric_columns,
        "categorical_column_names": categorical_columns,
        "datetime_column_names": datetime_columns,
    }
