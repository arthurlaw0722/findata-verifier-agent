import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


MAX_SCATTER_POINTS = 30000
MAX_DISTRIBUTION_POINTS = 50000


def _sample_for_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """Limit browser payload for distribution charts."""

    if len(df) > MAX_DISTRIBUTION_POINTS:
        return df.sample(
            n=MAX_DISTRIBUTION_POINTS,
            random_state=42,
        )

    return df


def histogram_chart(
    df: pd.DataFrame,
    column: str,
    color_column: str | None = None,
):
    """Create an interactive histogram."""

    plot_df = _sample_for_distribution(df)
    color = color_column if color_column in plot_df.columns else None

    fig = px.histogram(
        plot_df,
        x=column,
        color=color,
        marginal="box",
        title=f"Distribution of {column}",
    )

    fig.update_layout(
        xaxis_title=column,
        yaxis_title="Count",
    )

    return fig


def box_chart(
    df: pd.DataFrame,
    column: str,
    group_column: str | None = None,
):
    """Create an interactive box plot."""

    plot_df = _sample_for_distribution(df)
    group = group_column if group_column in plot_df.columns else None

    fig = px.box(
        plot_df,
        x=group,
        y=column,
        points="outliers",
        title=f"Box Plot of {column}",
    )

    return fig


def scatter_chart(
    df: pd.DataFrame,
    x_column: str,
    y_column: str,
    color_column: str | None = None,
):
    """Create a scatter plot with sampling for large datasets."""

    if len(df) > MAX_SCATTER_POINTS:
        plot_df = df.sample(
            n=MAX_SCATTER_POINTS,
            random_state=42,
        )
    else:
        plot_df = df

    color = color_column if color_column in plot_df.columns else None

    fig = px.scatter(
        plot_df,
        x=x_column,
        y=y_column,
        color=color,
        opacity=0.6,
        title=f"{y_column} vs {x_column}",
    )

    return fig


def correlation_heatmap(df: pd.DataFrame):
    """Create a correlation heatmap for numeric columns."""

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.empty:
        return None

    correlation = numeric_df.corr()

    fig = px.imshow(
        correlation,
        text_auto=False,
        aspect="auto",
        title="Correlation Heatmap",
    )

    return fig


def missing_values_chart(df: pd.DataFrame):
    """Create a bar chart of missing values by column."""

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        return None

    fig = go.Figure(
        data=[
            go.Bar(
                x=missing.index,
                y=missing.values,
            )
        ]
    )

    fig.update_layout(
        title="Missing Values by Column",
        xaxis_title="Column",
        yaxis_title="Missing Values",
    )

    return fig
