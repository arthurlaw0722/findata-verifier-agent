from __future__ import annotations

from typing import Any


def _number(value: Any) -> float | None:
    """Safely convert a financial value to float."""
    if value is None:
        return None

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _ratio(numerator: Any, denominator: Any) -> float | None:
    numerator = _number(numerator)
    denominator = _number(denominator)

    if numerator is None or denominator is None or denominator == 0:
        return None

    return numerator / denominator


def _growth(current: Any, previous: Any) -> float | None:
    """
    Return percentage growth.

    Example:
    previous = 100
    current = 120
    result = 20.0
    """
    current = _number(current)
    previous = _number(previous)

    if current is None or previous is None or previous == 0:
        return None

    return ((current - previous) / abs(previous)) * 100


def _signal(
    area: str,
    status: str,
    title: str,
    interpretation: str,
    action: str,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    return {
        "area": area,
        "status": status,
        "title": title,
        "interpretation": interpretation,
        "analyst_action": action,
        "metrics": metrics,
    }


def accounting_integrity_signal(
    current: dict[str, Any],
    tolerance_pct: float = 0.5,
) -> dict[str, Any]:
    """
    Check whether:
    Assets ≈ Liabilities + Equity
    """

    assets = _number(current.get("total_assets"))
    liabilities = _number(current.get("total_liabilities"))
    equity = _number(current.get("total_equity"))

    if None in (assets, liabilities, equity):
        return _signal(
            area="Accounting Integrity",
            status="Review",
            title="Insufficient balance-sheet data",
            interpretation=(
                "Assets, liabilities, and equity are required to test "
                "the accounting equation."
            ),
            action=(
                "Confirm that total assets, total liabilities, and "
                "total equity were extracted correctly."
            ),
            metrics={},
        )

    expected_assets = liabilities + equity
    difference = assets - expected_assets

    denominator = max(abs(assets), 1.0)
    difference_pct = abs(difference) / denominator * 100

    if difference_pct <= tolerance_pct:
        status = "Normal"
        interpretation = (
            "The balance sheet reconciles within the configured tolerance."
        )
        action = (
            "No immediate reconciliation issue detected. Continue with "
            "the remaining financial checks."
        )
    elif difference_pct <= 2.0:
        status = "Review"
        interpretation = (
            "The accounting equation shows a small reconciliation difference."
        )
        action = (
            "Review XBRL tags, rounding differences, minority interests, "
            "and extraction consistency."
        )
    else:
        status = "High Attention"
        interpretation = (
            "Reported assets do not reconcile with liabilities plus equity "
            "within a reasonable tolerance."
        )
        action = (
            "Validate the source filing and extracted accounting facts "
            "before relying on downstream analysis."
        )

    return _signal(
        area="Accounting Integrity",
        status=status,
        title="Balance-sheet reconciliation",
        interpretation=interpretation,
        action=action,
        metrics={
            "assets": assets,
            "liabilities_plus_equity": expected_assets,
            "difference": difference,
            "difference_pct": round(difference_pct, 4),
        },
    )


def working_capital_signal(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    revenue_growth = _growth(
        current.get("revenue"),
        previous.get("revenue"),
    )

    receivables_growth = _growth(
        current.get("accounts_receivable"),
        previous.get("accounts_receivable"),
    )

    inventory_growth = _growth(
        current.get("inventory"),
        previous.get("inventory"),
    )

    ar_gap = (
        receivables_growth - revenue_growth
        if revenue_growth is not None
        and receivables_growth is not None
        else None
    )

    inventory_gap = (
        inventory_growth - revenue_growth
        if revenue_growth is not None
        and inventory_growth is not None
        else None
    )

    status = "Normal"
    reasons = []

    if ar_gap is not None:
        if ar_gap > 25:
            status = "High Attention"
            reasons.append(
                "Accounts receivable is growing much faster than revenue."
            )
        elif ar_gap > 10:
            status = "Review"
            reasons.append(
                "Accounts receivable is growing faster than revenue."
            )

    if inventory_gap is not None:
        if inventory_gap > 30:
            status = "High Attention"
            reasons.append(
                "Inventory growth materially exceeds revenue growth."
            )
        elif inventory_gap > 15:
            if status == "Normal":
                status = "Review"
            reasons.append(
                "Inventory growth is running ahead of revenue growth."
            )

    if not reasons:
        interpretation = (
            "Receivables and inventory movements do not show a material "
            "divergence from revenue under the current heuristic rules."
        )
        action = (
            "Continue monitoring working-capital trends across future periods."
        )
    else:
        interpretation = " ".join(reasons)
        action = (
            "Review days sales outstanding, receivables ageing, inventory "
            "turnover, customer payment terms, and related management commentary."
        )

    return _signal(
        area="Working Capital",
        status=status,
        title="Revenue vs working-capital growth",
        interpretation=interpretation,
        action=action,
        metrics={
            "revenue_growth_pct": (
                round(revenue_growth, 2)
                if revenue_growth is not None
                else None
            ),
            "receivables_growth_pct": (
                round(receivables_growth, 2)
                if receivables_growth is not None
                else None
            ),
            "inventory_growth_pct": (
                round(inventory_growth, 2)
                if inventory_growth is not None
                else None
            ),
            "receivables_vs_revenue_gap_pp": (
                round(ar_gap, 2)
                if ar_gap is not None
                else None
            ),
        },
    )


def earnings_quality_signal(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    net_income_growth = _growth(
        current.get("net_income"),
        previous.get("net_income"),
    )

    cfo_growth = _growth(
        current.get("operating_cash_flow"),
        previous.get("operating_cash_flow"),
    )

    status = "Normal"

    if net_income_growth is None or cfo_growth is None:
        return _signal(
            area="Earnings Quality",
            status="Review",
            title="Earnings vs operating cash flow",
            interpretation=(
                "There is not enough comparable data to assess earnings "
                "and operating cash-flow growth."
            ),
            action=(
                "Check that comparable net income and operating cash-flow "
                "periods are available."
            ),
            metrics={
                "net_income_growth_pct": net_income_growth,
                "operating_cash_flow_growth_pct": cfo_growth,
            },
        )

    divergence = net_income_growth - cfo_growth

    if net_income_growth > 15 and cfo_growth < 0:
        status = "High Attention"
        interpretation = (
            "Reported earnings are increasing while operating cash flow "
            "is declining."
        )
        action = (
            "Review working-capital movements, non-cash adjustments, "
            "receivables, and revenue-recognition disclosures."
        )
    elif divergence > 25:
        status = "Review"
        interpretation = (
            "Net income growth is materially stronger than operating "
            "cash-flow growth."
        )
        action = (
            "Investigate cash conversion and the main reconciliation "
            "items between earnings and operating cash flow."
        )
    else:
        interpretation = (
            "Earnings growth and operating cash-flow growth appear "
            "reasonably aligned."
        )
        action = (
            "No major earnings-quality divergence detected under the "
            "current heuristic rules."
        )

    return _signal(
        area="Earnings Quality",
        status=status,
        title="Earnings vs operating cash flow",
        interpretation=interpretation,
        action=action,
        metrics={
            "net_income_growth_pct": round(net_income_growth, 2),
            "operating_cash_flow_growth_pct": round(cfo_growth, 2),
            "growth_divergence_pp": round(divergence, 2),
        },
    )


def liquidity_leverage_signal(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    current_ratio = _ratio(
        current.get("current_assets"),
        current.get("current_liabilities"),
    )

    debt_growth = _growth(
        current.get("total_debt"),
        previous.get("total_debt"),
    )

    cash_growth = _growth(
        current.get("cash_and_equivalents"),
        previous.get("cash_and_equivalents"),
    )

    status = "Normal"
    reasons = []

    if current_ratio is not None and current_ratio < 1.0:
        status = "Review"
        reasons.append(
            "Current assets are below current liabilities."
        )

    if debt_growth is not None and debt_growth > 25:
        if current_ratio is not None and current_ratio < 1.0:
            status = "High Attention"
        elif status == "Normal":
            status = "Review"

        reasons.append(
            "Debt has increased materially compared with the previous period."
        )

    if cash_growth is not None and cash_growth < -20:
        if status == "Normal":
            status = "Review"

        reasons.append(
            "Cash and cash equivalents have declined materially."
        )

    if not reasons:
        interpretation = (
            "No major liquidity or leverage pressure was detected "
            "under the current heuristic rules."
        )
        action = (
            "Continue monitoring liquidity, debt, and cash trends."
        )
    else:
        interpretation = " ".join(reasons)
        action = (
            "Review debt maturity, liquidity facilities, current liabilities, "
            "cash generation, and financing commentary."
        )

    return _signal(
        area="Liquidity & Leverage",
        status=status,
        title="Liquidity and financing pressure",
        interpretation=interpretation,
        action=action,
        metrics={
            "current_ratio": (
                round(current_ratio, 2)
                if current_ratio is not None
                else None
            ),
            "debt_growth_pct": (
                round(debt_growth, 2)
                if debt_growth is not None
                else None
            ),
            "cash_growth_pct": (
                round(cash_growth, 2)
                if cash_growth is not None
                else None
            ),
        },
    )


def analyze_financial_intelligence(
    current: dict[str, Any],
    previous: dict[str, Any],
) -> dict[str, Any]:
    """
    Run the first Financial Intelligence rule engine.

    The rules identify analyst-review signals; they do not claim fraud,
    manipulation, or accounting misconduct.
    """

    signals = [
        accounting_integrity_signal(current),
        working_capital_signal(current, previous),
        earnings_quality_signal(current, previous),
        liquidity_leverage_signal(current, previous),
    ]

    status_counts = {
        "Normal": sum(s["status"] == "Normal" for s in signals),
        "Review": sum(s["status"] == "Review" for s in signals),
        "High Attention": sum(
            s["status"] == "High Attention"
            for s in signals
        ),
    }

    if status_counts["High Attention"] > 0:
        decision = "HIGH ATTENTION"
        summary = (
            "One or more material financial signals require analyst review "
            "before relying on the data for downstream decisions."
        )
    elif status_counts["Review"] > 0:
        decision = "REVIEW REQUIRED"
        summary = (
            "The financial data is broadly usable, but one or more signals "
            "should be investigated before drawing strong conclusions."
        )
    else:
        decision = "PROCEED"
        summary = (
            "No material issues were identified by the current financial "
            "integrity and analytical rules."
        )

    priorities = [
        {
            "area": signal["area"],
            "status": signal["status"],
            "action": signal["analyst_action"],
        }
        for signal in signals
        if signal["status"] != "Normal"
    ]

    return {
        "decision": decision,
        "summary": summary,
        "status_counts": status_counts,
        "signals": signals,
        "review_priorities": priorities,
        "disclaimer": (
            "Signals are analytical heuristics intended to support review. "
            "They are not evidence of fraud, misconduct, or investment advice."
        ),
    }
