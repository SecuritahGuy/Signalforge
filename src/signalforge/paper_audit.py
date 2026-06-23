from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from signalforge.paper import mark_paper_positions, summarize_paper_account


@dataclass(frozen=True)
class PaperAuditConfig:
    initial_capital: float = 2_000.0
    max_sector_exposure: float = 0.35
    max_gross_exposure: float = 1.0
    max_price_staleness_days: int = 3
    account_tolerance: float = 0.01


def run_paper_realism_audit(
    ledger: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    account_summary: dict | None = None,
    as_of_date: str | pd.Timestamp | None = None,
    config: PaperAuditConfig | None = None,
) -> dict:
    cfg = config or PaperAuditConfig()
    normalized_prices = prices.copy()
    normalized_prices["date"] = pd.to_datetime(normalized_prices["date"])
    normalized_ledger = ledger.copy()
    if "planned_date" in normalized_ledger:
        normalized_ledger["planned_date"] = pd.to_datetime(normalized_ledger["planned_date"])
    if "fill_date" in normalized_ledger:
        normalized_ledger["fill_date"] = pd.to_datetime(normalized_ledger["fill_date"])

    calculated_account = summarize_paper_account(
        normalized_ledger,
        normalized_prices,
        initial_capital=cfg.initial_capital,
    )
    account = account_summary or calculated_account
    marks = mark_paper_positions(normalized_ledger, normalized_prices)
    active = normalized_ledger.loc[normalized_ledger["status"].isin(["planned", "open"])]
    open_marks = marks.loc[marks["status"] == "open"] if not marks.empty else marks
    latest_price_date = (
        normalized_prices["date"].max().normalize() if not normalized_prices.empty else pd.NaT
    )
    audit_date = pd.Timestamp(as_of_date).normalize() if as_of_date else latest_price_date

    findings = []
    _check_cash(findings, account)
    _check_duplicate_active_symbols(findings, active)
    _check_account_consistency(
        findings,
        account,
        calculated_account,
        tolerance=cfg.account_tolerance,
    )
    _check_price_freshness(
        findings,
        latest_price_date=latest_price_date,
        audit_date=audit_date,
        max_price_staleness_days=cfg.max_price_staleness_days,
    )
    _check_open_position_prices(findings, open_marks, latest_price_date=latest_price_date)
    _check_exposure_limits(
        findings,
        account=account,
        open_marks=open_marks,
        max_sector_exposure=cfg.max_sector_exposure,
        max_gross_exposure=cfg.max_gross_exposure,
    )
    _check_stale_planned_orders(findings, active, latest_price_date=latest_price_date)

    error_count = sum(finding["severity"] == "ERROR" for finding in findings)
    warning_count = sum(finding["severity"] == "WARN" for finding in findings)
    return {
        "status": "pass" if error_count == 0 else "fail",
        "error_count": error_count,
        "warning_count": warning_count,
        "finding_count": len(findings),
        "latest_price_date": None
        if pd.isna(latest_price_date)
        else latest_price_date.date().isoformat(),
        "audit_date": None if pd.isna(audit_date) else audit_date.date().isoformat(),
        "cash": float(account.get("cash", 0.0)),
        "equity": float(account.get("equity", 0.0)),
        "open_positions": int(account.get("open_positions", 0)),
        "planned_orders": int(account.get("planned_orders", 0)),
        "findings": findings,
    }


def render_audit_markdown(audit: dict) -> str:
    lines = [
        "# SignalForge Paper Realism Audit",
        "",
        f"Status: `{audit['status']}`",
        f"Latest price date: `{audit['latest_price_date']}`",
        f"Errors: `{audit['error_count']}`",
        f"Warnings: `{audit['warning_count']}`",
        "",
        "## Account",
        "",
        f"- Equity: `${audit['equity']:.2f}`",
        f"- Cash: `${audit['cash']:.2f}`",
        f"- Open positions: `{audit['open_positions']}`",
        f"- Planned orders: `{audit['planned_orders']}`",
        "",
        "## Findings",
        "",
    ]
    findings = audit["findings"]
    if not findings:
        lines.append("_No realism findings._")
    else:
        lines.extend(
            [
                "| severity | code | message |",
                "| --- | --- | --- |",
                *[
                    f"| {finding['severity']} | {finding['code']} | {finding['message']} |"
                    for finding in findings
                ],
            ]
        )
    lines.append("")
    return "\n".join(lines)


def _check_cash(findings: list[dict], account: dict) -> None:
    cash = float(account.get("cash", 0.0))
    if cash < -0.01:
        findings.append(
            _finding(
                "ERROR",
                "negative_cash",
                f"Paper account cash is negative: ${cash:.2f}.",
            )
        )


def _check_duplicate_active_symbols(findings: list[dict], active: pd.DataFrame) -> None:
    if active.empty:
        return
    duplicate_counts = active.groupby(active["symbol"].astype(str).str.upper()).size()
    duplicates = duplicate_counts.loc[duplicate_counts > 1]
    if not duplicates.empty:
        symbols = ", ".join(f"{symbol}({count})" for symbol, count in duplicates.items())
        findings.append(
            _finding(
                "ERROR",
                "duplicate_active_symbols",
                f"Active ledger contains duplicate symbols: {symbols}.",
            )
        )


def _check_account_consistency(
    findings: list[dict],
    account: dict,
    calculated: dict,
    *,
    tolerance: float,
) -> None:
    for key in ("cash", "equity"):
        reported = float(account.get(key, 0.0))
        expected = float(calculated.get(key, 0.0))
        if abs(reported - expected) > tolerance:
            findings.append(
                _finding(
                    "ERROR",
                    f"account_{key}_mismatch",
                    f"Reported {key} ${reported:.2f} differs from ledger value ${expected:.2f}.",
                )
            )


def _check_price_freshness(
    findings: list[dict],
    *,
    latest_price_date: pd.Timestamp,
    audit_date: pd.Timestamp,
    max_price_staleness_days: int,
) -> None:
    if pd.isna(latest_price_date) or pd.isna(audit_date):
        findings.append(_finding("ERROR", "missing_prices", "No price data was available."))
        return
    stale_days = (audit_date - latest_price_date).days
    if stale_days > max_price_staleness_days:
        findings.append(
            _finding(
                "WARN",
                "stale_price_file",
                f"Latest price file is {stale_days} calendar days behind the audit date.",
            )
        )


def _check_open_position_prices(
    findings: list[dict],
    open_marks: pd.DataFrame,
    *,
    latest_price_date: pd.Timestamp,
) -> None:
    if open_marks.empty:
        return
    stale = open_marks.loc[
        open_marks["latest_price_date"].isna()
        | (pd.to_datetime(open_marks["latest_price_date"]) < latest_price_date)
    ]
    if not stale.empty:
        symbols = ", ".join(sorted(stale["symbol"].astype(str).unique()))
        findings.append(
            _finding(
                "WARN",
                "stale_open_position_prices",
                f"Open positions are missing latest marks: {symbols}.",
            )
        )


def _check_exposure_limits(
    findings: list[dict],
    *,
    account: dict,
    open_marks: pd.DataFrame,
    max_sector_exposure: float,
    max_gross_exposure: float,
) -> None:
    if open_marks.empty:
        return
    equity = float(account.get("equity", 0.0))
    if equity <= 0:
        findings.append(_finding("ERROR", "non_positive_equity", "Paper equity is not positive."))
        return
    gross_exposure = float(open_marks["mark_value"].sum())
    gross_ratio = gross_exposure / equity
    if gross_ratio > max_gross_exposure:
        findings.append(
            _finding(
                "WARN",
                "gross_exposure_high",
                f"Gross exposure is {gross_ratio:.1%} of equity.",
            )
        )
    sector_ratios = open_marks.groupby("sector", dropna=False)["mark_value"].sum() / equity
    high_sectors = sector_ratios.loc[sector_ratios > max_sector_exposure]
    if not high_sectors.empty:
        sectors = ", ".join(f"{sector}: {ratio:.1%}" for sector, ratio in high_sectors.items())
        findings.append(
            _finding(
                "WARN",
                "sector_exposure_high",
                f"Sector exposure exceeds limit: {sectors}.",
            )
        )


def _check_stale_planned_orders(
    findings: list[dict],
    active: pd.DataFrame,
    *,
    latest_price_date: pd.Timestamp,
) -> None:
    planned = active.loc[active["status"] == "planned"]
    if planned.empty or pd.isna(latest_price_date):
        return
    stale_symbols = []
    for _, row in planned.iterrows():
        planned_date = pd.Timestamp(row["planned_date"])
        if len(pd.bdate_range(planned_date, latest_price_date)) > 3:
            stale_symbols.append(str(row["symbol"]))
    if stale_symbols:
        findings.append(
            _finding(
                "WARN",
                "stale_planned_orders",
                f"Planned orders have waited more than two sessions: {', '.join(stale_symbols)}.",
            )
        )


def _finding(severity: str, code: str, message: str) -> dict:
    return {"severity": severity, "code": code, "message": message}
