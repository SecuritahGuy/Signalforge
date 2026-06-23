import pandas as pd

from scripts.update_paper_ledger import _append_new_orders, _load_exit_rules_config


def test_append_new_orders_only_adds_unique_planned_active_symbols():
    existing = pd.DataFrame(
        {
            "order_id": ["2024-01-01-A-001", "2024-01-01-B-002"],
            "status": ["open", "planned"],
            "planned_date": pd.to_datetime(["2024-01-01", "2024-01-01"]),
            "symbol": ["A", "B"],
        }
    )
    planned = pd.DataFrame(
        {
            "order_id": [
                "2024-01-01-A-003",
                "2024-01-01-B-004",
                "2024-01-01-C-005",
                "2024-01-01-D-006",
            ],
            "status": ["planned", "planned", "planned", "skipped"],
            "planned_date": pd.to_datetime(
                ["2024-01-01", "2024-01-01", "2024-01-01", "2024-01-01"]
            ),
            "symbol": ["A", "B", "C", "D"],
        }
    )

    ledger = _append_new_orders(existing, planned)

    assert list(ledger["symbol"]) == ["A", "B", "C"]


def test_append_new_orders_rejects_same_symbol_date_even_when_order_id_changes():
    existing = pd.DataFrame(
        {
            "order_id": ["2024-01-01-A-001"],
            "status": ["closed"],
            "planned_date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
        }
    )
    planned = pd.DataFrame(
        {
            "order_id": ["2024-01-01-A-009"],
            "status": ["planned"],
            "planned_date": pd.to_datetime(["2024-01-01"]),
            "symbol": ["A"],
        }
    )

    ledger = _append_new_orders(existing, planned)

    assert len(ledger) == 1


def test_load_exit_rules_config_parses_nested_rules(tmp_path):
    path = tmp_path / "paper.yaml"
    path.write_text(
        "\n".join(
            [
                "exit_rules:",
                "  horizon_days: 15",
                "  stop_loss:",
                "    enabled: true",
                "    pct: -0.07",
                "  trailing_stop:",
                "    enabled: true",
                "    activate_at_return: 0.10",
                "    trail_from_high_pct: -0.05",
                "  score_deterioration:",
                "    enabled: true",
                "    min_days_held: 4",
                "    exit_below_score: 0.004",
                "    exit_if_score_declines_pct: 0.50",
                "  rebalance:",
                "    enabled: false",
                "  time_decay:",
                "    enabled: true",
                "    half_life_days: 15",
                "    min_days_hold: 3",
                "    min_score_for_decay: 0.003",
            ]
        )
    )

    config = _load_exit_rules_config(str(path), horizon_days=20)

    assert config.horizon_days == 15
    assert config.stop_loss.enabled is True
    assert config.stop_loss.pct == -0.07
    assert config.trailing_stop.enabled is True
    assert config.trailing_stop.activate_at_return == 0.10
    assert config.trailing_stop.trail_from_high_pct == -0.05
    assert config.score_deterioration.enabled is True
    assert config.score_deterioration.min_days_held == 4
    assert config.score_deterioration.exit_below_score == 0.004
    assert config.score_deterioration.exit_if_score_declines_pct == 0.50
    assert config.rebalance.enabled is False
    assert config.time_decay.enabled is True
    assert config.time_decay.half_life_days == 15
    assert config.time_decay.min_days_hold == 3
    assert config.time_decay.min_score_for_decay == 0.003
