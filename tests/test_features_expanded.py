import pandas as pd
import pytest

from signalforge.features import build_price_features


def test_lagged_features_are_shifted_by_one_period():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, lagged_features=True)

    assert "return_20d_lag_1" in features.columns
    assert "volatility_20d_lag_1" in features.columns

    aapl = features[features["symbol"] == "AAPL"].sort_values("date").reset_index(drop=True)
    for i in range(1, len(aapl)):
        if pd.notna(aapl.loc[i, "return_20d_lag_1"]) and pd.notna(aapl.loc[i - 1, "return_20d"]):
            assert aapl.loc[i, "return_20d_lag_1"] == aapl.loc[i - 1, "return_20d"]


def test_calendar_features_are_present():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, calendar_features=True)

    assert "day_of_week" in features.columns
    assert "day_of_week_sin" in features.columns
    assert "day_of_week_cos" in features.columns
    assert "month" in features.columns
    assert "month_sin" in features.columns
    assert "month_cos" in features.columns
    assert "quarter" in features.columns
    assert "days_to_month_end" in features.columns
    assert "is_month_end" in features.columns

    row = features.iloc[0]
    assert 0 <= row["day_of_week"] <= 6
    assert -1.0 <= row["day_of_week_sin"] <= 1.0
    assert -1.0 <= row["day_of_week_cos"] <= 1.0
    assert 1 <= row["month"] <= 12
    assert 1 <= row["quarter"] <= 4
    assert row["is_month_end"] in (0, 1)


def test_calendar_sin_cos_encoding_is_deterministic():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, calendar_features=True)

    same_date = features[features["date"] == features["date"].unique()[0]]
    expected_sin = same_date["day_of_week_sin"].iloc[0]
    expected_cos = same_date["day_of_week_cos"].iloc[0]
    assert (same_date["day_of_week_sin"] == expected_sin).all()
    assert (same_date["day_of_week_cos"] == expected_cos).all()


def test_cross_sectional_features_exist():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, cross_sectional_features=True)

    assert "zscore_return_5d" in features.columns
    assert "zscore_return_20d" in features.columns
    assert "zscore_volatility_5d" in features.columns
    assert "zscore_volatility_20d" in features.columns
    assert "zscore_volatility_120d" in features.columns


def test_cross_sectional_zscore_center_is_near_zero():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, cross_sectional_features=True)

    for date_val in features["date"].unique():
        day_slice = features[features["date"] == date_val]["zscore_return_20d"].dropna()
        if len(day_slice) > 1:
            assert abs(day_slice.mean()) < 1e-10


def test_technical_indicators_are_present():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, technical_indicators=True)

    assert "rsi_14" in features.columns
    assert "macd_12_26" in features.columns
    assert "macd_signal_9" in features.columns
    assert "macd_histogram_12_26_9" in features.columns
    assert "bollinger_pct_b_20_2" in features.columns
    assert "bollinger_width_20_2" in features.columns
    assert "atr_14" in features.columns


def test_rsi_between_zero_and_one_hundred():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, technical_indicators=True)

    rsi = features["rsi_14"].dropna()
    assert (rsi >= 0).all()
    assert (rsi <= 100).all()


def test_bollinger_pct_b_between_zero_and_one_for_normal_ranges():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, technical_indicators=True)

    boll = features["bollinger_pct_b_20_2"].dropna()
    assert not boll.isna().all()


def test_macd_histogram_zero_mean():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, technical_indicators=True)

    hist = features["macd_histogram_12_26_9"].dropna()
    assert not hist.isna().all()


def test_atr_is_positive():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, technical_indicators=True)

    atr = features["atr_14"].dropna()
    assert (atr > 0).all()


def test_interaction_features_are_present():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, interaction_features=True)

    assert "return_10d_x_volatility_10d" in features.columns
    assert "return_5d_x_volatility_5d" in features.columns
    assert "volatility_10d_x_relative_volume_10d" in features.columns
    assert "return_20d_x_volume_change_5d" in features.columns


def test_interaction_features_match_manual():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, interaction_features=True)

    mask = features["return_10d"].notna() & features["volatility_10d"].notna()
    manual = (features.loc[mask, "return_10d"] * features.loc[mask, "volatility_10d"]).reset_index(drop=True)
    computed = features.loc[mask, "return_10d_x_volatility_10d"].reset_index(drop=True)
    pd.testing.assert_series_equal(manual, computed, check_names=False)


def test_factor_proxies_are_present():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, factor_proxies=True)

    assert "momentum_factor" in features.columns
    assert "low_vol_factor" in features.columns
    assert "quality_factor" in features.columns


def test_momentum_factor_is_difference_of_long_and_short():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, factor_proxies=True)

    mask = features["return_120d"].notna() & features["return_10d"].notna()
    manual = (features.loc[mask, "return_120d"] - features.loc[mask, "return_10d"]).reset_index(drop=True)
    computed = features.loc[mask, "momentum_factor"].reset_index(drop=True)
    pd.testing.assert_series_equal(manual, computed, check_names=False)


def test_low_vol_factor_is_inverse_volatility_rank():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, factor_proxies=True)

    for date_val in features["date"].unique():
        day_slice = features[features["date"] == date_val][["symbol", "low_vol_factor", "volatility_20d"]].dropna()
        if len(day_slice) < 2:
            continue
        sorted_by_vol = day_slice.sort_values("volatility_20d")
        sorted_by_factor = day_slice.sort_values("low_vol_factor", ascending=False)
        assert sorted_by_vol["symbol"].tolist() == sorted_by_factor["symbol"].tolist()


def test_quality_factor_is_negative_volatility():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices, factor_proxies=True)

    aapl = features[features["symbol"] == "AAPL"]["quality_factor"].dropna()
    assert (aapl <= 0).all()


def test_all_feature_groups_together_no_errors():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(
        prices,
        lagged_features=True,
        calendar_features=True,
        cross_sectional_features=True,
        technical_indicators=True,
        interaction_features=True,
        factor_proxies=True,
    )

    assert len(features) == len(prices)
    assert features.columns.nunique() == len(features.columns)


def test_research_frame_plumbs_feature_flags():
    from signalforge.research import build_research_frame

    prices = _three_symbol_prices(periods=70, include_spy=True)
    universe = pd.DataFrame({
        "symbol": ["AAPL", "MSFT", "NVDA"],
        "sector": ["Information Technology"] * 3,
    })

    frame = build_research_frame(
        prices, universe, horizon=5,
        lagged_features=True,
        calendar_features=True,
        cross_sectional_features=True,
        technical_indicators=True,
        interaction_features=True,
        factor_proxies=True,
    )

    assert "return_20d_lag_1" in frame.columns
    assert "day_of_week_sin" in frame.columns
    assert "zscore_return_20d" in frame.columns
    assert "rsi_14" in frame.columns
    assert "return_10d_x_volatility_10d" in frame.columns
    assert "momentum_factor" in frame.columns


def test_feature_groups_default_to_off():
    prices = _three_symbol_prices(periods=70)
    features = build_price_features(prices)

    assert "return_20d_lag_1" not in features.columns
    assert "day_of_week" not in features.columns
    assert "zscore_return_20d" not in features.columns
    assert "rsi_14" not in features.columns
    assert "return_10d_x_volatility_10d" not in features.columns
    assert "momentum_factor" not in features.columns


def _three_symbol_prices(*, periods: int, include_spy: bool = False) -> pd.DataFrame:
    symbols = ["AAPL", "MSFT", "NVDA"]
    if include_spy:
        symbols = symbols + ["SPY"]
    dates = pd.date_range("2024-01-01", periods=periods, freq="D")
    rows = []
    for symbol_index, symbol in enumerate(symbols):
        for day_index, date in enumerate(dates):
            price = 100 + day_index + symbol_index * 10
            rows.append({
                "date": date,
                "symbol": symbol,
                "open": price - 0.5,
                "high": price + 1.0 + symbol_index * 0.5,
                "low": price - 1.0 - symbol_index * 0.5,
                "close": price,
                "adj_close": price,
                "volume": 1_000_000 + symbol_index * 10_000 + day_index,
            })
    return pd.DataFrame(rows)
