from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from signalforge.run_manifest import build_run_manifest, file_metadata, git_code_metadata


def test_file_metadata_hashes_small_files(tmp_path):
    source = tmp_path / "input.csv"
    source.write_text("symbol\nAAA\n")

    metadata = file_metadata(source, hash_size_limit_bytes=1_000)

    assert metadata["path"] == str(source)
    assert metadata["exists"] is True
    assert metadata["size_bytes"] == source.stat().st_size
    assert metadata["modified_at_utc"] is not None
    assert metadata["sha256"] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert metadata["sha256_skipped"] is False
    assert metadata["sha256_skip_reason"] is None


def test_file_metadata_skips_large_file_hashing(tmp_path):
    source = tmp_path / "large.csv"
    source.write_text("abcdef")

    metadata = file_metadata(source, hash_size_limit_bytes=3)

    assert metadata["exists"] is True
    assert metadata["size_bytes"] == 6
    assert metadata["modified_at_utc"] is not None
    assert metadata["sha256"] is None
    assert metadata["sha256_skipped"] is True
    assert metadata["sha256_skip_reason"] == "size_above_threshold"


def test_git_metadata_falls_back_when_git_is_unavailable(monkeypatch):
    def unavailable_git(*args, **kwargs):
        raise FileNotFoundError

    monkeypatch.setattr("signalforge.run_manifest.subprocess.run", unavailable_git)

    assert git_code_metadata(cwd="/not/a/repo") == {
        "git_commit": None,
        "git_branch": None,
        "git_dirty": None,
    }


def test_build_run_manifest_includes_file_metadata_and_outputs(tmp_path):
    source = tmp_path / "research_frame.csv"
    source.write_text("symbol\nAAA\n")
    artifact = tmp_path / "summary.json"
    artifact.write_text("{}\n")

    manifest = build_run_manifest(
        run_type="discovery",
        created_at_utc="2026-05-21T00:00:00Z",
        as_of_date="2026-05-21",
        parameters={"top_n_per_lane": 25, "horizons": (5, 20)},
        inputs={"research_frame": source, "watchlist": None},
        outputs={"summary": artifact},
        code_cwd=tmp_path,
    )

    assert manifest["run_type"] == "discovery"
    assert manifest["created_at_utc"] == "2026-05-21T00:00:00Z"
    assert manifest["as_of_date"] == "2026-05-21"
    assert manifest["parameters"]["horizons"] == [5, 20]
    assert manifest["inputs"]["research_frame"] == str(source)
    assert manifest["input_file_metadata"]["research_frame"]["exists"] is True
    assert manifest["input_file_metadata"]["watchlist"]["exists"] is False
    assert manifest["outputs"] == ["summary.json", "manifest.json"]
    assert manifest["environment"]["package_name"] == "signalforge"
    assert {"git_commit", "git_branch", "git_dirty"}.issubset(manifest["code_metadata"])


def test_run_stock_discovery_cli_writes_manifest(tmp_path):
    frame_path = tmp_path / "research_frame.csv"
    output_dir = tmp_path / "discovery_output"
    _synthetic_discovery_frame().to_csv(frame_path, index=False)

    completed = subprocess.run(
        [
            sys.executable,
            "scripts/run_stock_discovery.py",
            "--research-frame",
            str(frame_path),
            "--output-dir",
            str(output_dir),
            "--as-of-date",
            "2024-12-31",
            "--top-n",
            "1",
            "--no-market-cap-filter",
            "--min-price",
            "0",
            "--min-avg-dollar-volume-20d",
            "0",
        ],
        check=True,
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
    )

    assert "wrote discovery artifacts" in completed.stdout
    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["run_type"] == "discovery"
    assert manifest["as_of_date"] == "2024-12-31"
    assert manifest["parameters"]["top_n_per_lane"] == 1
    assert manifest["parameters"]["no_market_cap_filter"] is True
    assert manifest["inputs"]["research_frame"] == str(frame_path)
    assert manifest["input_file_metadata"]["research_frame"]["exists"] is True
    assert manifest["input_file_metadata"]["research_frame"]["size_bytes"] > 0
    assert manifest["input_file_metadata"]["research_frame"]["modified_at_utc"] is not None
    assert {
        "summary.json",
        "candidates.csv",
        "report.md",
        "manifest.json",
    }.issubset(set(manifest["outputs"]))


def _synthetic_discovery_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            _candidate("AAA", return_60d=0.25, relative_volume_20d=1.8),
            _candidate("BBB", return_60d=0.12, relative_volume_20d=1.1),
        ]
    )


def _candidate(
    symbol: str,
    *,
    return_60d: float,
    relative_volume_20d: float,
) -> dict:
    return {
        "date": pd.Timestamp("2024-12-31"),
        "symbol": symbol,
        "name": symbol,
        "sector": "Technology",
        "industry": "Software",
        "adj_close": 40.0,
        "avg_dollar_volume_20d": 25_000_000,
        "return_1d": 0.02,
        "return_5d": 0.04,
        "return_20d": 0.12,
        "return_60d": return_60d,
        "stock_minus_market_return_20d": 0.05,
        "stock_minus_sector_return_20d": 0.04,
        "stock_minus_sector_return_60d": return_60d / 2,
        "sector_return_20d": 0.08,
        "sector_return_60d": 0.12,
        "sector_rank_return_20d": 0.8,
        "sector_rank_return_60d": 0.8,
        "sector_rank_momentum_20d": 0.8,
        "relative_volume_20d": relative_volume_20d,
        "volume_change_5d": 0.05,
        "volume_zscore_20d": 0.5,
        "price_above_sma_20": 0.03,
        "price_above_sma_50": 0.05,
        "price_above_sma_200": 0.02,
        "distance_from_52w_high": -0.03,
        "drawdown_60d": -0.08,
    }
