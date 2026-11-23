"""Tests for the episode grouping, scoring and a small end-to-end fit."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from detect_overconsumption import evaluate, find_episodes, fit_office
from postoffice_common import public_holidays


def _predictions(flagged_positions: list[int], days: int = 20) -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=days, freq="D")
    flagged = np.zeros(days, dtype=bool)
    flagged[flagged_positions] = True
    return pd.DataFrame(
        {
            "ds": dates,
            "y": 100.0,
            "yhat": 90.0,
            "excess_pct": np.where(flagged, 0.11, 0.0),
            "flagged": flagged,
            "post_office_id": "PO-TEST",
            "post_office_name": "Test Office",
        }
    )


def test_find_episodes_merges_short_gaps_and_numbers_contiguously() -> None:
    predictions = _predictions([2, 3, 5, 12])

    episodes = find_episodes(predictions, merge_gap_days=1, min_days=1)

    assert list(episodes["episode_id"]) == ["PO-TEST-E01", "PO-TEST-E02"]
    assert list(episodes["detected_days"]) == [3, 1]
    assert episodes["start"].iloc[0] == pd.Timestamp("2024-01-03")
    assert episodes["end"].iloc[0] == pd.Timestamp("2024-01-06")


def test_find_episodes_filters_short_episodes_after_numbering() -> None:
    predictions = _predictions([2, 3, 12])

    episodes = find_episodes(predictions, merge_gap_days=1, min_days=2)

    assert list(episodes["episode_id"]) == ["PO-TEST-E01"]
    assert episodes["detected_days"].tolist() == [2]


def test_find_episodes_without_flags_returns_empty_frame() -> None:
    episodes = find_episodes(_predictions([]))

    assert episodes.empty
    assert "episode_id" in episodes.columns


def test_evaluate_scores_days_and_episodes() -> None:
    dates = pd.date_range("2024-01-01", periods=10, freq="D")
    flagged = np.zeros(10, dtype=bool)
    flagged[[4, 5]] = True
    predictions = pd.DataFrame(
        {
            "ds": dates,
            "y": 100.0,
            "yhat": 90.0,
            "excess_pct": np.where(flagged, 0.11, 0.0),
            "flagged": flagged,
            "post_office_id": ["PO-A"] * 10,
            "post_office_name": ["A"] * 10,
        }
    )
    episodes = pd.DataFrame(
        {
            "episode_id": ["PO-A-E01", "PO-B-E01"],
            "post_office_id": ["PO-A", "PO-B"],
            "post_office_name": ["A", "B"],
            "start": [dates[4], dates[0]],
            "end": [dates[5], dates[0]],
        }
    )
    events = pd.DataFrame(
        {
            "post_office_id": ["PO-A"],
            "cause": ["HVAC stuck on"],
            "start_date": [dates[4]],
            "end_date": [dates[6]],
            "days": [3],
            "excess_pct": [0.5],
        }
    )

    metrics, checks = evaluate(predictions, episodes, events)

    assert metrics["day_precision"] == pytest.approx(1.0)
    assert metrics["day_recall"] == pytest.approx(2 / 3)
    assert metrics["event_recall"] == pytest.approx(1.0)
    assert metrics["event_precision"] == pytest.approx(0.5)
    assert checks["detected"].tolist() == [True]


def test_fit_office_flags_an_injected_spike() -> None:
    rng = np.random.default_rng(42)
    dates = pd.date_range("2023-01-01", periods=730, freq="D")
    t = np.arange(len(dates))
    clean = (
        250.0
        * (1.0 + 0.15 * np.cos(2 * np.pi * (t - 15) / 365.25))
        * np.asarray([1.0, 1.0, 1.0, 1.0, 1.0, 0.85, 0.62])[dates.dayofweek]
        * np.clip(rng.normal(1.0, 0.03, len(dates)), 0.9, 1.1)
    )
    spike = slice(500, 506)
    kwh = clean.copy()
    kwh[spike] = clean[spike] * 1.4
    series = pd.DataFrame({"ds": dates, "y": kwh})

    predictions = fit_office(series, public_holidays([2023, 2024]), interval_width=0.99)
    predictions["post_office_id"] = "PO-TEST"
    predictions["post_office_name"] = "Test Office"
    episodes = find_episodes(predictions, merge_gap_days=2, min_days=1)

    flagged_days = predictions.loc[predictions["flagged"], "ds"]
    assert len(flagged_days) >= 5
    assert flagged_days.between(dates[spike.start], dates[spike.stop - 1]).all()
    assert len(episodes) == 1
    assert episodes["excess_kwh"].iloc[0] > 0
