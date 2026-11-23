"""Shared definitions for the post-office electricity toy dataset.

The synthetic-data generator and the overconsumption detector both import
this module so the post-office network and the holiday calendar stay in
sync between the two scripts.
"""

from __future__ import annotations

import calendar
from datetime import date

import pandas as pd

# The toy network: 12 post offices across 4 regions and 3 size classes.
# ``base_kwh`` is an average-day consumption in a neutral season (kWh/day).
POST_OFFICES: list[dict[str, object]] = [
    {"id": "PO-001", "name": "Harborview", "region": "North", "size": "metro", "base_kwh": 430.0},
    {"id": "PO-002", "name": "Northgate", "region": "North", "size": "large", "base_kwh": 270.0},
    {"id": "PO-003", "name": "Elm Street", "region": "North", "size": "small", "base_kwh": 150.0},
    {"id": "PO-004", "name": "Cedar Park", "region": "South", "size": "metro", "base_kwh": 400.0},
    {"id": "PO-005", "name": "Brookside", "region": "South", "size": "large", "base_kwh": 250.0},
    {"id": "PO-006", "name": "Sunset", "region": "South", "size": "small", "base_kwh": 140.0},
    {"id": "PO-007", "name": "Old Town", "region": "East", "size": "metro", "base_kwh": 380.0},
    {"id": "PO-008", "name": "Maple Hill", "region": "East", "size": "large", "base_kwh": 240.0},
    {"id": "PO-009", "name": "Lakeside", "region": "East", "size": "small", "base_kwh": 135.0},
    {"id": "PO-010", "name": "Westfield", "region": "West", "size": "metro", "base_kwh": 360.0},
    {"id": "PO-011", "name": "Pinecrest", "region": "West", "size": "large", "base_kwh": 225.0},
    {"id": "PO-012", "name": "Riverbend", "region": "West", "size": "small", "base_kwh": 130.0},
]

# Annual seasonality per region. ``peak_day`` is the day of year with the
# highest consumption: winter heating in the North/East, summer cooling in
# the South/West.
REGION_SEASONALITY: dict[str, dict[str, float]] = {
    "North": {"amplitude": 0.22, "peak_day": 15},
    "South": {"amplitude": 0.20, "peak_day": 202},
    "East": {"amplitude": 0.14, "peak_day": 25},
    "West": {"amplitude": 0.10, "peak_day": 213},
}

# Weekday multipliers, Monday first. Post offices are busiest mid-week and
# nearly closed on Sundays.
WEEKDAY_FACTOR: tuple[float, ...] = (1.02, 1.04, 1.04, 1.03, 1.01, 0.86, 0.62)

# Public holidays close the counters, so the building draws much less power.
HOLIDAY_FACTOR = 0.55


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    """Return the n-th ``weekday`` (Monday=0) of a month."""
    days = [d for d in calendar.Calendar().itermonthdates(year, month) if d.month == month]
    matches = [d for d in days if d.weekday() == weekday]
    return matches[n - 1]


def _last_weekday(year: int, month: int, weekday: int) -> date:
    days = [d for d in calendar.Calendar().itermonthdates(year, month) if d.month == month]
    matches = [d for d in days if d.weekday() == weekday]
    return matches[-1]


def public_holidays(years: list[int] | range) -> pd.DataFrame:
    """Small US-style holiday table in Prophet's expected format."""
    rows: list[tuple[str, date]] = []
    for year in years:
        rows.extend(
            [
                ("new_years_day", date(year, 1, 1)),
                ("memorial_day", _last_weekday(year, 5, 0)),
                ("independence_day", date(year, 7, 4)),
                ("labor_day", _nth_weekday(year, 9, 0, 1)),
                ("thanksgiving", _nth_weekday(year, 11, 3, 4)),
                ("christmas_day", date(year, 12, 25)),
                ("day_after_christmas", date(year, 12, 26)),
            ]
        )
    return pd.DataFrame(
        {
            "holiday": [name for name, _ in rows],
            "ds": pd.to_datetime([day for _, day in rows]),
            "lower_window": 0,
            "upper_window": 0,
        }
    )
