"""Generate a toy dataset of daily electricity consumption for post offices.

The series mixes the patterns a real portfolio would show -- annual
heating/cooling seasonality that differs by region, a weekly occupancy
cycle, a small growth trend, holiday closures and day-to-day noise -- and
then injects a handful of overconsumption episodes (faulty HVAC, lighting
left on, temporary equipment, ...). The injected episodes are written to a
separate ground-truth file so the detector can be scored honestly.

Usage:
    python generate_data.py
    python generate_data.py --out-dir data --seed 7
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from postoffice_common import (
    HOLIDAY_FACTOR,
    POST_OFFICES,
    REGION_SEASONALITY,
    WEEKDAY_FACTOR,
    public_holidays,
)

BASE_DIR = Path(__file__).resolve().parent

# Fault types and the range of extra load each one produces, as a fraction
# of the expected consumption.
CAUSES: dict[str, tuple[float, float]] = {
    "HVAC stuck on": (0.35, 0.65),
    "heating element fault": (0.30, 0.60),
    "faulty chiller": (0.30, 0.55),
    "temporary sorting equipment": (0.25, 0.45),
    "after-hours lighting left on": (0.22, 0.40),
}

MIN_EVENT_GAP_DAYS = 45
EVENT_EDGE_DAYS = 14


def build_clean_series(
    office: dict[str, object],
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    growth_per_year: float,
) -> np.ndarray:
    """Return the noise-level daily kWh without any injected overconsumption."""
    region = str(office["region"])
    day_of_year = dates.dayofyear.to_numpy()
    seasonal = 1.0 + REGION_SEASONALITY[region]["amplitude"] * np.cos(
        2.0 * np.pi * (day_of_year - REGION_SEASONALITY[region]["peak_day"]) / 365.25
    )

    weekly = np.asarray(WEEKDAY_FACTOR)[dates.dayofweek.to_numpy()]

    growth = 1.0 + growth_per_year * np.arange(len(dates)) / 365.25

    # Slow, office-specific drift (refurbishments, new sorting machines, ...).
    drift = np.exp(np.cumsum(rng.normal(0.0, 0.0025, len(dates))))

    holiday_dates = set(public_holidays(sorted(dates.year.unique()))["ds"].dt.date)
    holiday = np.where(np.isin(dates.date, list(holiday_dates)), HOLIDAY_FACTOR, 1.0)

    noise = np.clip(rng.normal(1.0, 0.045, len(dates)), 0.80, 1.20)

    return float(office["base_kwh"]) * seasonal * weekly * growth * drift * holiday * noise


def add_overconsumption(
    office: dict[str, object],
    dates: pd.DatetimeIndex,
    clean: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """Multiply a few random, non-overlapping windows by an extra-load factor."""
    kwh = clean.copy()
    events: list[dict[str, object]] = []
    taken: list[tuple[pd.Timestamp, pd.Timestamp]] = []

    target_events = int(rng.integers(1, 4))  # 1-3 episodes per office
    attempts = 0
    while len(events) < target_events and attempts < 400:
        attempts += 1
        duration = int(rng.integers(3, 13))  # 3-12 days
        first = EVENT_EDGE_DAYS + int(
            rng.integers(0, len(dates) - duration - 2 * EVENT_EDGE_DAYS)
        )
        start = dates[first]
        end = dates[first + duration - 1]

        # Keep episodes away from the winter holiday shutdown, where a dip in
        # consumption is normal, and away from each other.
        if (start.month == 12 and start.day >= 15) or (end.month == 1 and end.day <= 10):
            continue
        if any(
            abs((start - other_start).days) < MIN_EVENT_GAP_DAYS
            or abs((start - other_end).days) < MIN_EVENT_GAP_DAYS
            for other_start, other_end in taken
        ):
            continue

        cause = str(rng.choice(list(CAUSES)))
        low, high = CAUSES[cause]
        magnitude = float(rng.uniform(low, high))
        ramp = np.minimum(1.0, (np.arange(duration) + 1) / 2.0)  # soften the first day
        multiplier = 1.0 + magnitude * ramp

        window = slice(first, first + duration)
        kwh[window] = clean[window] * multiplier
        extra_kwh = float(np.sum(clean[window] * (multiplier - 1.0)))

        events.append(
            {
                "post_office_id": office["id"],
                "post_office_name": office["name"],
                "region": office["region"],
                "cause": cause,
                "start_date": start,
                "end_date": end,
                "days": duration,
                "excess_pct": magnitude,
                "extra_kwh": extra_kwh,
            }
        )
        taken.append((start, end))

    return kwh, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--start", default="2023-01-01")
    parser.add_argument("--end", default="2024-12-31")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--growth",
        type=float,
        default=0.03,
        help="annual network growth rate (fraction of base consumption)",
    )
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dates = pd.date_range(args.start, args.end, freq="D")
    if len(dates) < 120:
        raise SystemExit("Date range is too short; use at least 120 days.")

    frames: list[pd.DataFrame] = []
    all_events: list[dict[str, object]] = []

    for office in POST_OFFICES:
        clean = build_clean_series(office, dates, rng, args.growth)
        kwh, events = add_overconsumption(office, dates, clean, rng)
        frames.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "post_office_id": office["id"],
                    "post_office_name": office["name"],
                    "region": office["region"],
                    "size_class": office["size"],
                    "kwh": np.round(kwh, 1),
                }
            )
        )
        all_events.extend(events)

    readings = pd.concat(frames, ignore_index=True).sort_values(
        ["post_office_id", "date"], ignore_index=True
    )
    events = pd.DataFrame(all_events).sort_values(
        ["post_office_id", "start_date"], ignore_index=True
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    readings_path = args.out_dir / "postoffice_electricity.csv"
    events_path = args.out_dir / "overconsumption_events.csv"

    readings.to_csv(readings_path, index=False, date_format="%Y-%m-%d")
    events.to_csv(
        events_path,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.3f",
    )

    print(f"wrote {len(readings):,} readings ({len(dates)} days x {len(POST_OFFICES)} offices)")
    print(f"  {readings_path}")
    print(
        f"wrote {len(events)} overconsumption episodes, "
        f"{int(events['days'].sum())} anomalous days "
        f"(~{events['extra_kwh'].sum():,.0f} kWh wasted)"
    )
    print(f"  {events_path}")


if __name__ == "__main__":
    main()
