"""Spot overconsumption in post-office electricity readings with Prophet.

For every post office the script fits a Prophet model with multiplicative
seasonality (daily consumption behaves multiplicatively: a winter peak or a
weekend dip scales with the size of the building), then compares each day's
actual reading with Prophet's predictive interval.

A day is flagged when both conditions hold:
  1. actual kWh is above Prophet's upper interval bound (``--interval-width``), and
  2. the reading is at least ``--min-excess`` above the point forecast.

Flagged days that are close together are merged into episodes, which makes
the output readable ("HVAC ran for six days") instead of a wall of daily
alerts. If the ground-truth file produced by ``generate_data.py`` is present
the episodes are scored against it.

Usage:
    python detect_overconsumption.py
    python detect_overconsumption.py --interval-width 0.995 --min-excess 0.2
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless: write PNGs, never open a window

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from postoffice_common import public_holidays

BASE_DIR = Path(__file__).resolve().parent

# cmdstanpy installs its own INFO handler the first time a model is fitted, and
# overrides any level set before that. Registering a null handler first makes
# sure our quieter level sticks. This must happen before importing Prophet so
# its plotly notice is silenced too.
logging.getLogger("cmdstanpy").addHandler(logging.NullHandler())
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
logging.getLogger("prophet").setLevel(logging.ERROR)
# Prophet logs an ERROR at import time when plotly is absent; this demo writes
# static matplotlib PNGs, so the notice is pure noise.
logging.getLogger("prophet.plot").setLevel(logging.CRITICAL)

from prophet import Prophet  # noqa: E402  (level setup must come first)


def load_readings(path: Path) -> pd.DataFrame:
    """Load the long-format readings and normalise the column names for Prophet."""
    readings = pd.read_csv(path, parse_dates=["date"])
    missing = {"date", "post_office_id", "kwh"} - set(readings.columns)
    if missing:
        raise SystemExit(f"{path} is missing required columns: {', '.join(sorted(missing))}")

    if "post_office_name" not in readings.columns:
        readings["post_office_name"] = readings["post_office_id"]

    readings = readings.rename(columns={"date": "ds", "kwh": "y"})
    return readings.sort_values(["post_office_id", "ds"], ignore_index=True)


def fit_office(
    series: pd.DataFrame,
    holidays: pd.DataFrame,
    interval_width: float = 0.99,
    min_excess: float = 0.15,
) -> pd.DataFrame:
    """Fit Prophet on one office and return daily actual/expected/flag columns."""
    series = series.sort_values("ds").reset_index(drop=True)
    if len(series) < 60:
        raise ValueError("Need at least 60 daily readings to fit the model")

    # Prophet builds its uncertainty interval from numpy's global RNG, so seed
    # it to keep the reported episodes reproducible between runs.
    np.random.seed(0)

    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=False,
        daily_seasonality=False,
        seasonality_mode="multiplicative",
        interval_width=interval_width,
        changepoint_prior_scale=0.05,
        holidays=holidays,
    )
    # Data has a sharp weekend/Sunday cycle; Fourier order 5 tracks it better
    # than Prophet's weekly default of 3.
    model.add_seasonality(name="weekly", period=7, fourier_order=5)
    model.fit(series[["ds", "y"]])

    forecast = model.predict(series[["ds"]])
    predictions = series[["ds", "y"]].copy()
    predictions["yhat"] = forecast["yhat"].to_numpy()
    predictions["yhat_lower"] = forecast["yhat_lower"].to_numpy()
    predictions["yhat_upper"] = forecast["yhat_upper"].to_numpy()

    # Multiplicative view of the same quantity: how far above expectation is
    # the meter, as a percentage. Guard against a non-positive point forecast.
    predictions["excess_pct"] = np.where(
        predictions["yhat"] > 0.0, predictions["y"] / predictions["yhat"] - 1.0, np.nan
    )
    predictions["flagged"] = (
        predictions["y"] > predictions["yhat_upper"]
    ) & (predictions["excess_pct"] >= min_excess)
    return predictions


def find_episodes(
    predictions: pd.DataFrame,
    merge_gap_days: int = 2,
    min_days: int = 1,
) -> pd.DataFrame:
    """Group flagged days into episodes, allowing short unflagged gaps inside one."""
    columns = [
        "episode_id",
        "post_office_id",
        "post_office_name",
        "start",
        "end",
        "detected_days",
        "calendar_days",
        "max_excess_pct",
        "mean_excess_pct",
        "excess_kwh",
        "peak_kwh",
    ]
    flagged_positions = np.flatnonzero(predictions["flagged"].to_numpy())
    if len(flagged_positions) == 0:
        return pd.DataFrame(columns=columns)

    groups: list[list[int]] = [[int(flagged_positions[0])]]
    for position in flagged_positions[1:]:
        if position - groups[-1][-1] > merge_gap_days + 1:
            groups.append([int(position)])
        else:
            groups[-1].append(int(position))

    groups = [group for group in groups if len(group) >= min_days]
    office_id = str(predictions["post_office_id"].iloc[0])
    office_name = str(predictions["post_office_name"].iloc[0])

    rows = []
    for number, positions in enumerate(groups, start=1):
        block = predictions.iloc[positions]
        rows.append(
            {
                "episode_id": f"{office_id}-E{number:02d}",
                "post_office_id": office_id,
                "post_office_name": office_name,
                "start": block["ds"].iloc[0],
                "end": block["ds"].iloc[-1],
                "detected_days": len(block),
                "calendar_days": (block["ds"].iloc[-1] - block["ds"].iloc[0]).days + 1,
                "max_excess_pct": float(block["excess_pct"].max()),
                "mean_excess_pct": float(block["excess_pct"].mean()),
                "excess_kwh": float((block["y"] - block["yhat"]).sum()),
                "peak_kwh": float(block["y"].max()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def evaluate(
    predictions: pd.DataFrame,
    episodes: pd.DataFrame,
    events: pd.DataFrame,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Day-level and episode-level scores against the injected ground truth."""
    truth_days: set[tuple[str, pd.Timestamp]] = set()
    for event in events.itertuples(index=False):
        for day in pd.date_range(event.start_date, event.end_date, freq="D"):
            truth_days.add((str(event.post_office_id), day))

    predicted_days = {
        (str(row.post_office_id), row.ds.normalize())
        for row in predictions.loc[predictions["flagged"]].itertuples(index=False)
    }

    true_positive = len(predicted_days & truth_days)
    day_precision = _safe_div(true_positive, len(predicted_days))
    day_recall = _safe_div(true_positive, len(truth_days))

    # An event counts as found when a reported episode overlaps its window; a
    # reported episode counts as a hit when it overlaps any injected event.
    episode_spans: dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]] = {}
    for episode in episodes.itertuples(index=False):
        episode_spans.setdefault(str(episode.post_office_id), []).append(
            (pd.Timestamp(episode.start), pd.Timestamp(episode.end))
        )

    matched_events = 0
    event_checks = []
    for event in events.itertuples(index=False):
        found = any(
            start <= event.end_date and event.start_date <= end
            for start, end in episode_spans.get(str(event.post_office_id), [])
        )
        matched_events += int(found)
        event_checks.append(
            {
                "post_office_id": event.post_office_id,
                "cause": event.cause,
                "start_date": event.start_date,
                "end_date": event.end_date,
                "days": event.days,
                "excess_pct": event.excess_pct,
                "detected": bool(found),
            }
        )

    matched_episodes = 0
    for episode in episodes.itertuples(index=False):
        episode_days = pd.date_range(episode.start, episode.end, freq="D")
        if any((str(episode.post_office_id), day) in truth_days for day in episode_days):
            matched_episodes += 1

    metrics = {
        "true_event_count": float(len(events)),
        "predicted_episode_count": float(len(episodes)),
        "matched_events": float(matched_events),
        "matched_episodes": float(matched_episodes),
        "day_precision": day_precision,
        "day_recall": day_recall,
        "day_f1": _f1(day_precision, day_recall),
        "event_precision": _safe_div(matched_episodes, len(episodes)),
        "event_recall": _safe_div(matched_events, len(events)),
        "event_f1": _f1(
            _safe_div(matched_episodes, len(episodes)),
            _safe_div(matched_events, len(events)),
        ),
    }
    return metrics, pd.DataFrame(event_checks)


def _safe_div(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0


def _f1(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def plot_office(
    predictions: pd.DataFrame,
    episodes: pd.DataFrame,
    events: pd.DataFrame | None,
    path: Path,
) -> None:
    """Expected vs actual series for one office, with flags and truth spans."""
    figure, axis = plt.subplots(figsize=(13, 4.6))
    axis.fill_between(
        predictions["ds"],
        predictions["yhat_lower"],
        predictions["yhat_upper"],
        color="tab:blue",
        alpha=0.16,
        linewidth=0,
        label="Prophet interval",
    )
    axis.plot(
        predictions["ds"],
        predictions["yhat"],
        color="tab:blue",
        linewidth=1.1,
        label="expected (Prophet)",
    )
    axis.plot(
        predictions["ds"],
        predictions["y"],
        color="0.25",
        linewidth=0.7,
        alpha=0.85,
        label="actual",
    )

    flagged = predictions.loc[predictions["flagged"]]
    axis.scatter(
        flagged["ds"],
        flagged["y"],
        color="tab:red",
        s=18,
        zorder=5,
        label="flagged overconsumption",
    )

    if events is not None and not events.empty:
        for number, event in enumerate(events.itertuples(index=False)):
            axis.axvspan(
                event.start_date,
                event.end_date + pd.Timedelta(days=1),
                color="tab:orange",
                alpha=0.10,
                label="injected event" if number == 0 else None,
            )

    office_id = predictions["post_office_id"].iloc[0]
    office_name = predictions["post_office_name"].iloc[0]
    subtitle = f"{len(episodes)} episode(s) flagged" if len(episodes) else "no episodes flagged"
    axis.set_title(f"{office_id} {office_name} - daily electricity use ({subtitle})", loc="left")
    axis.set_ylabel("kWh / day")
    axis.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    axis.margins(x=0.01)
    axis.grid(axis="y", color="0.9")
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(loc="upper left", fontsize=8, frameon=False, ncols=2)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def plot_summary(episodes: pd.DataFrame, path: Path) -> None:
    """Rank offices by the energy attributed to flagged episodes."""
    totals = (
        episodes.groupby(["post_office_id", "post_office_name"], as_index=False)
        .agg(excess_kwh=("excess_kwh", "sum"), episodes=("episode_id", "count"))
        .sort_values("excess_kwh")
    )
    labels = totals["post_office_id"] + "  " + totals["post_office_name"]

    figure, axis = plt.subplots(figsize=(8.5, 0.45 * len(totals) + 1.8))
    bars = axis.barh(labels, totals["excess_kwh"], color="tab:red", alpha=0.75)
    axis.bar_label(
        bars,
        labels=[f"{value:,.0f} kWh / {count} ep." for value, count in zip(totals["excess_kwh"], totals["episodes"])],
        padding=4,
        fontsize=8,
    )
    axis.set_xlabel("excess energy above Prophet's point forecast (kWh)")
    axis.set_title("Overconsumption flagged in the post-office network", loc="left")
    axis.set_xlim(0, totals["excess_kwh"].max() * 1.35)
    axis.spines[["top", "right"]].set_visible(False)
    figure.tight_layout()
    figure.savefig(path)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Spot overconsumption in post-office electricity use with Prophet."
    )
    parser.add_argument("--data", type=Path, default=BASE_DIR / "data" / "postoffice_electricity.csv")
    parser.add_argument(
        "--events",
        type=Path,
        default=BASE_DIR / "data" / "overconsumption_events.csv",
        help="optional ground-truth file used for scoring",
    )
    parser.add_argument("--out-dir", type=Path, default=BASE_DIR / "outputs")
    parser.add_argument(
        "--interval-width",
        type=float,
        default=0.99,
        help="Prophet predictive interval used as the upper limit (default 0.99)",
    )
    parser.add_argument(
        "--min-excess",
        type=float,
        default=0.15,
        help="minimum relative excess over the point forecast (default 0.15)",
    )
    parser.add_argument("--merge-gap-days", type=int, default=2)
    parser.add_argument("--min-episode-days", type=int, default=1)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()

    if not args.data.exists():
        raise SystemExit(
            f"Readings not found: {args.data}\nRun `python generate_data.py` first."
        )

    readings = load_readings(args.data)
    holidays = public_holidays(sorted(readings["ds"].dt.year.unique().tolist()))
    plot_dir = args.out_dir / "plots"
    if not args.no_plots:
        plot_dir.mkdir(parents=True, exist_ok=True)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    predictions_frames: list[pd.DataFrame] = []
    episode_frames: list[pd.DataFrame] = []

    for office_id, series in readings.groupby("post_office_id", sort=True):
        predictions = fit_office(
            series,
            holidays,
            interval_width=args.interval_width,
            min_excess=args.min_excess,
        )
        predictions["post_office_id"] = office_id
        predictions["post_office_name"] = series["post_office_name"].iloc[0]

        episodes = find_episodes(
            predictions,
            merge_gap_days=args.merge_gap_days,
            min_days=args.min_episode_days,
        )

        print(
            f"{office_id} {series['post_office_name'].iloc[0]:<11} "
            f"{len(series):>4} days | {int(predictions['flagged'].sum()):>3} flagged days "
            f"| {len(episodes)} episode(s)"
        )

        predictions_frames.append(predictions)
        episode_frames.append(episodes)

        if not args.no_plots and not episodes.empty:
            office_events = None
            if args.events and args.events.exists():
                events = pd.read_csv(args.events, parse_dates=["start_date", "end_date"])
                office_events = events.loc[events["post_office_id"] == office_id].reset_index(drop=True)
            plot_office(predictions, episodes, office_events, plot_dir / f"{office_id}.png")

    all_predictions = pd.concat(predictions_frames, ignore_index=True)
    all_episodes = (
        pd.concat(episode_frames, ignore_index=True)
        if any(not frame.empty for frame in episode_frames)
        else pd.DataFrame()
    )

    all_predictions.to_csv(args.out_dir / "predictions.csv", index=False, date_format="%Y-%m-%d")
    detections = all_predictions.loc[
        all_predictions["flagged"],
        [
            "ds",
            "post_office_id",
            "post_office_name",
            "y",
            "yhat",
            "yhat_upper",
            "excess_pct",
        ],
    ].rename(columns={"ds": "date", "y": "kwh", "yhat": "expected_kwh"})
    detections.to_csv(
        args.out_dir / "detections.csv", index=False, date_format="%Y-%m-%d", float_format="%.3f"
    )

    if not all_episodes.empty:
        all_episodes.to_csv(
            args.out_dir / "episodes.csv", index=False, date_format="%Y-%m-%d", float_format="%.3f"
        )
        if not args.no_plots:
            plot_summary(all_episodes, plot_dir / "summary.png")

        print("\nFlagged episodes")
        for episode in all_episodes.itertuples(index=False):
            print(
                f"  {episode.episode_id}  {episode.start:%Y-%m-%d} -> {episode.end:%Y-%m-%d}  "
                f"{episode.calendar_days:>2}d  peak +{episode.max_excess_pct:5.1%}  "
                f"{episode.excess_kwh:8.1f} kWh above expectation"
            )
        print(
            f"\nTotal: {len(all_episodes)} episodes, "
            f"{all_episodes['excess_kwh'].sum():,.0f} kWh attributed to overconsumption"
        )
    else:
        print("\nNo overconsumption episodes flagged.")

    if args.events and args.events.exists():
        events = pd.read_csv(args.events, parse_dates=["start_date", "end_date"])
        metrics, event_checks = evaluate(all_predictions, all_episodes, events)
        lines = [
            "Ground-truth comparison (injected episodes)",
            "",
            f"injected events            : {int(metrics['true_event_count'])}",
            f"injected anomalous days    : {int(events['days'].sum())}",
            f"predicted episodes         : {int(metrics['predicted_episode_count'])}",
            f"events found (episode overlap): {int(metrics['matched_events'])} "
            f"/ {int(metrics['true_event_count'])}",
            f"episodes matching an event : {int(metrics['matched_episodes'])} "
            f"/ {int(metrics['predicted_episode_count'])}",
            "",
            f"day-level precision        : {metrics['day_precision']:.3f}",
            f"day-level recall           : {metrics['day_recall']:.3f}",
            f"day-level F1               : {metrics['day_f1']:.3f}",
            f"event-level precision      : {metrics['event_precision']:.3f}",
            f"event-level recall         : {metrics['event_recall']:.3f}",
            f"event-level F1             : {metrics['event_f1']:.3f}",
            "",
            event_checks.to_string(index=False),
        ]
        report = "\n".join(lines)
        print("\n" + report)
        (args.out_dir / "evaluation.txt").write_text(report + "\n", encoding="utf-8")
    else:
        print("\nNo ground-truth file found; skipping scoring.")

    print(f"\nOutputs written to {args.out_dir}")


if __name__ == "__main__":
    main()
