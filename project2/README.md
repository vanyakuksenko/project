# Post-office electricity overconsumption (toy dataset + Prophet)

A small, self-contained demo of the kind of analysis an energy manager might
run on a portfolio of buildings: generate daily electricity readings for a
dozen post offices, then flag days where a site uses far more power than its
normal pattern predicts.

Everything is synthetic, seeded and local. Prophet is the only modelling
dependency, matplotlib is used for the charts.

## Quick start

```bash
python -m pip install -r requirements.txt

python generate_data.py          # writes data/*.csv (~1 s)
python detect_overconsumption.py # fits 12 Prophet models (~30 s), writes outputs/
python -m pytest tests -q        # unit + end-to-end checks (~3 s)
```

Re-running the detector gives the same episodes: Prophet's uncertainty
sampling is seeded inside `fit_office`.

## What's generated

`generate_data.py` writes two files:

- **`data/postoffice_electricity.csv`** – 8,772 rows:
  `date, post_office_id, post_office_name, region, size_class, kwh`
  (12 offices × 731 days, 2023-01-01 → 2024-12-31).
- **`data/overconsumption_events.csv`** – the injected ground truth:
  `post_office_id, ..., cause, start_date, end_date, days, excess_pct, extra_kwh`.

The readings mix several patterns on purpose:

| Pattern | How it is modelled |
| --- | --- |
| Region-specific seasonality | Winter heating peak in the North/East, summer cooling peak in the South/West |
| Weekly occupancy | Busy mid-week, reduced Saturday, near-closed Sunday |
| Holiday closures | Seven US-style holidays at 55 % of normal load |
| Growth and drift | 3 %/year network growth plus a slow per-office random walk |
| Meter noise | Multiplicative Gaussian noise, σ ≈ 4.5 % |
| Overconsumption | 22 episodes of 3–12 days, +22 % to +65 % load, with a one-day ramp |

The episodes cover five plausible causes (HVAC stuck on, heating element
fault, faulty chiller, temporary sorting equipment, after-hours lighting),
so the detector output reads like a maintenance backlog.

## How the detector works

For each office independently:

1. Fit `Prophet` with `seasonality_mode="multiplicative"` — consumption scales
   with building size, so a 20 % seasonal swing is 20 % everywhere.
   Weekly seasonality uses Fourier order 5 to capture the sharp Sunday dip;
   holidays come from `postoffice_common.public_holidays()`.
2. Predict the same days **in sample** and keep `yhat_upper` for a 99 %
   interval.
3. Flag a day only when **both** hold:
   - `kwh > yhat_upper`, and
   - `kwh / yhat - 1 >= 15 %` (`--min-excess`).

   The second rule stops statistically "surprising" but operationally
   irrelevant blips on small offices.
4. Merge flagged days separated by up to two unflagged days into an
   **episode** (`--merge-gap-days`), then sum the energy above `yhat`.
5. If the ground-truth file exists, score days and episodes against it.

## Results with the committed seed

```
predicted episodes : 23
events found       : 21 / 22
day-level          precision 0.981  recall 0.739  F1 0.843
event-level        precision 0.913  recall 0.955  F1 0.933
```

The one missed event is a six-day, +26 % lighting fault at PO-009 whose
readings stayed inside Prophet's interval. Two single-day flags are false
positives (PO-001 at +15.1 %, PO-007 at +16.9 %); passing
`--min-episode-days 2` removes them, at the cost of the one-day PO-006
episode. Day-level recall is lower than event-level recall because the first
day of an episode is often masked by a weekend dip or the injected ramp.

The kWh totals in `episodes.csv` are deliberately conservative: Prophet's
expectation is fitted on data that already contains the anomaly, and only
flagged days are summed. They are useful for ranking sites, not for billing.

## Outputs

| Path | Content |
| --- | --- |
| `outputs/predictions.csv` | Daily actual, forecast, interval and flag for every office |
| `outputs/detections.csv` | Flagged days only, with relative excess |
| `outputs/episodes.csv` | One row per alert: dates, duration, peak/mean excess, excess kWh |
| `outputs/evaluation.txt` | Precision/recall plus a per-event detected/missed table |
| `outputs/plots/<office>.png` | Actual vs expected, interval, flags, injected events |
| `outputs/plots/summary.png` | Offices ranked by energy attributed to episodes |

## Tuning knobs

```bash
python detect_overconsumption.py --interval-width 0.995 --min-excess 0.2
python detect_overconsumption.py --min-episode-days 2
python detect_overconsumption.py --no-plots --out-dir runs/strict
```

- Looser interval / lower `--min-excess` → more recall, more false alarms.
- Higher `--min-excess` → fewer, stronger alerts.
- `--merge-gap-days` controls how much a mid-episode dip can split one alert.
- `--min-episode-days 2` suppresses single-day blips.

## Caveats

This is a teaching dataset, not a production anomaly detector. Notable gaps:

- Detection is **retrospective and in-sample**. A live system should fit on
  clean history and forecast forward, and/or use rolling-origin
  cross-validation.
- Sustained anomalies contaminate the fit; robust regression or iterative
  refitting on clean days reduces that.
- Real consumption is strongly weather-driven. Adding heating/cooling degree
  days (or a temperature regressor) would remove a major confounder.
- Testing 12 offices × 731 days means some false positives are expected;
  control the family-wise error rate or require persistence.
- Daily totals hide intraday equipment faults; half-hourly meter data would
  support a much sharper analysis.

## Files

```
postoffice_common.py       offices, seasonality, holiday calendar
generate_data.py           synthetic readings + injected episodes
detect_overconsumption.py  Prophet fitting, flagging, scoring, plots
tests/test_detection.py    episode grouping, scoring, end-to-end spike test
```
