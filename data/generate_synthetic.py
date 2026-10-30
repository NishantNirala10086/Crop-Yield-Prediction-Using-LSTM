"""Generate a SYNTHETIC demo dataset (data/crop_yield.csv).

It only exists so the whole pipeline can run end-to-end without downloading data.
Replace data/crop_yield.csv with real district-level records (see README) before
drawing any conclusion: results on synthetic data say nothing about real farms.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

DISTRICTS = {  # name: (rain factor, temperature offset in deg C)
    "Ludhiana": (1.00, 0.0), "Amritsar": (0.95, 0.2), "Patiala": (1.05, -0.1),
    "Bathinda": (0.70, 0.8), "Jalandhar": (1.00, 0.1), "Karnal": (1.10, 0.0),
    "Ambala": (1.15, -0.2), "Hisar": (0.65, 1.0),
}
CROPS = {  # base yield t/ha, annual technology gain, base fertiliser kg/ha
    "wheat": (4.2, 0.012, 150), "rice": (3.8, 0.010, 120), "maize": (3.0, 0.015, 110),
}
RAIN = np.array([20, 25, 20, 15, 20, 80, 200, 220, 110, 20, 5, 10], float)
TMAX = np.array([18, 22, 28, 36, 40, 39, 34, 33, 33, 32, 26, 20], float)
TMIN = np.array([6, 9, 14, 20, 25, 27, 26, 26, 24, 17, 10, 6], float)
HUM = np.array([75, 68, 55, 40, 40, 55, 78, 82, 72, 60, 65, 75], float)


def weather_year(rng, rain_f, temp_off, year):
    wet = rng.lognormal(0, 0.22)                      # year-wide rainfall anomaly
    warm = rng.normal(0, 0.9) + 0.03 * (year - 2000)  # year-wide temperature anomaly
    rain = np.clip(RAIN * rain_f * wet * rng.lognormal(0, 0.35, 12), 0, None)
    tmax = TMAX + temp_off + warm + rng.normal(0, 1.0, 12)
    tmin = TMIN + temp_off + warm + rng.normal(0, 0.8, 12)
    hum = np.clip(HUM + 6 * (wet - 1) + rng.normal(0, 4, 12), 15, 100)
    return rain, tmax, tmin, hum


def weather_factor(crop, rain, tmax):
    if crop == "wheat":      # terminal heat stress (Feb-Mar), winter rain helps
        heat = max(0.0, tmax[1:3].mean() - 25)
        return 1 - 0.035 * heat + 0.0006 * min(rain[[11, 0, 1]].sum(), 120)
    if crop == "rice":       # monsoon rainfall helps (saturating), heat hurts
        return 1 + 0.25 * np.tanh((rain[6:9].sum() - 450) / 300) - 0.02 * max(0, tmax[5:9].mean() - 36)
    return 1 + 0.20 * np.tanh((rain[5:9].sum() - 500) / 300) - 0.03 * max(0, tmax[5] - 40)


def generate(seed=config.SEED, y0=2000, y1=2022):
    rng = np.random.default_rng(seed)
    rows = []
    for d, (rf, to) in DISTRICTS.items():
        for crop, (base, gain, fert0) in CROPS.items():
            fert_d = fert0 * rng.uniform(0.85, 1.15)
            for year in range(y0, y1 + 1):
                rain, tmax, tmin, hum = weather_year(rng, rf, to, year)
                fert = fert_d * (1 + 0.012 * (year - y0)) * rng.uniform(0.95, 1.05)
                irr = min(99, 70 + 1.0 * (year - y0) + 5 * rf + rng.normal(0, 2))
                f_w = weather_factor(crop, rain, tmax)
                f_f = 1 + 0.18 * np.tanh((fert - fert0) / 60)
                f_i = 0.88 + 0.12 * irr / 100
                y = base * (1 + gain * (year - y0)) * f_w * f_f * f_i * rng.normal(1, 0.04)
                y = float(np.clip(y, 0.5, None))
                for m in range(12):
                    rows.append((d, crop, year, m + 1, rain[m], tmax[m], tmin[m], hum[m], fert, irr, y))
    cols = ["district", "crop", "year", "month", *config.WEATHER_COLS,
            "fertiliser", "irrigated_pct", config.TARGET]
    return pd.DataFrame(rows, columns=cols).round(3)


if __name__ == "__main__":
    df = generate()
    config.DATA_CSV.parent.mkdir(exist_ok=True)
    df.to_csv(config.DATA_CSV, index=False)
    print(f"Wrote {len(df):,} rows ({df.groupby(['district','crop','year']).ngroups} samples) to {config.DATA_CSV}")
