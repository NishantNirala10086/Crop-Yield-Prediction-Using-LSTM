"""Data loading, sequence creation, chronological split and scaling."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402

REQUIRED = ["district", "crop", "year", "month", *config.WEATHER_COLS,
            "fertiliser", "irrigated_pct", config.TARGET]


def load_data(path=config.DATA_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    df = df.sort_values(["district", "crop", "year", "month"]).reset_index(drop=True)
    # short gaps in weather -> interpolate within each district-crop-year; then medians
    keys = ["district", "crop", "year"]
    df[config.WEATHER_COLS] = (df.groupby(keys)[config.WEATHER_COLS]
                               .transform(lambda s: s.interpolate(limit_direction="both")))
    df[config.WEATHER_COLS] = df[config.WEATHER_COLS].fillna(df[config.WEATHER_COLS].median())
    df = df[df[config.TARGET] > 0]
    return df


def season_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per district-crop-year with yield and the previous year's yield."""
    t = (df.groupby(["district", "crop", "year"])
           .agg(fertiliser=("fertiliser", "first"), irrigated_pct=("irrigated_pct", "first"),
                yield_=(config.TARGET, "first")).reset_index())
    t["prev_yield"] = t.groupby(["district", "crop"])["yield_"].shift(1)
    return t


def make_meta(df: pd.DataFrame) -> dict:
    return {"districts": sorted(df.district.unique().tolist()),
            "crops": sorted(df.crop.unique().tolist())}


def feature_names(meta: dict) -> list:
    return (config.WEATHER_COLS + config.STATIC_COLS
            + [f"crop_{c}" for c in meta["crops"]] + [f"dist_{d}" for d in meta["districts"]])


def sample_features(month_weather: np.ndarray, fert, irr, prev, district, crop, meta) -> np.ndarray:
    """Build one (T, F) matrix from Tx4 weather and season-level values."""
    T = month_weather.shape[0]
    static = np.tile([fert, irr, prev], (T, 1))
    crop_oh = np.tile([float(crop == c) for c in meta["crops"]], (T, 1))
    dist_oh = np.tile([float(district == d) for d in meta["districts"]], (T, 1))
    return np.hstack([month_weather, static, crop_oh, dist_oh])


def build_samples(df: pd.DataFrame, meta: dict):
    """Return X (n,T,F), y (n,), info DataFrame (district, crop, year)."""
    st = season_table(df).dropna(subset=["prev_yield"])  # first year of each series has no lag
    g = {k: v for k, v in df.groupby(["district", "crop", "year"])}
    X, y, info = [], [], []
    for r in st.itertuples(index=False):
        w = g[(r.district, r.crop, r.year)].sort_values("month")[config.WEATHER_COLS].to_numpy(float)
        if w.shape[0] != config.TIMESTEPS:
            continue
        X.append(sample_features(w, r.fertiliser, r.irrigated_pct, r.prev_yield, r.district, r.crop, meta))
        y.append(r.yield_)
        info.append((r.district, r.crop, r.year))
    return (np.asarray(X, np.float32), np.asarray(y, np.float32),
            pd.DataFrame(info, columns=["district", "crop", "year"]))


def chronological_split(info: pd.DataFrame, split=config.SPLIT):
    years = np.sort(info.year.unique())
    n = len(years)
    n_tr = max(1, int(round(n * split[0])))
    n_va = max(1, int(round(n * split[1])))
    tr, va = set(years[:n_tr]), set(years[n_tr:n_tr + n_va])
    idx_tr = np.where(info.year.isin(tr))[0]
    idx_va = np.where(info.year.isin(va))[0]
    idx_te = np.where(~info.year.isin(tr | va))[0]
    return idx_tr, idx_va, idx_te


def fit_scalers(X_train, y_train):
    F = X_train.shape[2]
    xs = MinMaxScaler().fit(X_train.reshape(-1, F))   # fitted on TRAIN only (no leakage)
    ys = MinMaxScaler().fit(y_train.reshape(-1, 1))
    return xs, ys


def scale_X(xs, X):
    n, T, F = X.shape
    return xs.transform(X.reshape(-1, F)).reshape(n, T, F).astype(np.float32)


def prepare(df=None, save=True):
    """Full pipeline -> dict with scaled splits, scalers, meta."""
    df = load_data() if df is None else df
    meta = make_meta(df)
    X, y, info = build_samples(df, meta)
    itr, iva, ite = chronological_split(info)
    xs, ys = fit_scalers(X[itr], y[itr])
    out = {"meta": meta, "info": info, "xs": xs, "ys": ys, "idx": (itr, iva, ite),
           "X_raw": X, "y_raw": y, "X": scale_X(xs, X),
           "y": ys.transform(y.reshape(-1, 1)).ravel().astype(np.float32)}
    meta["feature_names"] = feature_names(meta)
    meta["split_years"] = {k: sorted(int(v) for v in info.iloc[i].year.unique())
                           for k, i in zip(("train", "val", "test"), (itr, iva, ite))}
    if save:
        config.MODEL_DIR.mkdir(exist_ok=True)
        joblib.dump(xs, config.MODEL_DIR / "x_scaler.joblib")
        joblib.dump(ys, config.MODEL_DIR / "y_scaler.joblib")
        (config.MODEL_DIR / "meta.json").write_text(json.dumps(meta, indent=2))
    return out
