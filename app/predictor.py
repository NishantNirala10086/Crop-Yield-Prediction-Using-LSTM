"""Inference logic (kept separate from Flask so it can be unit-tested)."""
import json
import sys
from pathlib import Path

import joblib
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import preprocess  # noqa: E402


class InputError(ValueError):
    """Raised for invalid user input (returned to the client as HTTP 400)."""


class Predictor:
    def __init__(self, model=None, df=None, xs=None, ys=None, meta=None, rmse=None):
        self.df = preprocess.load_data() if df is None else df
        self.meta = meta or json.loads((config.MODEL_DIR / "meta.json").read_text())
        self.xs = xs or joblib.load(config.MODEL_DIR / "x_scaler.joblib")
        self.ys = ys or joblib.load(config.MODEL_DIR / "y_scaler.joblib")
        if model is None:
            import tensorflow as tf
            model = tf.keras.models.load_model(config.MODEL_DIR / "lstm_yield.keras")
        self.model = model
        if rmse is None:
            p = config.MODEL_DIR / "test_metrics.json"
            rmse = json.loads(p.read_text())["RMSE"] if p.exists() else None
        self.rmse = rmse
        self.seasons = preprocess.season_table(self.df)

    # ---- metadata for the UI -------------------------------------------------
    def options(self):
        st = self.seasons.dropna(subset=["prev_yield"])
        years = sorted(int(y) for y in st.year.unique())
        return {"districts": self.meta["districts"], "crops": self.meta["crops"],
                "years": years, "default_year": years[-1], "test_rmse": self.rmse}

    def history(self, district, crop):
        s = self.seasons[(self.seasons.district == district) & (self.seasons.crop == crop)]
        return [{"year": int(r.year), "yield": round(float(r.yield_), 3)} for r in s.itertuples()]

    # ---- prediction ------------------------------------------------------------
    def predict(self, district, crop, year, rain_change_pct=0.0, temp_change_c=0.0):
        if district not in self.meta["districts"]:
            raise InputError(f"Unknown district: {district!r}")
        if crop not in self.meta["crops"]:
            raise InputError(f"Unknown crop: {crop!r}")
        try:
            year = int(year); rain_change_pct = float(rain_change_pct); temp_change_c = float(temp_change_c)
        except (TypeError, ValueError):
            raise InputError("year, rain_change_pct and temp_change_c must be numbers")
        if not -60 <= rain_change_pct <= 100:
            raise InputError("rain_change_pct must be between -60 and 100")
        if not -5 <= temp_change_c <= 5:
            raise InputError("temp_change_c must be between -5 and 5")

        rows = self.df[(self.df.district == district) & (self.df.crop == crop) & (self.df.year == year)]
        rows = rows.sort_values("month")
        if len(rows) != config.TIMESTEPS:
            raise InputError(f"No complete {config.TIMESTEPS}-month weather record for {crop} in {district}, {year}")
        s = self.seasons[(self.seasons.district == district) & (self.seasons.crop == crop)].set_index("year")
        if year - 1 not in s.index:
            raise InputError("Previous-season yield is required but not available for that year")

        w = rows[config.WEATHER_COLS].to_numpy(float).copy()
        w[:, 0] = np.clip(w[:, 0] * (1 + rain_change_pct / 100), 0, None)   # rainfall
        w[:, 1] += temp_change_c                                           # tmax
        w[:, 2] += temp_change_c                                           # tmin
        prev = float(s.loc[year - 1, "yield_"])
        x = preprocess.sample_features(w, float(rows.fertiliser.iloc[0]), float(rows.irrigated_pct.iloc[0]),
                                       prev, district, crop, self.meta)
        xs = preprocess.scale_X(self.xs, x[None, ...])
        yhat = float(self.ys.inverse_transform(np.asarray(self.model.predict(xs, verbose=0)).reshape(-1, 1))[0, 0])
        yhat = max(yhat, 0.0)

        past = s.loc[[y for y in s.index if y < year], "yield_"].tail(5)
        avg5 = float(past.mean()) if len(past) else None
        actual = float(s.loc[year, "yield_"]) if year in s.index else None
        r = self.rmse
        return {
            "district": district, "crop": crop, "year": year,
            "scenario": {"rain_change_pct": rain_change_pct, "temp_change_c": temp_change_c},
            "predicted": round(yhat, 3),
            "low": round(max(yhat - r, 0), 3) if r else None,
            "high": round(yhat + r, 3) if r else None,
            "avg5": round(avg5, 3) if avg5 is not None else None,
            "vs_avg5_pct": round((yhat / avg5 - 1) * 100, 1) if avg5 else None,
            "actual": round(actual, 3) if actual is not None else None,
            "prev_yield": round(prev, 3),
            "history": self.history(district, crop),
            "weather": [{"month": i + 1, "rainfall": round(float(a), 1), "tmax": round(float(b), 1),
                         "tmin": round(float(c), 1), "humidity": round(float(h), 1)}
                        for i, (a, b, c, h) in enumerate(w)],
        }
