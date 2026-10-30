"""Tests that do not need TensorFlow (the model is replaced by a tiny fake)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from app.app import create_app  # noqa: E402
from app.predictor import InputError, Predictor  # noqa: E402
from data.generate_synthetic import generate  # noqa: E402
from src import preprocess  # noqa: E402
from src.metrics import regression_metrics  # noqa: E402


def _clean(df):
    return df.sort_values(["district", "crop", "year", "month"]).reset_index(drop=True)


@pytest.fixture(scope="module")
def prepared():
    return preprocess.prepare(_clean(generate()), save=False)


def test_shapes_and_split_is_chronological(prepared):
    X, y = prepared["X"], prepared["y"]
    assert X.shape[1] == config.TIMESTEPS and X.shape[0] == y.shape[0]
    itr, iva, ite = prepared["idx"]
    yr = prepared["info"].year.to_numpy()
    assert yr[itr].max() < yr[iva].min() <= yr[iva].max() < yr[ite].min()
    assert len(set(itr) & set(ite)) == 0


def test_scalers_fit_on_train_only(prepared):
    itr = prepared["idx"][0]
    assert np.isclose(prepared["ys"].data_min_[0], prepared["y_raw"][itr].min())
    assert prepared["X"][itr].min() >= -1e-6 and prepared["X"][itr].max() <= 1 + 1e-6


def test_metrics_perfect_and_imperfect():
    m = regression_metrics([1, 2, 3], [1, 2, 3])
    assert m["RMSE"] == 0 and m["R2"] == 1
    assert regression_metrics([2, 4], [3, 3])["MAE"] == 1


class FakeModel:  # monotone in mean scaled rainfall, so the what-if sliders have an effect
    def predict(self, x, verbose=0):
        return x[:, :, 0].mean(axis=1, keepdims=True)


@pytest.fixture(scope="module")
def predictor(prepared):
    df = _clean(generate())
    return Predictor(model=FakeModel(), df=df, xs=prepared["xs"], ys=prepared["ys"],
                     meta=prepared["meta"], rmse=0.3)


def test_predict_and_scenarios(predictor):
    base = predictor.predict("Ludhiana", "wheat", 2020)
    wet = predictor.predict("Ludhiana", "wheat", 2020, rain_change_pct=50)
    assert wet["predicted"] > base["predicted"]
    assert base["low"] <= base["predicted"] <= base["high"]
    assert len(base["weather"]) == 12 and base["history"]


@pytest.mark.parametrize("kw", [dict(district="Nowhere", crop="wheat", year=2020),
                                dict(district="Ludhiana", crop="cotton", year=2020),
                                dict(district="Ludhiana", crop="wheat", year=1800),
                                dict(district="Ludhiana", crop="wheat", year=2020, rain_change_pct=500),
                                dict(district="Ludhiana", crop="wheat", year="abc")])
def test_invalid_inputs(predictor, kw):
    with pytest.raises(InputError):
        predictor.predict(**kw)


def test_api(predictor):
    c = create_app(predictor).test_client()
    assert c.get("/").status_code == 200
    assert "districts" in c.get("/api/options").get_json()
    ok = c.post("/api/predict", json={"district": "Hisar", "crop": "rice", "year": 2015})
    assert ok.status_code == 200 and ok.get_json()["predicted"] >= 0
    assert c.post("/api/predict", json={"district": "x"}).status_code == 400
