"""Train the stacked LSTM, train baselines, evaluate all on the chronological test set.

    python -m src.train               # full run
    python -m src.train --epochs 5    # quick smoke test
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config  # noqa: E402
from src import preprocess  # noqa: E402
from src.metrics import regression_metrics  # noqa: E402


def keras_fit(builder, d, epochs, name):
    from src.models import callbacks
    itr, iva, ite = d["idx"]
    T, F = d["X"].shape[1:]
    model = builder(T, F)
    hist = model.fit(d["X"][itr], d["y"][itr], validation_data=(d["X"][iva], d["y"][iva]),
                     epochs=epochs, batch_size=config.BATCH_SIZE, callbacks=callbacks(), verbose=2)
    pred = d["ys"].inverse_transform(model.predict(d["X"][ite], verbose=0)).ravel()
    print(f"[{name}] trained for {len(hist.history['loss'])} epochs")
    return model, hist.history, pred


def sk_baselines(d):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression
    itr, iva, ite = d["idx"]
    flat = lambda a: a.reshape(len(a), -1)  # noqa: E731
    Xtr, Xte = flat(d["X"][itr]), flat(d["X"][ite])
    ytr = d["y_raw"][itr]
    models = {"Linear Regression": LinearRegression(),
              "Random Forest": RandomForestRegressor(300, random_state=config.SEED, n_jobs=-1)}
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(n_estimators=400, learning_rate=0.05, max_depth=4,
                                         subsample=0.9, random_state=config.SEED)
    except ImportError:
        print("xgboost not installed -> skipping XGBoost baseline")
    return {n: m.fit(Xtr, ytr).predict(Xte) for n, m in models.items()}


def plots(results, y_test, pred_lstm, hist, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(results)
    plt.figure(figsize=(7, 4))
    bars = plt.bar(names, [results[n]["RMSE"] for n in names],
                   color=["#9aa9a4"] * (len(names) - 1) + ["#2E6F95"])
    plt.ylabel("RMSE (t/ha)"); plt.title("Test RMSE by model (lower is better)")
    plt.xticks(rotation=25, ha="right"); plt.bar_label(bars, fmt="%.3f", fontsize=8); plt.tight_layout()
    plt.savefig(outdir / "fig_rmse_by_model.png", dpi=150); plt.close()
    plt.figure(figsize=(6, 4))
    plt.plot(hist["loss"], label="train"); plt.plot(hist["val_loss"], label="validation")
    plt.xlabel("epoch"); plt.ylabel("MSE (scaled)"); plt.title("LSTM training / validation loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(outdir / "fig_loss_curve.png", dpi=150); plt.close()
    plt.figure(figsize=(5, 5))
    lo, hi = min(y_test.min(), pred_lstm.min()), max(y_test.max(), pred_lstm.max())
    plt.scatter(y_test, pred_lstm, s=12, alpha=.7); plt.plot([lo, hi], [lo, hi], "k--", lw=1)
    plt.xlabel("Actual yield (t/ha)"); plt.ylabel("Predicted yield (t/ha)")
    plt.title("LSTM: actual vs predicted (test)"); plt.tight_layout()
    plt.savefig(outdir / "fig_actual_vs_pred.png", dpi=150); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.EPOCHS)
    args = ap.parse_args()

    import tensorflow as tf
    tf.keras.utils.set_random_seed(config.SEED)

    if not config.DATA_CSV.exists():
        from data.generate_synthetic import generate
        config.DATA_CSV.parent.mkdir(exist_ok=True)
        generate().to_csv(config.DATA_CSV, index=False)
        print("No dataset found -> generated SYNTHETIC demo data (replace with real data).")

    d = preprocess.prepare()
    itr, iva, ite = d["idx"]
    print(f"samples: train={len(itr)} val={len(iva)} test={len(ite)} | "
          f"test years {d['meta']['split_years']['test']}")
    y_test = d["y_raw"][ite]

    from src.models import build_gru, build_lstm, build_rnn
    lstm, hist, p_lstm = keras_fit(build_lstm, d, args.epochs, "LSTM")
    _, _, p_rnn = keras_fit(build_rnn, d, args.epochs, "Simple RNN")
    _, _, p_gru = keras_fit(build_gru, d, args.epochs, "GRU")
    preds = sk_baselines(d)
    preds.update({"Simple RNN": p_rnn, "GRU": p_gru, "Stacked LSTM (proposed)": p_lstm})

    results = {n: regression_metrics(y_test, p) for n, p in preds.items()}
    table = pd.DataFrame(results).T.round(4)
    config.OUT_DIR.mkdir(exist_ok=True)
    table.to_csv(config.OUT_DIR / "metrics_table.csv")
    print("\nTest-set performance (original units, t/ha):\n", table.to_string())

    lstm.save(config.MODEL_DIR / "lstm_yield.keras")
    (config.MODEL_DIR / "test_metrics.json").write_text(json.dumps(results["Stacked LSTM (proposed)"], indent=2))
    out = pd.DataFrame({"actual": y_test, "predicted": p_lstm}).join(d["info"].iloc[ite].reset_index(drop=True))
    out.to_csv(config.OUT_DIR / "test_predictions.csv", index=False)
    plots(results, y_test, p_lstm, hist, config.OUT_DIR)
    print(f"\nSaved model -> {config.MODEL_DIR/'lstm_yield.keras'}; tables/figures -> {config.OUT_DIR}/")


if __name__ == "__main__":
    main()
