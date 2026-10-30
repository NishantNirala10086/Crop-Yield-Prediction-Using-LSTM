"""Flask web application.   Run:  python -m app.app   (after training)."""
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app.predictor import InputError, Predictor  # noqa: E402


def create_app(predictor=None):
    app = Flask(__name__)
    state = {"p": predictor}

    def get_predictor():
        if state["p"] is None:
            state["p"] = Predictor()   # loads model + scalers once, on first request
        return state["p"]

    @app.get("/")
    def index():
        return render_template("index.html")

    @app.get("/api/options")
    def options():
        return jsonify(get_predictor().options())

    @app.get("/api/history")
    def history():
        d, c = request.args.get("district", ""), request.args.get("crop", "")
        p = get_predictor()
        if d not in p.meta["districts"] or c not in p.meta["crops"]:
            return jsonify(error="Unknown district or crop"), 400
        return jsonify(p.history(d, c))

    @app.post("/api/predict")
    def predict():
        body = request.get_json(silent=True) or {}
        try:
            return jsonify(get_predictor().predict(
                body.get("district"), body.get("crop"), body.get("year"),
                body.get("rain_change_pct", 0), body.get("temp_change_c", 0)))
        except InputError as e:
            return jsonify(error=str(e)), 400

    @app.errorhandler(FileNotFoundError)
    def missing_model(_):
        return jsonify(error="Model files not found. Run `python -m src.train` first."), 503

    return app


if __name__ == "__main__":
    create_app().run(host="127.0.0.1", port=5000, debug=False)
