from flask import Flask, request, jsonify, render_template
import sys
import os

app = Flask(__name__)


_predict = None

def get_predict():
    global _predict
    if _predict is None:
        
        from inference import predict
        _predict = predict
    return _predict


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = (data.get("text") or "").strip()

    if not text:
        return jsonify({"error": "No text provided"}), 400

    try:
        predict = get_predict()
        results = predict(text)
        return jsonify({"results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, port=5000)
