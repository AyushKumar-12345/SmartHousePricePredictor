from flask import Flask, render_template, request, jsonify
import joblib
import json
import os

app = Flask(__name__)

try:
    model = joblib.load("models/model.pkl")
except:
    model = None

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        features = [
            float(data.get("MedInc", 3.5)),
            float(data.get("HouseAge", 28.0)),
            float(data.get("AveRooms", 5.0)),
            float(data.get("AveBedrms", 1.0)),
            float(data.get("Population", 1400.0)),
            float(data.get("AveOccup", 3.0)),
            float(data.get("Latitude", 35.5)),
            float(data.get("Longitude", -119.5))
        ]
        
        if model:
            prediction = model.predict([features])[0]
            prediction_usd = float(prediction) * 100000
        else:
            prediction_usd = 250000.0
            
        return jsonify({"success": True, "prediction": round(prediction_usd, 2)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route("/api/importance", methods=["GET"])
def importance():
    path = "models/importance.json"
    if os.path.exists(path):
        with open(path, "r") as f:
            return jsonify(json.load(f))
    return jsonify({
        "MedInc": 0.5, "HouseAge": 0.15, "AveRooms": 0.1, 
        "AveBedrms": 0.05, "Population": 0.05, "AveOccup": 0.1, 
        "Latitude": 0.03, "Longitude": 0.02
    })

if __name__ == "__main__":
    app.run(debug=True)