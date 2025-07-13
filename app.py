from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

model = joblib.load("diabetes_rf_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

expected_fields = [
    "Gender", "Age", "Polyuria", "Polydipsia", "sudden weight loss",
    "weakness", "Polyphagia", "Genital thrush", "visual blurring", "Itching",
    "Irritability", "delayed healing", "partial paresis", "muscle stiffness",
    "Alopecia", "Obesity"
]

@app.route("/predict", methods=["POST"])
def predict():
    try:
        input_data = request.get_json()
        print("🔍 GELEN JSON VERİSİ:", input_data)

        if input_data is None:
            return jsonify({"error": "JSON verisi alınamadı."}), 400

        values = []
        for field in expected_fields:
            val = input_data.get(field, "")

            if field in label_encoders and isinstance(val, str):
                val = label_encoders[field].transform([val])[0]
            else:
                try:
                    val = int(val)
                except ValueError:
                    return jsonify({"error": f"{field} alanı sayı olmalı ama şu geldi: {val}"}), 400

            values.append(val)

        prediction = model.predict([values])[0]
        result = label_encoders["class"].inverse_transform([prediction])[0]
        return jsonify({"prediction": result})

    except Exception as e:
        print("❌ HATA:", e)
        return jsonify({"error": str(e)}), 400

import os

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render'ın sağladığı PORT değişkeni
    app.run(host="0.0.0.0", port=port, debug=True)
