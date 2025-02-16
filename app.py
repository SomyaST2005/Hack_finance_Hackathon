import joblib
import pandas as pd
from flask import Flask, request, jsonify
import logging

rf_model = joblib.load("rf_fraud_detection.pkl")

expected_features = rf_model.feature_names_in_

app = Flask(__name__)

logging.basicConfig(
    filename='app.log',
    level=logging.INFO,      
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@app.route("/")
def home():
    return jsonify({"message": "Fraud Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        logging.info(f"Incoming request data: {data}")

        print("Received data:", data)  # Debugging step

        for field in expected_features:
            if field not in data:
                return jsonify({"error": f"Missing field: {field}"}), 400

        new_transaction_df = pd.DataFrame([data])

        print("Data before prediction:", new_transaction_df)  # Debugging step

        new_transaction_df = new_transaction_df[expected_features]

        prediction = rf_model.predict(new_transaction_df)
        logging.info(f"Prediction result: {prediction[0]}")

        return jsonify({"fraud": bool(prediction[0])})
    
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)