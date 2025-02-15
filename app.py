import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Load the trained fraud detection model
rf_model = joblib.load("rf_fraud_detection.pkl")

# Get the list of features used during training
expected_features = rf_model.feature_names_in_

# Initialize Flask app
app = Flask(__name__)

@app.route("/")
def home():
    return jsonify({"message": "Fraud Detection API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get JSON data from request
        data = request.get_json()

        # Convert input to DataFrame
        new_transaction_df = pd.DataFrame([data])

        # Ensure only the expected features are used
        new_transaction_df = new_transaction_df[expected_features]

        # Predict fraud (1) or not fraud (0)
        prediction = rf_model.predict(new_transaction_df)

        # Return result
        return jsonify({"fraud": bool(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
