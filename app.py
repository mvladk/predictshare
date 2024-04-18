from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
from keras.models import load_model
import yfinance as yf
from fetch_data import fetch_stock_data, preprocess_data

app = Flask(__name__)

# Load the trained model
model = load_model('stock_prediction_model.keras')

@app.route('/predict', methods=['POST'])
def predict():
    json_input = request.json
    symbol = json_input['symbol']
    # Fetch recent stock data
    recent_data = yf.download(symbol, period="24mo", interval="1d")
    recent_data = recent_data[['Close']]

    if len(recent_data) < 60:
        # Not enough data to make a prediction
        return jsonify(error="Not enough data to make a prediction"), 400

    try:
        # Preprocess the data
        X, y, scaler = preprocess_data(recent_data)
        last_60_days = recent_data[-60:].values
        last_60_days_scaled = scaler.transform(last_60_days)
        X_test = np.array([last_60_days_scaled])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        
        # Make prediction
        predicted_price = model.predict(X_test)
        predicted_price = scaler.inverse_transform(predicted_price)
        
        return jsonify(predicted_price=float(predicted_price[0][0]))
    except Exception as e:
        # Handle unexpected errors
        return jsonify(error=str(e)), 500

if __name__ == '__main__':
    app.run(debug=True)
