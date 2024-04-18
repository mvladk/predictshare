from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from fetch_data import fetch_stock_data, preprocess_data

# Fetch and preprocess data
data = fetch_stock_data('AAPL')
X, y, scaler = preprocess_data(data)

# Build the model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Prediction of the next closing value

model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X, y, epochs=100, batch_size=32)

# Save the model and scaler
# model.save('stock_prediction_model.h5')
model.save('stock_prediction_model.keras')
