import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

df = pd.read_csv("data.csv")
df = df[["Close"]]
df.dropna(inplace=True)

scaler = MinMaxScaler()
scaled = scaler.fit_transform(df)

x = []
y = []

for i in range(60, len(scaled)):
    x.append(scaled[i-60:i, 0])
    y.append(scaled[i, 0])

x = np.array(x)
y = np.array(y)

x = x.reshape((x.shape[0], x.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x, y, epochs=10, batch_size=32, verbose=0)

print("Model ready hai!")

last_60 = scaled[-60:]
last_60 = last_60.reshape(1, 60, 1)

pred = model.predict(last_60)
pred_price = scaler.inverse_transform(pred)

print("Predicted next day closing price:", round(pred_price[0][0], 2))