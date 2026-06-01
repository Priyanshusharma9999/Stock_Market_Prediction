import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, r2_score
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

split = int(len(x) * 0.8)
x_train, x_test = x[:split], x[split:]
y_train, y_test = y[:split], y[split:]

x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))
x_test  = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(60, 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer="adam", loss="mean_squared_error")
model.fit(x_train, y_train, epochs=10, batch_size=32)

train_pred = model.predict(x_train)
pred = model.predict(x_test)

train_pred = scaler.inverse_transform(train_pred)
pred       = scaler.inverse_transform(pred)
y_train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))
y_test_actual  = scaler.inverse_transform(y_test.reshape(-1, 1))

train_mae = mean_absolute_error(y_train_actual, train_pred)
train_r2  = r2_score(y_train_actual, train_pred)
mae = mean_absolute_error(y_test_actual, pred)
r2  = r2_score(y_test_actual, pred)

print("\nTraining Accuracy:")
print("MAE:", round(train_mae, 2))
print("R2 Score:", round(train_r2, 4))

print("\nTesting Accuracy:")
print("MAE:", round(mae, 2))
print("R2 Score:", round(r2, 4))

plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label="Actual Price", color="blue")
plt.plot(pred, label="Predicted Price", color="red")
plt.title("Tata Motors Stock Prediction - LSTM")
plt.xlabel("Points")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()