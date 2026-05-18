import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

df = pd.read_csv("data.csv")
print(df.head())

df["Target"] = df["Close"].shift(-1)
df = df.dropna()

x = df[["Open", "High", "Low", "Close", "Volume"]]
y = df["Target"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(x_train, y_train)

pred = model.predict(x_test)

mae = mean_absolute_error(y_test, pred)
r2 = r2_score(y_test, pred)

print("MAE:", mae)
print("R2 Score:", r2)

plt.figure(figsize=(10, 5))
plt.plot(y_test.values, label="Actual")
plt.plot(pred, label="Predicted")
plt.title("Tata Motors Stock Prediction")
plt.xlabel("Points")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()