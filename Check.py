import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.read_csv("data.csv")

df["Target"] = df["Close"].shift(-1)
df = df.dropna()

x = df[["Open", "High", "Low", "Close", "Volume"]]
y = df["Target"]

model = LinearRegression()
model.fit(x, y)

o = float(input("Open: "))
h = float(input("High: "))
l = float(input("Low: "))
c = float(input("Close: "))
v = float(input("Volume: "))

new = pd.DataFrame([[o, h, l, c, v]], columns=["Open", "High", "Low", "Close", "Volume"])
result = model.predict(new)

print("Predicted next day closing price:", round(result[0], 2))