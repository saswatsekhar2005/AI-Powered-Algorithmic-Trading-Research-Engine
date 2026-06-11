import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load dataset
df = pd.read_csv("data/RELIANCE_LABELED.csv")

print(df.columns.tolist())
# Remove rows with indicator NaN values
df = df.dropna()

# Features
features = [
    "SMA20",
    "SMA50",
    "EMA20",
    "EMA50",
    "RSI",
    "MACD",
    "ATR",
    "Returns",
    "Volatility",
    "Momentum_5",
    "Momentum_10",
    "Volume_Change"
]


X = df[features]
y = df["Target"]

# Time-series split
split_index = int(len(df) * 0.8)

X_train = X[:split_index]
X_test = X[split_index:]

y_train = y[:split_index]
y_test = y[split_index:]

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Metrics
print("Accuracy :", accuracy_score(y_test, predictions))
print("Precision:", precision_score(y_test, predictions))
print("Recall   :", recall_score(y_test, predictions))
print("F1 Score :", f1_score(y_test, predictions))
df["Returns"] = df["Close"].pct_change()

df["Volatility"] = (
    df["Returns"]
    .rolling(20)
    .std()
)

df["Momentum_5"] = (
    df["Close"] /
    df["Close"].shift(5)
)

df["Momentum_10"] = (
    df["Close"] /
    df["Close"].shift(10)
)

df["Volume_Change"] = (
    df["Volume"].pct_change()
)