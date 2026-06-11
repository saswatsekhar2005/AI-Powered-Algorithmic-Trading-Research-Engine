import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv("data/RELIANCE_LABELED.csv")
df = df.dropna()

features = [
    "SMA20",
    "SMA50",
    "EMA20",
    "EMA50",
    "RSI",
    "MACD",
    "ATR"
]

X = df[features]
y = df["Target"]

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X, y)

joblib.dump(model, "saved_models/random_forest.pkl")

print("Model saved successfully")