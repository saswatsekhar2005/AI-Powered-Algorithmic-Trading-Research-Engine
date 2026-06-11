import pandas as pd
import pandas as pd

df = pd.read_csv("data/RELIANCE_LABELED.csv")
print(df.columns.tolist())

df = pd.read_csv("data/RELIANCE_FEATURES.csv")

df["Target"] = (
    df["Close"].shift(-1) > df["Close"]
).astype(int)

df = df[:-1]

df.to_csv(
    "data/RELIANCE_LABELED.csv",
    index=False
)

print("Labels created successfully")