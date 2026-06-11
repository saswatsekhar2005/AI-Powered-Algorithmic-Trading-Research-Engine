import pandas as pd
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator
from ta.volatility import AverageTrueRange

df = pd.read_csv("data/RELIANCE.csv")

# Handle multi-level columns if yfinance created them
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
print(df.dtypes)
print(df.head())
# Indicators
df["SMA20"] = SMAIndicator(df["Close"], window=20).sma_indicator()
df["SMA50"] = SMAIndicator(df["Close"], window=50).sma_indicator()

df["EMA20"] = EMAIndicator(df["Close"], window=20).ema_indicator()
df["EMA50"] = EMAIndicator(df["Close"], window=50).ema_indicator()

df["RSI"] = RSIIndicator(df["Close"], window=14).rsi()

macd = MACD(df["Close"])
df["MACD"] = macd.macd()

atr = AverageTrueRange(
    high=df["High"],
    low=df["Low"],
    close=df["Close"],
    window=14
)

df["ATR"] = atr.average_true_range()
df["Returns"] = df["Close"].pct_change()
df["Volatility"] = df["Returns"].rolling(20).std()
df["Momentum_5"] = df["Close"] / df["Close"].shift(5)
df["Momentum_10"] = df["Close"] / df["Close"].shift(10)
df["Volume_Change"] = df["Volume"].pct_change()

df.to_csv("data/RELIANCE_FEATURES.csv", index=False)

print(df.tail())
print("Feature engineering completed.")