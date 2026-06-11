import yfinance as yf

df = yf.download(
    "RELIANCE.NS",
    start="2020-01-01",
    end="2025-12-31",
    auto_adjust=True
)

# Flatten yfinance multi-index columns
if hasattr(df.columns, "droplevel"):
    try:
        df.columns = df.columns.droplevel(1)
    except:
        pass

df.to_csv("data/RELIANCE.csv")

print(df.head())