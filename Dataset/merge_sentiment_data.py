import pandas as pd

df1 = pd.read_csv("sp500_with_indicators.csv")
df2 = pd.read_csv("daily_sentiment_interpolated.csv")

merged = pd.merge(df1, df2, on="Date", how="inner")
merged.to_csv("sp500_with_indicators_llm.csv", index=False)
