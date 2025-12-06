import pandas as pd

df1 = pd.read_csv("combined_dataframe_NYSE.csv")
df2 = pd.read_csv("daily_sentiment_interpolated.csv")

merged = pd.merge(df1, df2, on="Date", how="inner")
merged.to_csv("combined_dataframe_NYSE_LLM.csv", index=False)
