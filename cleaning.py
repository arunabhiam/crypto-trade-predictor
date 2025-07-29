import pandas as pd

trades = pd.read_csv('csv_files/historical_data.csv')
sentiment = pd.read_csv('csv_files/fear_greed_index.csv')
trades.columns = trades.columns.str.lower().str.replace(' ', '_')
sentiment.columns = sentiment.columns.str.lower().str.replace(' ', '_')
trades['leverage'] = trades.apply(
    lambda row: (row['execution_price'] * row['size_tokens']) / row['size_usd']
    if row['size_usd'] != 0 else 0,
    axis=1
)

trades.replace([float('inf'), -float('inf')], 0, inplace=True)
trades['leverage'] = trades['leverage'].fillna(0)
trades['date'] = pd.to_datetime(trades['timestamp_ist'], format="%d-%m-%Y %H:%M", errors='coerce').dt.date
sentiment['date'] = pd.to_datetime(sentiment['date'], errors='coerce').dt.date
merged = pd.merge(trades, sentiment[['date', 'classification']], on='date', how='left')
merged.rename(columns={'classification': 'sentiment'}, inplace=True)
merged = merged.dropna(subset=['closed_pnl', 'leverage', 'sentiment', 'coin'])
merged['profitable'] = merged['closed_pnl'].apply(lambda x: 1 if x > 0 else 0)
merged.to_csv('merged_pred.csv', index=False)