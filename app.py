import pandas as pd
import plotly.express as px
import streamlit as st
import numpy as np
import joblib

# Load your trained model + encoders
rf_model = joblib.load("model_files/random_forest_model.pkl")
le_coin = joblib.load("model_files/le_coin_encoder.pkl")
le_sentiment = joblib.load("model_files/le_sentiment_encoder.pkl")

st.set_page_config(page_title="Crypto Sentiment Dashboard", layout="wide")
@st.cache_data
def load_data():
    df = pd.read_csv("csv_files/merged_trading_sentiment_data.csv")
    df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
    df['Timestamp IST'] = pd.to_datetime(df['Timestamp IST'], errors='coerce')
    df.dropna(subset=['date', 'Timestamp IST'], inplace=True)

    df['Estimated Leverage'] = abs(df['Start Position']) / df['Size USD'].replace(0, np.nan)
    df['Estimated Leverage'].replace([np.inf, -np.inf], np.nan, inplace=True)
    return df
df=load_data()

#Sidebar Filters
st.sidebar.title("🔍 Filters")
sentiment_filter = st.sidebar.multiselect("Sentiment", options=df['classification'].unique(), default=df['classification'].unique())
coin_filter = st.sidebar.multiselect("Coin", options=df['Coin'].unique(), default=df['Coin'].unique())
start_date = st.sidebar.date_input("Start Date", df['date'].min())
end_date = st.sidebar.date_input("End Date", df['date'].max())

filtered_df = df[
    (df['classification'].isin(sentiment_filter)) &
    (df['Coin'].isin(coin_filter)) &
    (df['date'] >= pd.to_datetime(start_date)) &
    (df['date'] <= pd.to_datetime(end_date))
]

st.title("Crypto Trading Dashboard and Sentiment Analysis")

#KPI Metrics
col1, col2, col3 = st.columns(3)
col1.metric("Avg PnL", f"${filtered_df['Closed PnL'].mean():.2f}")
col2.metric("Avg Leverage", f"{filtered_df['Estimated Leverage'].mean():.2f}x")
col3.metric("Total Trades", f"{len(filtered_df)}")


#1. PnL Distribution by Sentiment
st.subheader("PnL Distribution by Sentiment")
fig1 = px.box(filtered_df, x="classification", y="Closed PnL", color="classification")
st.plotly_chart(fig1, use_container_width=True)

#2. Estimated Leverage vs Closed PnL
st.subheader("Estimated Leverage vs PnL")
fig2 = px.scatter(filtered_df, x="Estimated Leverage", y="Closed PnL", color="classification", trendline="ols")
st.plotly_chart(fig2, use_container_width=True)

#3. Daily Average PnL Trend
st.subheader("Daily Avg PnL Trend")
daily_avg = filtered_df.groupby('date')['Closed PnL'].mean().reset_index()
fig3 = px.line(daily_avg, x='date', y='Closed PnL', title='Daily Average PnL')
st.plotly_chart(fig3, use_container_width=True)

#4. Sentiment Distribution Over Time
st.subheader("Sentiment Over Time")
sentiment_trend = filtered_df.groupby(['date', 'classification']).size().reset_index(name='count')

fig4 = px.line(
    sentiment_trend,
    x="date",
    y="count",
    color="classification",
    markers=True,
)

fig4.update_layout(
    yaxis_title="Count of Sentiment Classifications",
    xaxis_title="Date",
    legend_title="Market Sentiment",
    template="plotly_white"
)
st.plotly_chart(fig4, use_container_width=True)

#5. Top Coins by Trade Volume
st.subheader("Top Traded Coins")
top_coins = filtered_df.groupby('Coin')['Size USD'].sum().reset_index().sort_values(by='Size USD', ascending=False).head(10)
fig5 = px.bar(top_coins, x='Coin', y='Size USD', color='Coin')
st.plotly_chart(fig5, use_container_width=True)

#6. Win vs Loss Rate by Sentiment
st.subheader("Win/Loss Rate by Sentiment")
filtered_df['Win'] = filtered_df['Closed PnL'] > 0
sentiment_wins = filtered_df.groupby('classification')['Win'].mean().reset_index()
fig6 = px.bar(sentiment_wins, x='classification', y='Win', title='Win Rate by Sentiment')
st.plotly_chart(fig6, use_container_width=True)


st.download_button("📥 Download Filtered Data", filtered_df.to_csv(index=False), file_name="filtered_trades.csv")
st.markdown("---")
st.header("Predict Trade Profitability")

with st.form("predictor_form"):
    st.subheader("Enter Trade Details:")

    coin_input = st.selectbox("Coin", le_coin.classes_)
    leverage_input = st.slider("Leverage", 1, 100, 10)
    sentiment_input = st.selectbox("Market Sentiment", le_sentiment.classes_)
    side = st.radio("Position Type", ["Buy", "Sell"])
    hour = st.slider("Hour of Trade (0-23)", 0, 23, 12)
    weekday = st.slider("Day of Week (0 = Monday)", 0, 6, 3)
    size = st.number_input("Trade Size (tokens)", min_value=0.01, value=1.0, step=0.01)
    execution_price = st.number_input("Execution Price", min_value=0.01, value=100.0, step=0.01)

    submitted = st.form_submit_button("Predict Profitability")

    if submitted:
        # Feature Engineering
        entry_value = execution_price * size
        was_long = 1 if side.lower() == "buy" else 0
        is_weekend = 1 if weekday in [5, 6] else 0

        # Encode
        coin_encoded = le_coin.transform([coin_input])[0]
        sentiment_encoded = le_sentiment.transform([sentiment_input])[0]

        # Prepare input
        input_features = [[
            coin_encoded,
            leverage_input,
            sentiment_encoded,
            was_long,
            hour,
            weekday,
            is_weekend,
            entry_value
        ]]

        # Predict
        prediction = rf_model.predict(input_features)[0]
        probability = rf_model.predict_proba(input_features)[0][1]

        st.markdown("### 🧾 Prediction Result")
        if prediction == 1:
            st.success(f"This trade is likely to be **Profitable**! Confidence: `{probability:.2%}`")
        else:
            st.error(f"This trade is likely to be **Unprofitable**. Confidence: `{(1 - probability):.2%}`")

st.subheader("🧠 Dashboard Insights")

num_trades = len(filtered_df)
visible_coins = ', '.join(coin_filter) if coin_filter else "No coins selected"

st.markdown(f"""
###
- **Total Trades Displayed:** `{num_trades}`  
- **Date Range Selected:** `{start_date}` → `{end_date}`  
- **Coins in Focus:** `{visible_coins}`  
- **PnL vs. Sentiment:** Observe how trader performance shifts between **Fear** and **Greed** phases.  
- **Risk Behavior:** Track trader **Estimated Leverage** to understand high vs. low risk tendencies.  
""")

