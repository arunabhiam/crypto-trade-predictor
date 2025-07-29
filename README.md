 Crypto Trade Analyzer & Profitability Predictor
---

This Streamlit-based dashboard helps visualize Crypto market sentiment, analyze real trading performance, and predict whether a future trade will be profitable or not using a trained Random Forest Classifier

## Overview

**Crypto Trade Analyzer** has 2 main modules:

1. **Market + Trader Analytics Dashboard**
   - Visualizes Crypto sentiment trends (Fear/Greed Index)
   - Displays historical trader activity (price, size, leverage, profit/loss)
   - Supports filters for deep-dive insights

2. **Profitability Predictor**
   - Input your hypothetical trade details (coin, leverage, sentiment, etc.)
   - Get a real-time prediction of whether the trade would be profitable
   - Powered by machine learning (Random Forest)



---

## Model Details

- **Model Used:** RandomForestClassifier (from scikit-learn)
- **Target Variable:** Whether a trade ended in profit (`closedPnL > 0`)
- **Features:**
  - Coin (encoded)
  - Market Sentiment (encoded)
  - Leverage
  - Trade Side (Buy/Sell → Long/Short)
  - Hour of Trade
  - Day of Week
  - Weekend Indicator
  - Entry Value (Execution Price × Size)

---

## How to Run Locally

1. Clone the repo:
git clone https://github.com/arunabhiam/crypto-trade-dashboard.git
2. Set up a virtual environment (recommended):
python -m venv venv
source venv/bin/activate
3. Install dependencies:
pip install -r requirements.txt
4. Run the dashboard:
streamlit run app.py
---
## Future Additions

- **Support for multiple coins** (ETH, SOL, etc.)
- **Add a model training tab** inside the dashboard
- **Real-time data integration** via APIs
- **A bot** for instant predictions
---
## Author
Built by **Arunabh**  
AI/ML Enthusiast 