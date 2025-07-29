import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

merged = pd.read_csv('csv_files/merged_pred.csv')

# Feature Engineering
merged['entry_value'] = merged['execution_price'] * merged['size_tokens']
merged['was_long'] = (merged['side'].str.lower() == 'buy').astype(int)
merged['timestamp_ist'] = pd.to_datetime(merged['timestamp_ist'], format="%d-%m-%Y %H:%M", errors='coerce')
merged['hour'] = merged['timestamp_ist'].dt.hour
merged['weekday'] = merged['timestamp_ist'].dt.weekday
merged['is_weekend'] = merged['weekday'].isin([5, 6]).astype(int)

le_sentiment = LabelEncoder()
merged['sentiment'] = le_sentiment.fit_transform(merged['sentiment'])
le_coin = LabelEncoder()
merged['coin'] = le_coin.fit_transform(merged['coin'])

feature_cols = ['coin', 'leverage', 'sentiment', 'was_long', 'hour', 'weekday', 'is_weekend', 'entry_value']
X = merged[feature_cols]
y = merged['profitable']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

print("Classification Report:")
print(classification_report(y_test, y_pred))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Random Forest Confusion Matrix')
plt.show()

joblib.dump(rf_model, 'model_files/random_forest_model.pkl')
joblib.dump(le_coin, 'model_files/le_coin_encoder.pkl')
joblib.dump(le_sentiment, 'model_files/le_sentiment_encoder.pkl')
