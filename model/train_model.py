import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'energy_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'features.pkl')

print("[INFO] Loading dataset...")

if not os.path.exists(DATA_PATH):
    print("[INFO] No real dataset found. Generating synthetic hourly energy data...")
    np.random.seed(42)
    dates = pd.date_range(start='2016-01-01', end='2018-12-31 23:00:00', freq='h')
    base = 20000
    hourly_pattern = 5000 * np.sin(2 * np.pi * (dates.hour / 24) - np.pi / 2)
    weekly_pattern = 3000 * np.sin(2 * np.pi * (dates.dayofweek / 7))
    yearly_pattern = 8000 * np.sin(2 * np.pi * (dates.dayofyear / 365) - np.pi / 2)
    noise = np.random.normal(0, 1500, len(dates))
    values = base + hourly_pattern + weekly_pattern + yearly_pattern + noise
    values = np.clip(values, 5000, 50000)
    df = pd.DataFrame({'Datetime': dates, 'PJME_MW': values.astype(int)})
    df.to_csv(DATA_PATH, index=False)
    print(f"[INFO] Synthetic dataset saved to {DATA_PATH}")
else:
    df = pd.read_csv(DATA_PATH, parse_dates=['Datetime'])

print(f"[INFO] Dataset shape: {df.shape}")
print(df.head(3))

print("[INFO] Extracting time-based features...")
df['hour'] = df['Datetime'].dt.hour
df['day_of_week'] = df['Datetime'].dt.dayofweek
df['month'] = df['Datetime'].dt.month

features = ['hour', 'day_of_week', 'month']
X = df[features]
y = df['PJME_MW']

print("[INFO] Splitting into train (80%) and test (20%)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, shuffle=False
)

print("[INFO] Training Linear Regression model...")
model = LinearRegression()
model.fit(X_train, y_train)

print("[INFO] Making predictions on test set...")
y_pred = model.predict(X_test)

mae = mean_absolute_error(y_test, y_pred)
print(f"[RESULT] Mean Absolute Error (MAE): {mae:.2f} MW")
print(f"[RESULT] R^2 Score: {model.score(X_test, y_test):.4f}")

print("[INFO] Saving model and feature list...")
with open(MODEL_PATH, 'wb') as f:
    pickle.dump(model, f)
with open(FEATURES_PATH, 'wb') as f:
    pickle.dump(features, f)

print(f"[INFO] Model saved to {MODEL_PATH}")
print("[DONE] Training complete.")
