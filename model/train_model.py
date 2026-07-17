import csv
import pickle
import os
import math
from datetime import datetime, timedelta
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'dataset', 'energy_data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
FEATURES_PATH = os.path.join(os.path.dirname(__file__), 'features.pkl')

print("[INFO] Loading dataset...")

hours = []
days = []
months = []
loads = []

if not os.path.exists(DATA_PATH):
    print("[INFO] No real dataset found. Generating synthetic hourly energy data...")
    np.random.seed(42)
    start = datetime(2016, 1, 1, 0, 0, 0)
    total_hours = 26304
    for i in range(total_hours):
        current = start + timedelta(hours=i)
        h = current.hour
        d = current.weekday()
        m = current.month
        hour_of_year = current.timetuple().tm_yday * 24 + h
        base = 20000
        hourly_pattern = 5000 * math.sin(2 * math.pi * (h / 24) - math.pi / 2)
        weekly_pattern = 3000 * math.sin(2 * math.pi * (d / 7))
        yearly_pattern = 8000 * math.sin(2 * math.pi * (hour_of_year / 8760) - math.pi / 2)
        noise = np.random.normal(0, 1500)
        value = base + hourly_pattern + weekly_pattern + yearly_pattern + noise
        value = max(5000, min(50000, value))
        hours.append(h)
        days.append(d)
        months.append(m)
        loads.append(int(value))

    with open(DATA_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Datetime', 'PJME_MW'])
        for i in range(total_hours):
            dt = start + timedelta(hours=i)
            writer.writerow([dt.strftime('%Y-%m-%d %H:%M:%S'), loads[i]])
    print(f"[INFO] Synthetic dataset saved to {DATA_PATH}")
else:
    with open(DATA_PATH, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            dt = datetime.strptime(row['Datetime'], '%Y-%m-%d %H:%M:%S')
            hours.append(dt.hour)
            days.append(dt.weekday())
            months.append(dt.month)
            loads.append(float(row['PJME_MW']))

print(f"[INFO] Total samples loaded: {len(loads)}")

X = np.column_stack([hours, days, months])
y = np.array(loads)

features = ['hour', 'day_of_week', 'month']

print("[INFO] Splitting into train (80%) and test (20%)...")
split = int(len(X) * 0.8)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

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
