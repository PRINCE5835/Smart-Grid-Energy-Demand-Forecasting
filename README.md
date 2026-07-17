# Smart Grid Energy Demand Forecasting

A beginner-friendly ML project that predicts hourly electricity demand (in Megawatts) using simple time-based features.

## Project Structure

```
├── dataset/          # Stores the energy consumption data (CSV)
├── model/            # Training script + saved model
│   ├── train_model.py
│   ├── model.pkl
│   └── features.pkl
├── backend/          # FastAPI server
│   └── app.py
├── frontend/         # HTML + JS client
│   └── index.html
├── requirements.txt
└── README.md
```

## Features Used

The model uses only **3 simple features** extracted from the timestamp:

| Feature        | Description                                      |
|----------------|--------------------------------------------------|
| `hour`         | Hour of the day (0–23). Captures daily patterns. |
| `day_of_week`  | Day of week (0=Mon, 6=Sun). Captures weekly trends. |
| `month`        | Month (1–12). Captures seasonal changes.          |

**Algorithm:** Linear Regression — a simple baseline model that learns linear relationships between the features and energy demand.

**Metric:** Mean Absolute Error (MAE) — average error in Megawatts.

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the model

```bash
python model/train_model.py
```

This script:
- Checks if `dataset/energy_data.csv` exists.
- If not, it **generates a realistic synthetic dataset** (no Kaggle login needed).
- Extracts hour, day_of_week, and month from the timestamp.
- Splits data into 80% train / 20% test.
- Trains a LinearRegression model.
- Prints the **MAE** and **R² score**.
- Saves `model.pkl` and `features.pkl` to the `model/` folder.

### 3. Start the API server

```bash
python backend/app.py
```

The server will start at `http://127.0.0.1:8000`.

### 4. Open the frontend

Open `frontend/index.html` in your browser. Pick a date/time and click **Get Forecast**.

### API Endpoint

**GET** `/predict?dt=YYYY-MM-DD HH:MM:SS`

Example:

```
http://127.0.0.1:8000/predict?dt=2025-06-15 14:00:00
```

Response:

```json
{
  "datetime": "2025-06-15 14:00:00",
  "predicted_load_mw": 28453.12,
  "features_used": {
    "hour": 14,
    "day_of_week": 6,
    "month": 6
  }
}
```

## Deploy to Render (Backend) + Netlify (Frontend)

### 1. Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

### 2. Deploy API on Render

1. Go to [render.com](https://render.com) and sign up / log in.
2. Click **New +** → **Web Service**.
3. Connect your GitHub repo.
4. Render will auto-detect the `render.yaml` — or manually fill:
   - **Name:** `smart-grid-energy-api`
   - **Runtime:** Python
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`
5. Click **Create Web Service**.
6. Wait for the deploy to finish, then copy your URL (e.g. `https://smart-grid-energy-api.onrender.com`).

### 3. Update Netlify proxy to point at your Render URL

Open `netlify.toml` and replace `YOUR_RENDER_URL` with your actual Render URL:

```toml
[[redirects]]
  from = "/api/*"
  to = "https://smart-grid-energy-api.onrender.com/:splat"
  status = 200
  force = true
```

Commit and push the change:

```bash
git add netlify.toml
git commit -m "Set Render backend URL"
git push
```

### 4. Deploy Frontend on Netlify

1. Go to [netlify.com](https://netlify.com) and log in.
2. Click **Add new site** → **Import an existing project**.
3. Connect your GitHub repo.
4. **Build settings:** leave defaults (publish directory = `frontend`).
5. Click **Deploy site**.
6. Done! Your frontend is live. The `/api/*` requests will automatically proxy to your Render backend.

## Notes

- If you have a real Kaggle dataset (e.g., PJME hourly energy consumption), place it at `dataset/energy_data.csv` with columns `Datetime` and `PJME_MW` — the script will use it automatically.
- The model is deliberately simple (LinearRegression on 3 features) to help beginners understand the full ML pipeline.
- Try improving accuracy by adding features like lag values, rolling averages, or temperature data!
