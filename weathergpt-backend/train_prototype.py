"""WeatherGPT - First-Time Prototype ML Engine Training Script.

Trains lightweight Scikit-Learn models using synthetic agro-meteorological features
tailored to South Indian agricultural micro-climates:
1. DecisionTreeClassifier: Predicts rain occurrence / rain risk.
2. RandomForestRegressor: Predicts 24-hour temperature shift (°C).

Saves joblib artifacts into app/models/ for local micro-climate risk scoring.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import joblib

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "app" / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

FEATURE_COLUMNS = [
    "temperature_2m",
    "relative_humidity_2m",
    "surface_pressure",
    "precipitation",
    "wind_speed_10m",
]

def generate_synthetic_weather_data(n_samples: int = 5000, random_seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic weather data for South Indian agricultural regions."""
    np.random.seed(random_seed)
    
    # 1. Temperature: 22°C to 42°C (tropical South Indian climate)
    temperature = np.random.normal(loc=31.5, scale=4.5, size=n_samples)
    temperature = np.clip(temperature, 18.0, 44.0)
    
    # 2. Relative Humidity: 35% to 95%
    relative_humidity = np.random.normal(loc=68.0, scale=16.0, size=n_samples)
    relative_humidity = np.clip(relative_humidity, 20.0, 99.0)
    
    # 3. Surface Pressure: 998 hPa to 1018 hPa (monsoon depressions drop pressure)
    surface_pressure = np.random.normal(loc=1009.0, scale=4.2, size=n_samples)
    surface_pressure = np.clip(surface_pressure, 990.0, 1025.0)
    
    # 4. Wind Speed: 2 to 35 km/h
    wind_speed = np.random.gamma(shape=2.5, scale=4.0, size=n_samples)
    wind_speed = np.clip(wind_speed, 1.0, 45.0)
    
    # 5. Precipitation at current observation: skewed, mostly 0
    precip_prob = np.where(relative_humidity > 75, 0.45, 0.08)
    has_precip = np.random.rand(n_samples) < precip_prob
    precipitation = np.where(has_precip, np.random.exponential(scale=3.5, size=n_samples), 0.0)
    precipitation = np.round(np.clip(precipitation, 0.0, 60.0), 2)
    
    # Target 1: Rain Probability (Synthetic physical formula + noise)
    # Higher humidity + low pressure + existing moisture + wind convergence = rain
    rain_score = (
        0.55 * (relative_humidity / 100.0)
        - 0.35 * ((surface_pressure - 1000.0) / 20.0)
        + 0.20 * (precipitation > 0)
        + 0.15 * (wind_speed / 40.0)
        + np.random.normal(0, 0.12, size=n_samples)
    )
    # Convert rain score to binary label
    rain_threshold = np.percentile(rain_score, 65)  # ~35% rain events
    rain_imminent = (rain_score > rain_threshold).astype(int)
    
    # Target 2: 24h Temperature Shift (°C)
    # If rain imminent, temp drops by 2-5°C. High humidity & low wind retains heat.
    temp_shift = (
        -3.2 * rain_imminent
        + 0.08 * (surface_pressure - 1009.0)
        - 0.05 * (relative_humidity - 65.0)
        + 0.06 * (wind_speed - 10.0)
        + np.random.normal(0, 0.75, size=n_samples)
    )
    temp_shift = np.round(np.clip(temp_shift, -7.0, 5.0), 2)
    
    df = pd.DataFrame({
        "temperature_2m": np.round(temperature, 2),
        "relative_humidity_2m": np.round(relative_humidity, 2),
        "surface_pressure": np.round(surface_pressure, 2),
        "precipitation": precipitation,
        "wind_speed_10m": np.round(wind_speed, 2),
        "rain_imminent": rain_imminent,
        "temp_shift_24h": temp_shift,
    })
    return df

def train_and_export_models():
    print("=" * 65)
    print("WeatherGPT: First-Time ML Prototype Training (SIH 26068)")
    print("=" * 65)
    
    print("\n[1/4] Generating synthetic agro-meteorological dataset...")
    df = generate_synthetic_weather_data(n_samples=6000, random_seed=42)
    print(f"Generated {len(df)} samples with features: {FEATURE_COLUMNS}")
    print(df.describe().round(2))
    
    X = df[FEATURE_COLUMNS]
    y_rain = df["rain_imminent"]
    y_shift = df["temp_shift_24h"]
    
    X_train, X_test, y_rain_train, y_rain_test, y_shift_train, y_shift_test = train_test_split(
        X, y_rain, y_shift, test_size=0.2, random_state=42
    )
    
    # 1. Train DecisionTreeClassifier for Rain Prediction
    print("\n[2/4] Training DecisionTreeClassifier for Rain Prediction...")
    clf = DecisionTreeClassifier(
        max_depth=5,
        min_samples_split=15,
        min_samples_leaf=10,
        random_state=42,
    )
    clf.fit(X_train, y_rain_train)
    rain_pred = clf.predict(X_test)
    rain_acc = accuracy_score(y_rain_test, rain_pred)
    print(f"DecisionTreeClassifier Rain Accuracy: {rain_acc * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_rain_test, rain_pred, target_names=["No Rain", "Rain Expected"]))
    
    # 2. Train RandomForestRegressor for 24h Temperature Shift
    print("\n[3/4] Training RandomForestRegressor for 24h Temperature Shift...")
    reg = RandomForestRegressor(
        n_estimators=50,
        max_depth=7,
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1,
    )
    reg.fit(X_train, y_shift_train)
    shift_pred = reg.predict(X_test)
    mae = mean_absolute_error(y_shift_test, shift_pred)
    rmse = np.sqrt(mean_squared_error(y_shift_test, shift_pred))
    r2 = r2_score(y_shift_test, shift_pred)
    print(f"RandomForestRegressor 24h Temp Shift Evaluation:")
    print(f"  MAE:  {mae:.3f} °C")
    print(f"  RMSE: {rmse:.3f} °C")
    print(f"  R²:   {r2:.3f}")
    
    # 3. Export Artifacts
    print("\n[4/4] Exporting model artifacts to app/models/...")
    clf_path = MODELS_DIR / "rain_classifier.joblib"
    reg_path = MODELS_DIR / "temp_shift_regressor.joblib"
    meta_path = MODELS_DIR / "model_metadata.joblib"
    
    joblib.dump(clf, clf_path)
    joblib.dump(reg, reg_path)
    
    metadata = {
        "features": FEATURE_COLUMNS,
        "rain_accuracy": float(rain_acc),
        "temp_shift_mae": float(mae),
        "temp_shift_r2": float(r2),
        "target_names": ["No Rain", "Rain Expected"],
        "trained_date": "2026-09-23",
        "description": "WeatherGPT SIH Prototype ML Models"
    }
    joblib.dump(metadata, meta_path)
    
    print(f" Saved: {clf_path}")
    print(f" Saved: {reg_path}")
    print(f" Saved: {meta_path}")
    print("\n Prototype ML training successfully completed!")
    print("=" * 65)

if __name__ == "__main__":
    train_and_export_models()
