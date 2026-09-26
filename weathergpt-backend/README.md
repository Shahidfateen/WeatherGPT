# WeatherGPT - Conversational AI Weather Assistant & 7-Day Forecast

> **Real-time weather reports, future 7-day predictions, Live GPS location tracking, Gemini conversational guide, and offline 7-day weather downloading for everyone.**

---

## 🌟 Key Features

1. **Open-Meteo REST API Integration (Real-Time & 7-Day Predictions):**
   - Supports optional **Open-Meteo API key** (commercial/enterprise tier) or automatic free keyless access.
   - Comprehensive atmospheric metrics: temperature, feels-like temperature, relative humidity, surface pressure, precipitation, rain probability, wind speed and direction, cloud cover, and UV index.
   - Full 7-day daily predictions (High/Low temps, weather codes & condition descriptions, precipitation probability, and wind speeds).

2. **Live GPS Tool & Reverse Geocoding:**
   - One-click **Live GPS** pinpointing using the browser's HTML5 Geolocation API (`navigator.geolocation`).
   - Reverse geocoding translates coordinates directly into city/region names (e.g., Chennai, Bengaluru, Mumbai, London).
   - Global and Indian city search with instant autocomplete.

3. **Gemini Conversational Weather Guide (For All People):**
   - Universal AI Weather Assistant powered by **Google Gemini** (`gemini-3.6-flash`).
   - Helps everyone plan their day: umbrella/rain alerts, clothing advice, outdoor sports/running suitability, and weekend travel outlook.
   - Multi-turn conversation support with voice output (Gemini audio and browser SpeechSynthesis) and voice input (Microphone speech-to-text).
   - Multilingual support across **English, Tamil (தமிழ்), Telugu (తెలుగు), Kannada (ಕನ್ನಡ), Malayalam (മലയാളം), and Hindi (हिन्दी)**.
   - Zero Baashini dependency.

4. **Offline 7-Day Weather Downloading & Offline Viewer:**
   - One-click **"Save 7-Day Offline"** button downloads next 7 days of weather and insights directly to `localStorage`.
   - **Export JSON / Report** file option to save and share weather reports when completely disconnected.
   - **Offline Mode** toggle and automatic network loss detection (`window.onoffline`) to view saved forecasts without internet.

5. **In-App API Key Configuration:**
   - Quick settings modal to enter or update your Gemini and Open-Meteo API keys directly from the browser UI or `.env`.

---

## 📁 Project Structure

```text
weathergpt-backend/
├── app/
│   ├── __init__.py
│   ├── main.py              # FastAPI application, REST endpoints & web app
│   ├── config.py            # Environment & API settings
│   ├── services/
│   │   ├── weather_service.py   # Open-Meteo ingestion, 7-day forecast & geocoding
│   │   ├── llm_service.py       # Gemini conversational guide & multilingual speech
│   │   ├── ml_service.py        # Micro-climate risk inference & heuristics
│   │   └── cache_service.py     # SQLite offline caching engine & audit log
│   ├── models/                  # Saved .joblib ML artifacts
│   │   ├── rain_classifier.joblib
│   │   ├── temp_shift_regressor.joblib
│   │   └── model_metadata.joblib
│   └── data/
│       └── weather_mock.json    # Offline fallback weather payload
├── train_prototype.py       # ML prototype training script
├── requirements.txt         # Project dependencies
├── .env.example             # Sample environment variables
├── .env                     # Local environment settings
└── README.md
```

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
py -m pip install -r requirements.txt
```

### 2. Configure API Keys (Optional)
Copy `.env.example` to `.env`:
```env
# Gemini API Key (Get a free key from https://aistudio.google.com/)
GEMINI_API_KEY=your_gemini_api_key_here

# Open-Meteo API Key (Optional; free tier works without a key from https://open-meteo.com/)
OPEN_METEO_API_KEY=your_openmeteo_api_key_here

# Default Coordinates
DEFAULT_LAT=13.0827
DEFAULT_LON=80.2707
```
*(Note: You can also enter and save API keys directly in the Web Application via the **Keys** button.)*

### 3. Run the Application Server
```bash
py -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Open your browser at:
- **Web Application:** `http://127.0.0.1:8000/`
- **Swagger API Docs:** `http://127.0.0.1:8000/docs`

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | **Interactive Visual Weather Application & AI Guide** |
| `GET` | `/health` | System health and API key configuration status |
| `GET` | `/api/weather` | Query real-time metrics and structured 7-day predictions |
| `POST` | `/api/chat` | Conversational Gemini AI Guide (multilingual) |
| `GET` | `/api/geocode/reverse` | Convert GPS coordinates to city & region name |
| `GET` | `/api/geocode/search` | Search global and Indian cities by name |
| `GET` | `/api/offline-bundle` | Download complete 7-day offline weather bundle |
| `POST` | `/api/settings/verify` | Test and verify Open-Meteo and Gemini API keys |
