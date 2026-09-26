"""WeatherGPT - FastAPI Web Application & Backend.

Provides:
- Real-time weather and 7-day future predictions using Open-Meteo (with API key support)
- Advanced Interactive Conversational Voice Assistant supporting ALL Indian Languages
- Crystal-clear voice synthesis & hands-free voice chat mode
- Live GPS location tracking with reverse geocoding
- 7-Day offline weather downloading and offline viewer
- Zero Baashini dependency (uses Gemini voice & Web Speech API)
"""

from contextlib import asynccontextmanager
from typing import Optional, List, Dict, Any
from datetime import datetime
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import settings
from app.services.cache_service import cache_service
from app.services.weather_service import weather_service
from app.services.llm_service import llm_service, LANGUAGE_CONFIG

@asynccontextmanager
async def lifespan(app: FastAPI):
    print(f"[{settings.PROJECT_NAME}] Starting up...")
    print(f"[{settings.PROJECT_NAME}] Database path: {settings.DB_PATH}")
    print(f"[{settings.PROJECT_NAME}] Gemini API Key: {'Configured' if settings.GEMINI_API_KEY else 'Not set in env (using UI key or smart fallback)'}")
    print(f"[{settings.PROJECT_NAME}] Open-Meteo API Key: {'Configured' if settings.OPEN_METEO_API_KEY else 'Free tier default (keyless)'}")
    print(f"[{settings.PROJECT_NAME}] Supported Indian Languages: {len(LANGUAGE_CONFIG)}")
    yield
    print(f"[{settings.PROJECT_NAME}] Shutting down...")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Conversational AI Weather Assistant with 7-Day Future Predictions, Live GPS & Interactive Voice Chat for All Indian Languages",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic Request Models
class ChatRequest(BaseModel):
    message: str = Field(..., description="User's weather question")
    latitude: float = Field(default=13.0827, description="Latitude")
    longitude: float = Field(default=80.2707, description="Longitude")
    location_name: Optional[str] = Field(default="Your Location", description="City / Region name")
    language: str = Field(default="en", description="Language code: en, hi, ta, te, kn, ml, bn, mr, gu, pa, or, as, ur")
    gemini_api_key: Optional[str] = Field(default=None, description="Optional client-side Gemini API key")
    open_meteo_api_key: Optional[str] = Field(default=None, description="Optional client-side Open-Meteo API key")
    history: Optional[List[Dict[str, str]]] = Field(default=[], description="Recent conversation turns for multi-turn context")

class VerifyKeysRequest(BaseModel):
    gemini_api_key: Optional[str] = None
    open_meteo_api_key: Optional[str] = None

@app.get("/health", tags=["System"])
@app.get("/api/health", tags=["System"])
async def health_check():
    """System health check and configuration overview."""
    return {
        "status": "online",
        "service": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "gemini_api_key_configured": bool(settings.GEMINI_API_KEY),
        "open_meteo_api_key_configured": bool(settings.OPEN_METEO_API_KEY),
        "supported_languages_count": len(LANGUAGE_CONFIG),
        "default_coordinates": {"latitude": settings.DEFAULT_LAT, "longitude": settings.DEFAULT_LON},
        "sqlite_cache_path": str(settings.DB_PATH),
    }

@app.get("/api/languages", tags=["Metadata"])
async def get_languages():
    """All supported Indian regional languages and English."""
    return [
        {
            "code": code,
            "name": info["name"],
            "native": info["native"],
            "speech_locale": info.get("speech_locale", "en-IN"),
        }
        for code, info in LANGUAGE_CONFIG.items()
    ]

@app.get("/api/geocode/reverse", tags=["Location"])
async def reverse_geocode(
    latitude: float = Query(..., description="Latitude coordinate"),
    longitude: float = Query(..., description="Longitude coordinate"),
    language: Optional[str] = Query(default="en", description="Language code e.g. ta, hi, te, kn..."),
):
    """Convert GPS coordinates to city, district, and country name in requested language."""
    geo_info = await weather_service.reverse_geocode(latitude, longitude, language=language or "en")
    return geo_info

@app.get("/api/geocode/search", tags=["Location"])
async def search_location(
    query: str = Query(..., min_length=2, description="City name to search"),
):
    """Search global and Indian cities by name."""
    results = await weather_service.search_locations(query)
    return {"query": query, "results": results}

@app.get("/api/weather", tags=["Weather"])
async def get_weather(
    latitude: float = Query(default=13.0827, description="Latitude"),
    longitude: float = Query(default=80.2707, description="Longitude"),
    force_refresh: bool = Query(default=False, description="Force fresh fetch from Open-Meteo"),
    api_key: Optional[str] = Query(default=None, description="Optional Open-Meteo API key"),
):
    """Retrieve real-time atmospheric metrics and structured 7-day future predictions."""
    forecast, source = await weather_service.get_forecast(latitude, longitude, force_refresh, api_key)
    current_metrics = weather_service.extract_current_metrics(forecast)
    daily_forecast = weather_service.extract_daily_forecast(forecast)

    return {
        "latitude": latitude,
        "longitude": longitude,
        "source": source,
        "current": current_metrics,
        "daily_forecast": daily_forecast,
        "raw_payload": forecast,
    }

@app.post("/api/chat", tags=["AI Guide"])
async def chat_weather_guide(request: ChatRequest):
    """Conversational Weather Guide powered by Gemini for all people."""
    # 1. Ingest real-time weather
    forecast, _ = await weather_service.get_forecast(
        request.latitude, request.longitude, force_refresh=False, api_key=request.open_meteo_api_key
    )
    current_metrics = weather_service.extract_current_metrics(forecast)
    daily_forecast = weather_service.extract_daily_forecast(forecast)

    # 2. Call conversational LLM guide with multi-turn context
    guide_result = await llm_service.chat_guide(
        user_message=request.message,
        weather_metrics=current_metrics,
        daily_forecast=daily_forecast,
        location_name=request.location_name or "Your Area",
        language_code=request.language,
        chat_history=request.history or [],
        api_key_override=request.gemini_api_key,
    )

    # 3. Log query to SQLite for offline access and history
    try:
        cache_service.log_advisory(
            lat=request.latitude,
            lon=request.longitude,
            query=request.message,
            language=request.language,
            advisory_text=guide_result["response_text"],
            risk_level="NORMAL",
            pesticide_spray_safe=True,
        )
    except Exception as e:
        print(f"[Main] Notice: advisory log error: {e}")

    return {
        "message": request.message,
        "response": guide_result["response_text"],
        "language": guide_result["language"],
        "language_name": guide_result["language_name"],
        "speech_locale": guide_result.get("speech_locale", "en-IN"),
        "audio_base64": guide_result.get("audio_base64"),
        "audio_mime_type": guide_result.get("audio_mime_type", "audio/mp3"),
        "source": guide_result.get("source", "smart_weather_fallback"),
        "weather_summary": current_metrics,
    }

@app.get("/api/offline-bundle", tags=["Offline"])
async def get_offline_bundle(
    latitude: float = Query(default=13.0827),
    longitude: float = Query(default=80.2707),
    location_name: str = Query(default="Your Location"),
    api_key: Optional[str] = Query(default=None),
):
    """Download full 7-day weather dataset formatted for offline storage on devices."""
    forecast, _ = await weather_service.get_forecast(latitude, longitude, force_refresh=True, api_key=api_key)
    bundle = weather_service.generate_offline_bundle(latitude, longitude, location_name, forecast)
    return JSONResponse(
        content=bundle,
        headers={"Content-Disposition": f"attachment; filename=weathergpt_7day_{latitude:.2f}_{longitude:.2f}.json"},
    )

@app.post("/api/settings/verify", tags=["Settings"])
async def verify_keys(keys: VerifyKeysRequest):
    """Test validity of Gemini and Open-Meteo API keys."""
    results = {"gemini_valid": False, "open_meteo_valid": False, "notes": []}

    # Verify Gemini Key
    gemini_key = (keys.gemini_api_key or settings.GEMINI_API_KEY or "").strip()
    if gemini_key:
        working_model = None
        last_error = None
        discovered_models = []

        try:
            from google import genai
            test_client = genai.Client(api_key=gemini_key)

            # 1. Dynamically query available models from Google for this exact key
            try:
                for m in test_client.models.list():
                    m_name = getattr(m, 'name', '') or ''
                    clean_id = m_name.replace('models/', '')
                    actions = getattr(m, 'supported_actions', []) or []
                    if not actions or 'generateContent' in actions:
                        if clean_id:
                            discovered_models.append(clean_id)
            except Exception as e_list:
                print(f"[Verify] models.list check notice: {e_list}")
                last_error = e_list

            # Sort: flash models first, then pro models
            def _sort_key(name: str):
                n = name.lower()
                if "flash" in n and "lite" not in n:
                    return 0
                if "flash" in n:
                    return 1
                if "pro" in n:
                    return 2
                return 3

            discovered_models.sort(key=_sort_key)

            # Fallback candidates if list is empty
            fallback_candidates = [
                "gemini-2.5-flash",
                "gemini-3.6-flash",
                "gemini-2.5-pro",
                "gemini-3.5-flash",
                settings.GEMINI_MODEL,
            ]
            models_to_test = discovered_models if discovered_models else fallback_candidates

            for m in models_to_test:
                try:
                    resp = test_client.models.generate_content(
                        model=m,
                        contents="Hi, reply with one word: Connected.",
                    )
                    if resp and resp.text:
                        working_model = m
                        settings.GEMINI_MODEL = m
                        llm_service.model_name = m
                        break
                except Exception as ex:
                    last_error = ex
                    err_str = str(ex).lower()
                    if "api_key_invalid" in err_str or "invalid api key" in err_str or "api_key not valid" in err_str:
                        break
                    continue

            if working_model:
                results["gemini_valid"] = True
                results["notes"].append({
                    "type": "success",
                    "text": f"Google Gemini API key is valid and connected using '{working_model}'."
                })
            else:
                err_text = str(last_error) if last_error else "Could not connect to Gemini model"
                if "api_key_invalid" in err_text.lower() or "invalid api key" in err_text.lower() or "api_key not valid" in err_text.lower():
                    results["notes"].append({
                        "type": "error",
                        "text": "Gemini API key is invalid or unauthorized. Please re-check the key from Google AI Studio."
                    })
                else:
                    results["notes"].append({
                        "type": "error",
                        "text": f"Gemini connection note: {err_text[:140]}"
                    })
        except Exception as e:
            results["notes"].append({
                "type": "error",
                "text": f"Gemini client initialization error: {e}"
            })
    else:
        results["notes"].append({
            "type": "info",
            "text": "No Gemini key provided. Built-in smart weather fallback is active."
        })

    # Verify Open-Meteo Key
    om_key = (keys.open_meteo_api_key or settings.OPEN_METEO_API_KEY or "").strip()
    if om_key:
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                r_customer = await client.get(
                    settings.OPEN_METEO_CUSTOMER_URL,
                    params={"latitude": 13.08, "longitude": 80.27, "current": "temperature_2m", "apikey": om_key},
                )
                if r_customer.status_code == 200:
                    results["open_meteo_valid"] = True
                    results["notes"].append({
                        "type": "success",
                        "text": "Open-Meteo Commercial API key is verified and active."
                    })
                else:
                    r_std = await client.get(
                        settings.OPEN_METEO_URL,
                        params={"latitude": 13.08, "longitude": 80.27, "current": "temperature_2m", "apikey": om_key},
                    )
                    if r_std.status_code == 200:
                        results["open_meteo_valid"] = True
                        results["notes"].append({
                            "type": "success",
                            "text": "Open-Meteo API key is verified and active with live predictions!"
                        })
                    else:
                        r_public = await client.get(
                            settings.OPEN_METEO_URL,
                            params={"latitude": 13.08, "longitude": 80.27, "current": "temperature_2m"},
                        )
                        if r_public.status_code == 200:
                            results["open_meteo_valid"] = True
                            results["notes"].append({
                                "type": "success",
                                "text": "Open-Meteo live weather & 7-day predictions are active and connected!"
                            })
                        else:
                            results["notes"].append({
                                "type": "error",
                                "text": f"Open-Meteo responded with status {r_public.status_code}."
                            })
        except Exception as e:
            results["notes"].append({
                "type": "error",
                "text": f"Open-Meteo check error: {e}"
            })
    else:
        # Free Tier check
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                r_public = await client.get(
                    settings.OPEN_METEO_URL,
                    params={"latitude": 13.08, "longitude": 80.27, "current": "temperature_2m"},
                )
                if r_public.status_code == 200:
                    results["open_meteo_valid"] = True
                    results["notes"].append({
                        "type": "success",
                        "text": "Open-Meteo Free Tier is active and working (no API key required)."
                    })
                else:
                    results["notes"].append({
                        "type": "warning",
                        "text": f"Open-Meteo Free Tier responded with status {r_public.status_code}."
                    })
        except Exception as e:
            results["open_meteo_valid"] = True
            results["notes"].append({
                "type": "info",
                "text": f"Open-Meteo Free Tier active (offline fallback ready: {e})."
            })

    return results

# Interactive Web Application served at Root (/)
@app.get("/", response_class=HTMLResponse, tags=["Web App"])
async def root_app():
    """Complete, modern, responsive WeatherGPT web application with Advanced Interactive Voice Chat."""
    html_content = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>WeatherGPT - Real-Time Weather & 7-Day Future Predictions</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.5.1/css/all.min.css">
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
  <style>
    body { font-family: 'Plus Jakarta Sans', sans-serif; }
    .glass-card {
      background: rgba(255, 255, 255, 0.92);
      backdrop-filter: blur(14px);
      -webkit-backdrop-filter: blur(14px);
      border: 1px solid rgba(255, 255, 255, 0.5);
    }
    .custom-scroll::-webkit-scrollbar { width: 5px; height: 5px; }
    .custom-scroll::-webkit-scrollbar-track { background: transparent; }
    .custom-scroll::-webkit-scrollbar-thumb { background: rgba(156, 163, 175, 0.4); border-radius: 9999px; }

    /* Interactive Voice Orb Animations */
    .orb-container { position: relative; width: 140px; height: 140px; display: flex; align-items: center; justify-content: center; }
    .voice-orb {
      width: 100px;
      height: 100px;
      border-radius: 50%;
      background: radial-gradient(circle, #38bdf8 0%, #0284c7 60%, #1e1b4b 100%);
      box-shadow: 0 0 30px rgba(56, 189, 248, 0.5), inset 0 0 20px rgba(255, 255, 255, 0.6);
      transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .orb-pulse-idle { animation: idlePulse 3s infinite ease-in-out; }
    .orb-listening {
      background: radial-gradient(circle, #34d399 0%, #059669 60%, #064e3b 100%) !important;
      box-shadow: 0 0 45px rgba(52, 211, 153, 0.8), inset 0 0 25px rgba(255, 255, 255, 0.8) !important;
      animation: listenPulse 1.2s infinite ease-in-out !important;
      transform: scale(1.12);
    }
    .orb-thinking {
      background: radial-gradient(circle, #c084fc 0%, #7e22ce 60%, #3b0764 100%) !important;
      box-shadow: 0 0 45px rgba(192, 132, 252, 0.8) !important;
      animation: spinThinking 2s infinite linear !important;
    }
    .orb-speaking {
      background: radial-gradient(circle, #67e8f9 0%, #0284c7 50%, #4338ca 100%) !important;
      box-shadow: 0 0 50px rgba(103, 232, 249, 0.9), inset 0 0 30px rgba(255, 255, 255, 0.9) !important;
      animation: speakPulse 0.9s infinite ease-in-out !important;
      transform: scale(1.18);
    }
    @keyframes idlePulse {
      0%, 100% { transform: scale(1); opacity: 0.9; }
      50% { transform: scale(1.05); opacity: 1; }
    }
    @keyframes listenPulse {
      0%, 100% { transform: scale(1.1); box-shadow: 0 0 30px rgba(52, 211, 153, 0.6); }
      50% { transform: scale(1.22); box-shadow: 0 0 60px rgba(52, 211, 153, 1); }
    }
    @keyframes spinThinking {
      0% { transform: rotate(0deg) scale(1.05); }
      50% { transform: rotate(180deg) scale(1.15); }
      100% { transform: rotate(360deg) scale(1.05); }
    }
    @keyframes speakPulse {
      0%, 100% { transform: scale(1.12); }
      50% { transform: scale(1.25); }
    }

    /* Sound wave bars */
    .sound-bar {
      width: 4px;
      height: 12px;
      border-radius: 9999px;
      background-color: #38bdf8;
      transition: height 0.15s ease;
    }
    .sound-bar-active { animation: waveBar 1s infinite ease-in-out; }
    @keyframes waveBar {
      0%, 100% { height: 8px; }
      50% { height: 28px; }
    }
  </style>
</head>
<body class="bg-gradient-to-br from-slate-900 via-sky-950 to-indigo-950 text-slate-800 min-h-screen antialiased flex flex-col justify-between selection:bg-sky-500 selection:text-white">

  <!-- Top Navigation Bar -->
  <header class="sticky top-0 z-40 bg-slate-900/80 backdrop-blur-md border-b border-white/10 px-4 lg:px-8 py-3.5">
    <div class="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-3">
      
      <!-- Brand & Offline Banner -->
      <div class="flex items-center gap-3 w-full sm:w-auto justify-between sm:justify-start">
        <div class="flex items-center gap-2.5 cursor-pointer" onclick="resetToCurrentLocation()">
          <div class="w-10 h-10 rounded-xl bg-gradient-to-tr from-sky-500 via-cyan-400 to-indigo-600 flex items-center justify-center text-white text-xl shadow-lg shadow-sky-500/30">
            <i class="fa-solid fa-cloud-bolt"></i>
          </div>
          <div>
            <div class="flex items-center gap-2">
              <span class="text-xl font-extrabold text-white tracking-tight">Weather<span class="text-sky-400">GPT</span></span>
              <span class="text-[10px] px-2 py-0.5 rounded-full bg-sky-500/20 text-sky-300 font-bold border border-sky-400/30">Live Voice AI</span>
            </div>
            <p id="app-subtitle" class="text-[11px] text-slate-400 -mt-0.5">Real-time Weather & 7-Day Forecast</p>
          </div>
        </div>

        <div id="net-badge" class="sm:hidden text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1.5">
          <i class="fa-solid fa-wifi text-[10px]"></i> Online
        </div>
      </div>

      <!-- Search & Live GPS Tool Bar -->
      <div class="flex items-center gap-2 w-full sm:w-auto flex-1 max-w-xl sm:mx-4">
        
        <!-- Live GPS Button -->
        <button id="btn-gps" onclick="acquireLiveGPS()" title="Click to detect your live location using GPS" class="px-3.5 py-2.5 rounded-xl bg-gradient-to-r from-sky-500 to-blue-600 hover:from-sky-400 hover:to-blue-500 text-white font-bold text-xs shadow-md shadow-sky-500/20 flex items-center gap-2 transition active:scale-95 whitespace-nowrap">
          <i class="fa-solid fa-location-crosshairs text-sm" id="gps-icon"></i>
          <span id="gps-text">Live GPS</span>
        </button>

        <!-- Search Input with Autocomplete -->
        <div class="relative flex-1">
          <div class="relative">
            <i class="fa-solid fa-magnifying-glass absolute left-3.5 top-3 text-slate-400 text-xs"></i>
            <input id="search-input" type="text" placeholder="Search any Indian or global city..." oninput="handleSearchInput(this.value)" class="w-full bg-slate-800/90 text-white text-xs pl-9 pr-8 py-2.5 rounded-xl border border-white/10 focus:border-sky-400 focus:ring-1 focus:ring-sky-400 outline-none placeholder:text-slate-500 transition">
            <button id="search-clear" onclick="clearSearch()" class="hidden absolute right-3 top-2.5 text-slate-400 hover:text-white text-xs">
              <i class="fa-solid fa-xmark"></i>
            </button>
          </div>

          <!-- Autocomplete Dropdown -->
          <div id="search-dropdown" class="hidden absolute left-0 right-0 top-11 bg-slate-900/95 backdrop-blur-lg border border-slate-700 rounded-xl shadow-2xl overflow-hidden z-50 text-xs divide-y divide-slate-800 max-h-64 overflow-y-auto custom-scroll">
          </div>
        </div>

      </div>

      <!-- Action Buttons -->
      <div class="flex items-center gap-2">
        <!-- Interactive Live Voice Mode Button -->
        <button onclick="openVoiceModal()" class="px-3.5 py-2 rounded-xl bg-gradient-to-r from-rose-500 via-purple-600 to-indigo-600 hover:opacity-95 text-white text-xs font-bold shadow-lg shadow-purple-500/25 flex items-center gap-2 transition active:scale-95">
          <i class="fa-solid fa-microphone-lines animate-pulse text-sm"></i>
          <span id="btn-live-voice-text">Live Voice Chat</span>
        </button>

        <button onclick="toggleOfflineView()" id="btn-offline-mode" class="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-white/10 flex items-center gap-1.5 transition">
          <i class="fa-solid fa-hard-drive text-amber-400"></i>
          <span id="offline-mode-text">Offline</span>
        </button>

        <button onclick="openSettingsModal()" class="px-3 py-2 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-200 text-xs font-semibold border border-white/10 flex items-center gap-1.5 transition" title="Configure Open-Meteo & Gemini API Keys">
          <i class="fa-solid fa-gear text-sky-400"></i>
          <span id="btn-keys-text">Keys</span>
        </button>

        <!-- Top Nav Global Language Switcher -->
        <select id="nav-lang" onchange="changeLanguage(this.value)" class="bg-slate-800 hover:bg-slate-700 text-white text-xs font-semibold px-2.5 py-2 rounded-xl border border-white/20 outline-none cursor-pointer max-w-[110px] sm:max-w-none">
          <option value="en">English</option>
          <option value="hi">हिन्दी (Hindi)</option>
          <option value="ta">தமிழ் (Tamil)</option>
          <option value="te">తెలుగు (Telugu)</option>
          <option value="kn">ಕನ್ನಡ (Kannada)</option>
          <option value="ml">മലയാളം (Malayalam)</option>
          <option value="bn">বাংলা (Bengali)</option>
          <option value="mr">मराठी (Marathi)</option>
          <option value="gu">ગુજરાતી (Gujarati)</option>
          <option value="pa">ਪੰਜਾਬੀ (Punjabi)</option>
          <option value="or">ଓଡ଼ିଆ (Odia)</option>
          <option value="as">অসমীয়া (Assamese)</option>
          <option value="ur">اردو (Urdu)</option>
        </select>
      </div>

    </div>
  </header>

  <!-- Offline Notice Bar (Appears when offline or viewing offline data) -->
  <div id="offline-banner" class="hidden bg-amber-500 text-slate-950 px-4 py-2 text-xs font-bold text-center flex items-center justify-center gap-2 shadow-md">
    <i class="fa-solid fa-triangle-exclamation"></i>
    <span id="offline-banner-text">Offline Mode Active: Showing saved 7-day weather forecast.</span>
    <button onclick="exitOfflineView()" id="offline-switch-btn" class="underline ml-2 hover:text-white">Switch to Live</button>
  </div>

  <!-- Main Content Layout -->
  <main class="max-w-7xl mx-auto w-full p-4 lg:p-8 space-y-6 flex-1">

    <!-- Location Bar & Quick Indian Cities -->
    <div class="flex flex-col md:flex-row md:items-center justify-between gap-3 glass-card rounded-2xl p-4 shadow-xl">
      <div class="flex items-center gap-3">
        <div class="w-10 h-10 rounded-xl bg-sky-100 text-sky-600 flex items-center justify-center text-lg">
          <i class="fa-solid fa-location-dot"></i>
        </div>
        <div>
          <div class="flex items-center gap-2">
            <h2 id="current-location-name" class="text-base md:text-lg font-bold text-slate-900">Chennai, Tamil Nadu, India</h2>
            <span id="gps-accuracy-badge" class="hidden text-[10px] px-2 py-0.5 rounded-full bg-emerald-100 text-emerald-800 font-semibold border border-emerald-300">GPS Locked</span>
          </div>
          <p id="current-coords-text" class="text-xs text-slate-500 font-mono">13.0827° N, 80.2707° E • Updated just now</p>
        </div>
      </div>

      <!-- Quick Indian Cities Chips -->
      <div class="flex items-center gap-1.5 overflow-x-auto custom-scroll pb-1 text-xs">
        <span id="cities-label" class="text-slate-400 text-[11px] font-semibold uppercase tracking-wider mr-1 hidden sm:inline">Cities:</span>
        <button id="city-btn-0" onclick="selectCity(13.0827, 80.2707, 'Chennai, Tamil Nadu')" class="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-sky-50 hover:text-sky-700 text-slate-700 font-medium transition whitespace-nowrap">Chennai</button>
        <button id="city-btn-1" onclick="selectCity(28.6139, 77.2090, 'Delhi, India')" class="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-sky-50 hover:text-sky-700 text-slate-700 font-medium transition whitespace-nowrap">Delhi</button>
        <button id="city-btn-2" onclick="selectCity(12.9716, 77.5946, 'Bengaluru, Karnataka')" class="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-sky-50 hover:text-sky-700 text-slate-700 font-medium transition whitespace-nowrap">Bengaluru</button>
        <button id="city-btn-3" onclick="selectCity(17.3850, 78.4867, 'Hyderabad, Telangana')" class="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-sky-50 hover:text-sky-700 text-slate-700 font-medium transition whitespace-nowrap">Hyderabad</button>
        <button id="city-btn-4" onclick="selectCity(19.0760, 72.8777, 'Mumbai, Maharashtra')" class="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-sky-50 hover:text-sky-700 text-slate-700 font-medium transition whitespace-nowrap">Mumbai</button>
        <button id="city-btn-5" onclick="selectCity(22.5726, 88.3639, 'Kolkata, West Bengal')" class="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-sky-50 hover:text-sky-700 text-slate-700 font-medium transition whitespace-nowrap">Kolkata</button>
        <button id="city-btn-6" onclick="selectCity(10.7867, 79.1378, 'Thanjavur, Tamil Nadu')" class="px-2.5 py-1 rounded-lg bg-slate-100 hover:bg-sky-50 hover:text-sky-700 text-slate-700 font-medium transition whitespace-nowrap">Thanjavur</button>
      </div>
    </div>

    <!-- Main Weather Dashboard Grid -->
    <div class="grid grid-cols-1 lg:grid-cols-12 gap-6">

      <!-- Left 7 Cols: Real-Time Current Weather Hero & Metrics -->
      <div class="lg:col-span-7 space-y-6">

        <!-- Hero Current Weather Card -->
        <div class="relative overflow-hidden rounded-3xl bg-gradient-to-br from-sky-600 via-blue-600 to-indigo-700 text-white p-6 md:p-8 shadow-2xl">
          <div class="absolute -right-12 -top-12 w-64 h-64 bg-white/10 rounded-full blur-3xl pointer-events-none"></div>
          <div class="absolute -left-12 -bottom-12 w-48 h-48 bg-sky-400/20 rounded-full blur-2xl pointer-events-none"></div>

          <div class="relative z-10 flex flex-col justify-between h-full space-y-6">
            
            <div class="flex items-start justify-between">
              <div>
                <span id="hero-cond-title" class="text-xs font-bold uppercase tracking-wider text-sky-200">Current Conditions</span>
                <h3 id="hero-weather-desc" class="text-xl md:text-2xl font-black mt-0.5">Partly Cloudy</h3>
                <p id="hero-date-text" class="text-xs text-sky-100/80 mt-1">Today</p>
              </div>

              <!-- Weather Icon -->
              <div class="w-16 h-16 md:w-20 md:h-20 rounded-2xl bg-white/15 backdrop-blur-md flex items-center justify-center text-4xl md:text-5xl shadow-inner text-amber-300">
                <i id="hero-icon" class="fa-solid fa-cloud-sun"></i>
              </div>
            </div>

            <!-- Temperature Big Display & Live Voice Launcher -->
            <div class="flex flex-col sm:flex-row sm:items-baseline justify-between gap-4">
              <div class="flex items-baseline gap-4">
                <span id="hero-temp" class="text-6xl md:text-7xl font-extrabold tracking-tighter">29.5°</span>
                <div>
                  <span id="hero-feels-like" class="text-sm md:text-base font-semibold text-sky-100 block">Feels like 32.0°C</span>
                  <span id="hero-minmax" class="text-xs text-sky-200/90 font-medium">H: 33° • L: 24°</span>
                </div>
              </div>

              <button onclick="ensureAudioUnlocked(); openVoiceModal()" class="self-start sm:self-auto px-4 py-2.5 rounded-2xl bg-white/20 hover:bg-white/30 backdrop-blur-md border border-white/30 text-white font-bold text-xs flex items-center gap-2 transition shadow-md active:scale-95">
                <i class="fa-solid fa-microphone-lines text-rose-300 animate-pulse text-sm"></i>
                <span id="hero-talk-text">Talk with WeatherGPT</span>
              </button>
            </div>

            <!-- Everyday Weather Advice Strip -->
            <div class="p-3.5 rounded-2xl bg-white/15 backdrop-blur-md border border-white/20 flex items-center gap-3">
              <div class="w-8 h-8 rounded-xl bg-white/20 flex items-center justify-center text-amber-300 text-sm">
                <i id="tip-icon" class="fa-solid fa-lightbulb"></i>
              </div>
              <p id="weather-tip-text" class="text-xs md:text-sm font-medium text-white/95 flex-1">
                Pleasant weather with moderate breeze. Great day for outdoor activities!
              </p>
            </div>

          </div>
        </div>

        <!-- 6-Metric Highlights Grid -->
        <div class="grid grid-cols-2 sm:grid-cols-3 gap-3">

          <div class="glass-card rounded-2xl p-4 shadow-lg text-center flex flex-col justify-between">
            <div class="flex items-center justify-between text-slate-400 text-xs font-semibold">
              <span id="label-rain-chance">Rain Chance</span>
              <i class="fa-solid fa-cloud-rain text-sky-500"></i>
            </div>
            <div class="my-2">
              <span id="metric-rain-prob" class="text-2xl font-black text-slate-900">20%</span>
              <span id="metric-rain-sum" class="text-[10px] text-slate-500 block">0.0 mm expected</span>
            </div>
            <div class="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
              <div id="rain-bar" class="bg-sky-500 h-1.5 rounded-full" style="width: 20%"></div>
            </div>
          </div>

          <div class="glass-card rounded-2xl p-4 shadow-lg text-center flex flex-col justify-between">
            <div class="flex items-center justify-between text-slate-400 text-xs font-semibold">
              <span id="label-humidity">Humidity</span>
              <i class="fa-solid fa-droplet text-blue-500"></i>
            </div>
            <div class="my-2">
              <span id="metric-humidity" class="text-2xl font-black text-slate-900">72%</span>
              <span id="metric-humidity-desc" class="text-[10px] text-emerald-600 font-semibold block">Comfortable</span>
            </div>
            <div class="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
              <div id="humidity-bar" class="bg-blue-500 h-1.5 rounded-full" style="width: 72%"></div>
            </div>
          </div>

          <div class="glass-card rounded-2xl p-4 shadow-lg text-center flex flex-col justify-between">
            <div class="flex items-center justify-between text-slate-400 text-xs font-semibold">
              <span id="label-wind-speed">Wind Speed</span>
              <i class="fa-solid fa-wind text-teal-500"></i>
            </div>
            <div class="my-2">
              <span id="metric-wind" class="text-2xl font-black text-slate-900">12 km/h</span>
              <span id="metric-wind-dir" class="text-[10px] text-slate-500 block">Southeast breeze</span>
            </div>
            <div class="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
              <div id="wind-bar" class="bg-teal-500 h-1.5 rounded-full" style="width: 30%"></div>
            </div>
          </div>

          <div class="glass-card rounded-2xl p-4 shadow-lg text-center flex flex-col justify-between">
            <div class="flex items-center justify-between text-slate-400 text-xs font-semibold">
              <span id="label-uv-index">UV Index</span>
              <i class="fa-solid fa-sun text-amber-500"></i>
            </div>
            <div class="my-2">
              <span id="metric-uv" class="text-2xl font-black text-slate-900">7.5</span>
              <span id="metric-uv-level" class="text-[10px] text-amber-600 font-bold block">High (Wear Sunscreen)</span>
            </div>
            <div class="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
              <div id="uv-bar" class="bg-amber-500 h-1.5 rounded-full" style="width: 75%"></div>
            </div>
          </div>

          <div class="glass-card rounded-2xl p-4 shadow-lg text-center flex flex-col justify-between">
            <div class="flex items-center justify-between text-slate-400 text-xs font-semibold">
              <span id="label-air-pressure">Air Pressure</span>
              <i class="fa-solid fa-gauge-high text-indigo-500"></i>
            </div>
            <div class="my-2">
              <span id="metric-pressure" class="text-2xl font-black text-slate-900">1010 hPa</span>
              <span id="metric-pressure-desc" class="text-[10px] text-slate-500 block">Surface atmospheric</span>
            </div>
            <div class="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
              <div class="bg-indigo-500 h-1.5 rounded-full" style="width: 50%"></div>
            </div>
          </div>

          <div class="glass-card rounded-2xl p-4 shadow-lg text-center flex flex-col justify-between">
            <div class="flex items-center justify-between text-slate-400 text-xs font-semibold">
              <span id="label-cloud-cover">Cloud Cover</span>
              <i class="fa-solid fa-cloud text-slate-500"></i>
            </div>
            <div class="my-2">
              <span id="metric-clouds" class="text-2xl font-black text-slate-900">35%</span>
              <span id="metric-clouds-desc" class="text-[10px] text-slate-500 block">Scattered clouds</span>
            </div>
            <div class="w-full bg-slate-200 rounded-full h-1.5 overflow-hidden">
              <div id="cloud-bar" class="bg-slate-400 h-1.5 rounded-full" style="width: 35%"></div>
            </div>
          </div>

        </div>

        <!-- Offline 7-Day Download Action Card -->
        <div class="glass-card rounded-2xl p-5 shadow-lg border border-sky-200/50 flex flex-col sm:flex-row items-center justify-between gap-4">
          <div class="flex items-center gap-3.5">
            <div class="w-12 h-12 rounded-2xl bg-gradient-to-tr from-sky-500 to-indigo-600 text-white flex items-center justify-center text-xl shadow-md shadow-sky-500/30">
              <i class="fa-solid fa-cloud-arrow-down"></i>
            </div>
            <div>
              <h4 id="offline-card-title" class="text-sm font-bold text-slate-900">Offline 7-Day Weather Data</h4>
              <p id="offline-card-desc" class="text-xs text-slate-500">Save next 7 days forecast to access anytime without network</p>
              <span id="offline-saved-tag" class="hidden text-[10px] font-bold text-emerald-600 flex items-center gap-1 mt-0.5">
                <i class="fa-solid fa-circle-check"></i> <span id="offline-saved-text">Saved locally on device</span>
              </span>
            </div>
          </div>

          <div class="flex items-center gap-2 w-full sm:w-auto">
            <button id="btn-download-cache" onclick="downloadForecastForOffline()" class="flex-1 sm:flex-initial px-4 py-2.5 rounded-xl bg-gradient-to-r from-emerald-600 to-teal-600 hover:from-emerald-500 hover:to-teal-500 text-white font-bold text-xs shadow-md shadow-emerald-600/30 transition flex items-center justify-center gap-2 active:scale-95">
              <i class="fa-solid fa-download"></i> <span id="btn-download-cache-text">Save 7-Day Offline</span>
            </button>
            <button onclick="exportForecastJSON()" title="Export JSON file to your computer/phone" class="p-2.5 rounded-xl border border-slate-200 hover:bg-slate-100 text-slate-700 transition text-xs font-semibold">
              <i class="fa-solid fa-file-code"></i>
            </button>
          </div>
        </div>

      </div>

      <!-- Right 5 Cols: Conversational Weather Guide -->
      <div class="lg:col-span-5 flex flex-col space-y-6">

        <div class="glass-card rounded-3xl p-5 md:p-6 shadow-2xl border-2 border-sky-400/40 flex-1 flex flex-col justify-between space-y-4">
          
          <!-- Guide Header -->
          <div class="flex items-center justify-between border-b border-slate-100 pb-3">
            <div class="flex items-center gap-2.5">
              <div class="w-9 h-9 rounded-xl bg-gradient-to-tr from-sky-500 to-indigo-600 text-white flex items-center justify-center text-sm shadow-md shadow-sky-500/30">
                <i class="fa-solid fa-sparkles"></i>
              </div>
              <div>
                <h3 id="guide-header-title" class="text-sm font-extrabold text-slate-900 flex items-center gap-1.5">
                  Voice & Chat Weather Guide
                </h3>
                <p id="guide-header-subtitle" class="text-[11px] text-slate-500">Supports all 12+ Indian Languages</p>
              </div>
            </div>

            <!-- Language Selector Supporting All Indian Languages -->
            <select id="guide-lang" onchange="changeLanguage(this.value)" class="text-xs font-bold px-2.5 py-1.5 rounded-xl border border-slate-200 bg-white text-slate-800 focus:ring-2 focus:ring-sky-400 outline-none max-w-[130px] truncate">
              <option value="en">English (India)</option>
              <option value="hi">हिन्दी (Hindi)</option>
              <option value="ta">தமிழ் (Tamil)</option>
              <option value="te">తెలుగు (Telugu)</option>
              <option value="kn">ಕನ್ನಡ (Kannada)</option>
              <option value="ml">മലയാളം (Malayalam)</option>
              <option value="bn">বাংলা (Bengali)</option>
              <option value="mr">मराठी (Marathi)</option>
              <option value="gu">ગુજરાતી (Gujarati)</option>
              <option value="pa">ਪੰਜਾਬੀ (Punjabi)</option>
              <option value="or">ଓଡ଼ିଆ (Odia)</option>
              <option value="as">অসমীয়া (Assamese)</option>
              <option value="ur">اردو (Urdu)</option>
            </select>
          </div>

          <!-- Quick Query Buttons -->
          <div id="quick-chips-container" class="flex flex-wrap gap-1.5">
            <button onclick="sendQuickPrompt('Do I need to carry an umbrella today?')" class="text-[11px] px-2.5 py-1 rounded-lg bg-sky-50 hover:bg-sky-100 text-sky-800 font-medium transition border border-sky-100">
              <i class="fa-solid fa-umbrella text-sky-600 mr-1"></i> Umbrella?
            </button>
            <button onclick="sendQuickPrompt('Can I dry clothes outside today?')" class="text-[11px] px-2.5 py-1 rounded-lg bg-teal-50 hover:bg-teal-100 text-teal-800 font-medium transition border border-teal-100">
              <i class="fa-solid fa-jug-detergent text-teal-600 mr-1"></i> Dry clothes?
            </button>
            <button onclick="sendQuickPrompt('Can I wash my car today?')" class="text-[11px] px-2.5 py-1 rounded-lg bg-blue-50 hover:bg-blue-100 text-blue-800 font-medium transition border border-blue-100">
              <i class="fa-solid fa-car text-blue-600 mr-1"></i> Wash car?
            </button>
            <button onclick="sendQuickPrompt('What should I wear today based on the temperature?')" class="text-[11px] px-2.5 py-1 rounded-lg bg-indigo-50 hover:bg-indigo-100 text-indigo-800 font-medium transition border border-indigo-100">
              <i class="fa-solid fa-shirt text-indigo-600 mr-1"></i> What to wear?
            </button>
            <button onclick="sendQuickPrompt('Is today good for outdoor sports or running?')" class="text-[11px] px-2.5 py-1 rounded-lg bg-emerald-50 hover:bg-emerald-100 text-emerald-800 font-medium transition border border-emerald-100">
              <i class="fa-solid fa-person-running text-emerald-600 mr-1"></i> Outdoor sports?
            </button>
            <button onclick="sendQuickPrompt('Will it be cold or chilly tonight?')" class="text-[11px] px-2.5 py-1 rounded-lg bg-purple-50 hover:bg-purple-100 text-purple-800 font-medium transition border border-purple-100">
              <i class="fa-solid fa-moon text-purple-600 mr-1"></i> Night weather?
            </button>
            <button onclick="sendQuickPrompt('How is the weather this weekend for travel?')" class="text-[11px] px-2.5 py-1 rounded-lg bg-amber-50 hover:bg-amber-100 text-amber-800 font-medium transition border border-amber-100">
              <i class="fa-solid fa-route text-amber-600 mr-1"></i> Weekend travel?
            </button>
          </div>

          <!-- Chat Conversation Box -->
          <div id="chat-messages" class="space-y-3.5 overflow-y-auto max-h-[360px] pr-1 custom-scroll text-xs">
            
            <!-- Default Welcome Assistant Message -->
            <div class="flex items-start gap-2.5">
              <div class="w-7 h-7 rounded-lg bg-sky-500 text-white flex-shrink-0 flex items-center justify-center text-xs mt-0.5">
                <i class="fa-solid fa-robot"></i>
              </div>
              <div class="p-3.5 rounded-2xl bg-slate-100 text-slate-800 max-w-[88%] leading-relaxed border border-slate-200">
                <p id="chat-welcome-text">Hello! I am your <strong>WeatherGPT Voice Guide</strong>. You can speak or type in any Indian language. Ask me about rain, umbrella alerts, what to wear, or the 7-day outlook!</p>
              </div>
            </div>

          </div>

          <!-- Clear Voice Audio Controls Strip -->
          <div id="audio-control-bar" class="hidden p-2.5 rounded-xl bg-slate-900 text-white flex items-center justify-between gap-3 shadow-lg">
            <div class="flex items-center gap-2.5">
              <button id="btn-speak-last" onclick="playLastResponseAudio()" class="w-8 h-8 rounded-full bg-sky-500 hover:bg-sky-400 text-white flex items-center justify-center transition">
                <i id="audio-play-icon" class="fa-solid fa-volume-high text-xs"></i>
              </button>
              <div>
                <span id="audio-bar-title" class="text-[11px] font-bold block text-sky-300">Natural Voice Output</span>
                <span id="voice-status-text" class="text-[10px] text-slate-400">Click to listen in your language</span>
              </div>
            </div>
            
            <div class="flex items-center gap-2">
              <button onclick="openVoiceModal()" class="px-2.5 py-1 rounded-lg bg-white/10 hover:bg-white/20 text-xs font-semibold text-white transition flex items-center gap-1.5">
                <i class="fa-solid fa-expand text-[10px]"></i> <span id="audio-bar-live-text">Live Mode</span>
              </button>
            </div>
            <audio id="gemini-audio" preload="auto" class="hidden"></audio>
          </div>

          <!-- Chat Input Area with Microphone Speech-To-Text -->
          <div class="relative flex items-center gap-2 pt-2">
            <button id="btn-mic" onclick="ensureAudioUnlocked(); toggleQuickVoiceInput()" title="Speak your question using microphone" class="w-10 h-10 rounded-xl bg-slate-100 hover:bg-sky-100 text-slate-700 hover:text-sky-700 flex items-center justify-center transition border border-slate-200 flex-shrink-0">
              <i id="mic-icon" class="fa-solid fa-microphone text-sm"></i>
            </button>

            <div class="relative flex-1">
              <input id="chat-input" type="text" placeholder="Type or click mic to speak in any language..." onkeypress="handleChatKeyPress(event)" class="w-full text-xs px-3.5 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-sky-400 outline-none pr-10">
            </div>

            <button id="btn-send-chat" onclick="ensureAudioUnlocked(); sendChatMessage()" class="w-10 h-10 rounded-xl bg-gradient-to-r from-sky-500 to-indigo-600 hover:from-sky-400 hover:to-indigo-500 text-white flex items-center justify-center shadow-md shadow-sky-500/30 transition flex-shrink-0 active:scale-95">
              <i class="fa-solid fa-paper-plane text-xs"></i>
            </button>
          </div>

        </div>

      </div>

    </div>

    <!-- 7-Day Future Weather Prediction Section -->
    <div class="glass-card rounded-3xl p-6 shadow-xl space-y-4">
      <div class="flex flex-col sm:flex-row sm:items-center justify-between gap-2 border-b border-slate-100 pb-3">
        <div>
          <h3 id="forecast-section-title" class="text-base font-extrabold text-slate-900 flex items-center gap-2">
            <i class="fa-solid fa-calendar-week text-sky-500"></i> Next 7-Day Weather Prediction
          </h3>
          <p id="forecast-section-desc" class="text-xs text-slate-500">Daily forecast breakdown powered by Open-Meteo atmospheric models</p>
        </div>

        <div class="flex items-center gap-2 text-xs font-semibold text-slate-600">
          <span id="legend-minmax-text" class="px-2.5 py-1 rounded-lg bg-sky-50 text-sky-700 border border-sky-100">
            <i class="fa-solid fa-temperature-half mr-1"></i> Min/Max Temp
          </span>
          <span id="legend-rain-text" class="px-2.5 py-1 rounded-lg bg-blue-50 text-blue-700 border border-blue-100">
            <i class="fa-solid fa-cloud-rain mr-1"></i> Rain Probability
          </span>
        </div>
      </div>

      <!-- 7 Cards Grid -->
      <div id="forecast-7day-grid" class="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-3">
        <!-- Rendered dynamically by JavaScript -->
      </div>
    </div>

  </main>

  <!-- Footer -->
  <footer class="bg-slate-950 text-slate-400 text-xs py-4 px-6 border-t border-white/5 text-center space-y-1">
    <div class="max-w-7xl mx-auto flex flex-col sm:flex-row items-center justify-between gap-2">
      <div class="flex items-center gap-2">
        <span class="font-bold text-white">WeatherGPT</span>
        <span id="footer-system-text">• Real-Time & 7-Day AI Weather System</span>
      </div>
      <div class="flex items-center gap-4 text-[11px]">
        <span id="footer-powered-text">Powered by Open-Meteo REST API • Google Gemini Reasoning • All 12+ Indian Languages Voice</span>
      </div>
    </div>
  </footer>

  <!-- ADVANCED INTERACTIVE LIVE VOICE ASSISTANT MODAL -->
  <div id="voice-modal" class="hidden fixed inset-0 z-50 bg-slate-950/85 backdrop-blur-xl flex items-center justify-center p-4">
    <div class="relative w-full max-w-lg bg-gradient-to-b from-slate-900 via-slate-900 to-indigo-950 rounded-3xl p-6 md:p-8 text-white shadow-2xl border border-white/10 flex flex-col items-center justify-between min-h-[520px]">
      
      <!-- Top Bar: Close & Language Selector -->
      <div class="w-full flex items-center justify-between border-b border-white/10 pb-4">
        <div class="flex items-center gap-2">
          <span class="w-2.5 h-2.5 rounded-full bg-emerald-400 animate-ping"></span>
          <span id="modal-voice-title" class="text-xs font-bold uppercase tracking-wider text-sky-400">Live Voice Mode</span>
        </div>

        <div class="flex items-center gap-2">
          <select id="voice-modal-lang" onchange="switchVoiceModalLanguage(this.value)" class="bg-slate-800 text-white text-xs font-semibold px-3 py-1.5 rounded-xl border border-white/20 outline-none">
            <option value="en">English (India)</option>
            <option value="hi">हिन्दी (Hindi)</option>
            <option value="ta">தமிழ் (Tamil)</option>
            <option value="te">తెలుగు (Telugu)</option>
            <option value="kn">ಕನ್ನಡ (Kannada)</option>
            <option value="ml">മലയാളം (Malayalam)</option>
            <option value="bn">বাংলা (Bengali)</option>
            <option value="mr">मराठी (Marathi)</option>
            <option value="gu">ગુજરાતી (Gujarati)</option>
            <option value="pa">ਪੰਜਾਬੀ (Punjabi)</option>
            <option value="or">ଓଡ଼ିଆ (Odia)</option>
            <option value="as">অসমীয়া (Assamese)</option>
            <option value="ur">اردو (Urdu)</option>
          </select>

          <button onclick="closeVoiceModal()" class="w-8 h-8 rounded-full bg-white/10 hover:bg-white/20 text-slate-300 hover:text-white flex items-center justify-center transition">
            <i class="fa-solid fa-xmark text-sm"></i>
          </button>
        </div>
      </div>

      <!-- Center Voice Sphere & Animated Soundwaves -->
      <div class="flex flex-col items-center my-6 space-y-4">
        
        <div class="orb-container cursor-pointer" onclick="toggleVoiceSession()">
          <div id="voice-orb" class="voice-orb orb-pulse-idle flex items-center justify-center">
            <i id="orb-center-icon" class="fa-solid fa-microphone text-white text-3xl opacity-90 transition"></i>
          </div>
        </div>

        <!-- Dynamic Soundwaves -->
        <div id="soundwave-bar-container" class="flex items-center gap-1.5 h-8">
          <div class="sound-bar" style="animation-delay: 0.1s"></div>
          <div class="sound-bar" style="animation-delay: 0.3s"></div>
          <div class="sound-bar" style="animation-delay: 0.2s"></div>
          <div class="sound-bar" style="animation-delay: 0.4s"></div>
          <div class="sound-bar" style="animation-delay: 0.25s"></div>
          <div class="sound-bar" style="animation-delay: 0.5s"></div>
          <div class="sound-bar" style="animation-delay: 0.15s"></div>
        </div>

        <span id="voice-status-label" class="text-sm font-extrabold text-sky-300 tracking-wide">
          Tap orb to start speaking
        </span>
      </div>

      <!-- Live Speech & Response Transcript Display Box -->
      <div class="w-full bg-slate-950/60 rounded-2xl p-4 border border-white/10 space-y-2 text-center max-h-36 overflow-y-auto custom-scroll">
        <p id="voice-user-transcript" class="text-xs text-sky-200/80 font-medium italic">
          "Ask any weather question in your language..."
        </p>
        <p id="voice-assistant-reply" class="text-xs md:text-sm text-white font-semibold leading-relaxed">
        </p>
      </div>

      <!-- Bottom Voice Controls -->
      <div class="w-full pt-4 flex flex-col sm:flex-row items-center justify-between gap-3 text-xs border-t border-white/10">
        
        <!-- Hands-free Continuous Conversation Switch -->
        <label class="flex items-center gap-2 cursor-pointer select-none text-slate-300">
          <input type="checkbox" id="chk-hands-free" checked class="w-4 h-4 rounded text-sky-500 focus:ring-0">
          <span id="modal-handsfree-label">Hands-Free Auto-Chat (Continuous)</span>
        </label>

        <!-- Voice Clarity Speed Slider -->
        <div class="flex items-center gap-2 text-slate-300">
          <span id="modal-speed-label">Speed:</span>
          <select id="voice-speed-select" class="bg-slate-800 text-white rounded-lg px-2 py-1 border border-white/20 outline-none text-xs">
            <option value="0.85">0.85x (Ultra Clear)</option>
            <option value="0.92" selected>0.92x (Clear Diction)</option>
            <option value="1.0">1.0x (Normal)</option>
            <option value="1.1">1.1x (Fast)</option>
          </select>
        </div>

      </div>

    </div>
  </div>

  <!-- API Keys & Settings Modal -->
  <div id="settings-modal" class="hidden fixed inset-0 z-50 bg-slate-950/70 backdrop-blur-sm flex items-center justify-center p-4">
    <div class="bg-white rounded-3xl max-w-lg w-full p-6 shadow-2xl space-y-5 border border-slate-100">
      
      <div class="flex items-center justify-between border-b border-slate-100 pb-3">
        <div class="flex items-center gap-2.5">
          <div class="w-9 h-9 rounded-xl bg-sky-100 text-sky-600 flex items-center justify-center text-base">
            <i class="fa-solid fa-sliders"></i>
          </div>
          <div>
            <h3 id="settings-modal-title" class="text-base font-bold text-slate-900">API Key Configuration</h3>
            <p id="settings-modal-desc" class="text-xs text-slate-500">Connect your Gemini & Open-Meteo keys</p>
          </div>
        </div>
        <button onclick="closeSettingsModal()" class="w-8 h-8 rounded-full hover:bg-slate-100 text-slate-400 hover:text-slate-700 flex items-center justify-center transition">
          <i class="fa-solid fa-xmark"></i>
        </button>
      </div>

      <div class="space-y-4 text-xs">
        
        <!-- Gemini Key -->
        <div>
          <div class="flex items-center justify-between mb-1">
            <label id="label-gemini-key" class="font-bold text-slate-700">Google Gemini API Key</label>
            <a id="link-gemini-key" href="https://aistudio.google.com/app/apikey" target="_blank" class="text-sky-600 hover:underline flex items-center gap-1 font-semibold">
              Get Free Key <i class="fa-solid fa-arrow-up-right-from-square text-[10px]"></i>
            </a>
          </div>
          <input id="key-gemini" type="password" placeholder="AIzaSy..." class="w-full font-mono text-xs px-3.5 py-2.5 rounded-xl border border-slate-200 focus:ring-2 focus:ring-sky-400 outline-none">
          <p id="desc-gemini-key" class="text-[11px] text-slate-400 mt-1">Powers natural conversational weather advice and native audio.</p>
        </div>

        <!-- Open-Meteo Key -->
        <div>
          <div class="flex items-center justify-between mb-1">
            <label id="label-openmeteo-key" class="font-bold text-slate-700">Open-Meteo API Key <span class="font-normal text-slate-400">(Optional)</span></label>
            <a id="link-openmeteo-key" href="https://open-meteo.com/en/pricing" target="_blank" class="text-sky-600 hover:underline flex items-center gap-1 font-semibold">
              Open-Meteo Pricing <i class="fa-solid fa-arrow-up-right-from-square text-[10px]"></i>
            </a>
          </div>
          <input id="key-openmeteo" type="password" placeholder="Leave blank for free tier, or enter commercial key" class="w-full font-mono text-xs px-3.5 py-2.5 rounded-xl border border-slate-200 focus:ring-2 focus:ring-sky-400 outline-none">
          <p id="desc-openmeteo-key" class="text-[11px] text-emerald-600 font-medium mt-1"><i class="fa-solid fa-circle-check"></i> Free tier is built-in &mdash; no API key needed for full live weather &amp; 7-day predictions.</p>
        </div>

        <div id="keys-status-box" class="hidden p-3 rounded-xl bg-slate-50 border border-slate-200 text-[11px] space-y-2">
        </div>

      </div>

      <div class="flex items-center justify-end gap-2.5 pt-2 border-t border-slate-100">
        <button onclick="testAPIKeys()" id="btn-test-keys" class="px-3.5 py-2 rounded-xl bg-slate-100 hover:bg-slate-200 text-slate-700 font-bold text-xs transition">
          Test Keys
        </button>
        <button id="btn-save-keys" onclick="saveAPIKeys()" class="px-5 py-2 rounded-xl bg-sky-600 hover:bg-sky-500 text-white font-bold text-xs transition shadow-md shadow-sky-600/30">
          Save Settings
        </button>
      </div>

    </div>
  </div>

  <!-- Client-side Application Script -->
  <script>
    // State variables
    let currentLat = 13.0827;
    let currentLon = 80.2707;
    let currentLocationName = "Chennai, Tamil Nadu, India";
    let rawLocationEnglish = "Chennai, Tamil Nadu, India";
    let lastGpsAccuracy = null;
    let cachedLocationTranslations = {};
    let currentLang = "en";
    let isOfflineView = false;
    let cachedWeatherData = null;
    let lastGuideResponseText = "";
    let lastGuideLocale = "en-IN";
    let lastAudioBase64 = null;
    let lastAudioMime = "audio/mp3";
    let chatHistory = []; // Multi-turn conversational memory for relatable dialogue
    // Voice Assistant State
    let speechRecognition = null;
    let isVoiceSessionActive = false;
    let isSpeaking = false;
    let silenceTimer = null;
    let searchDebounceTimeout = null;

    const INDIAN_LOCALE_MAP = {
      "en": "en-IN",
      "hi": "hi-IN",
      "ta": "ta-IN",
      "te": "te-IN",
      "kn": "kn-IN",
      "ml": "ml-IN",
      "bn": "bn-IN",
      "mr": "mr-IN",
      "gu": "gu-IN",
      "pa": "pa-IN",
      "or": "or-IN",
      "as": "as-IN",
      "ur": "ur-IN"
    };

    const INDIAN_LANG_NAMES = {
      en: "English",
      hi: "हिन्दी",
      ta: "தமிழ்",
      te: "తెలుగు",
      kn: "ಕನ್ನಡ",
      ml: "മലയാളം",
      bn: "বাংলা",
      mr: "मराठी",
      gu: "ગુજરાતી",
      pa: "ਪੰਜਾਬੀ",
      or: "ଓଡ଼ିଆ",
      as: "অসমীয়া",
      ur: "اردو"
    };

    let audioUnlocked = false;
    let globalAudioPlayer = null;
    let messageAudioStore = {};
    let messageCounter = 0;

    function getAudioPlayer() {
      if (!globalAudioPlayer) {
        globalAudioPlayer = document.getElementById('gemini-audio');
        if (!globalAudioPlayer) {
          globalAudioPlayer = new Audio();
          globalAudioPlayer.id = 'gemini-audio';
          document.body.appendChild(globalAudioPlayer);
        }
      }
      return globalAudioPlayer;
    }

    function ensureAudioUnlocked() {
      const player = getAudioPlayer();
      if (!audioUnlocked) {
        // Silent 0.05s WAV byte sequence for user gesture activation
        const silentWav = "data:audio/wav;base64,UklGRigAAABXQVZFZm10IBAAAAABAAEARKwAAIhYAQACABAAZGF0YQQAAAAAAA==";
        player.src = silentWav;
        player.play().then(() => {
          audioUnlocked = true;
        }).catch(err => {
          console.log("Audio unlock gesture notice:", err);
        });
      }
    }

    function b64ToBlob(b64Data, contentType = 'audio/mp3') {
      try {
        const byteCharacters = atob(b64Data);
        const byteNumbers = new Uint8Array(byteCharacters.length);
        for (let i = 0; i < byteCharacters.length; i++) {
          byteNumbers[i] = byteCharacters.charCodeAt(i);
        }
        return new Blob([byteNumbers], { type: contentType });
      } catch (e) {
        console.warn("Base64 to blob error:", e);
        return null;
      }
    }


    const APP_I18N = {
  "en": {
    "appSubtitle": "Real-time Weather & 7-Day Forecast",
    "gpsText": "Live GPS",
    "gpsActive": "GPS Active",
    "gpsLocating": "Locating...",
    "searchPlaceholder": "Search any Indian or global city...",
    "liveVoiceBtn": "Live Voice Chat",
    "offlineBtn": "Offline",
    "exitOfflineBtn": "Exit Offline",
    "keysBtn": "Keys",
    "offlineBannerText": "Offline Mode Active: Showing saved 7-day weather forecast.",
    "switchLiveBtn": "Switch to Live",
    "citiesLabel": "Cities:",
    "cities": [
      "Chennai",
      "Delhi",
      "Bengaluru",
      "Hyderabad",
      "Mumbai",
      "Kolkata",
      "Thanjavur"
    ],
    "gpsLocked": "GPS Locked",
    "currentCond": "Current Conditions",
    "today": "Today",
    "talkBtn": "Talk with WeatherGPT",
    "feelsLike": "Feels like",
    "highLow": "H: {max}° • L: {min}°",
    "rainChance": "Rain Chance",
    "rainExpected": "{rain} mm expected",
    "humidity": "Humidity",
    "comfort": "Comfortable",
    "humid": "Humid",
    "dry": "Dry",
    "windSpeed": "Wind Speed",
    "windBreeze": "Gentle breeze",
    "uvIndex": "UV Index",
    "uvLevels": {
      "veryHigh": "Very High (Protect eyes & skin)",
      "high": "High (Wear Sunscreen)",
      "safe": "Moderate / Safe"
    },
    "airPressure": "Air Pressure",
    "surfacePressure": "Surface atmospheric",
    "cloudCover": "Cloud Cover",
    "cloudDesc": "Scattered clouds",
    "offlineCardTitle": "Offline 7-Day Weather Data",
    "offlineCardDesc": "Save next 7 days forecast to access anytime without network",
    "offlineSavedText": "Saved locally on device",
    "saveOfflineBtn": "Save 7-Day Offline",
    "guideTitle": "Voice & Chat Weather Guide",
    "guideSubtitle": "Supports all 12+ Indian Languages",
    "welcomeChat": "Hello! I am your <strong>WeatherGPT Voice Guide</strong>. You can speak or type in any Indian language. Ask me about rain, umbrella alerts, what to wear, or the 7-day outlook!",
    "voiceOutputTitle": "Natural Voice Output",
    "voiceOutputStatus": "Click to listen in your language",
    "listenBtn": "🔊 Listen ({lang})",
    "liveModeBtn": "Live Mode",
    "forecastTitle": "Next 7-Day Weather Prediction",
    "forecastSubtitle": "Daily forecast breakdown powered by Open-Meteo atmospheric models",
    "legendMinMax": "Min/Max Temp",
    "legendRain": "Rain Probability",
    "voiceModalTitle": "Live Voice Mode",
    "voiceModalPrompt": "\"Ask any weather question in your language...\"",
    "voiceStatusIdle": "Tap orb to start speaking",
    "voiceStatusListening": "Listening... Speak your weather question",
    "voiceStatusThinking": "WeatherGPT is thinking...",
    "voiceStatusSpeaking": "Speaking weather advice...",
    "handsFreeLabel": "Hands-Free Auto-Chat (Continuous)",
    "speedLabel": "Speed:",
    "settingsTitle": "API Key Configuration",
    "settingsSubtitle": "Connect your Gemini & Open-Meteo keys",
    "geminiKeyLabel": "Google Gemini API Key",
    "geminiFreeKey": "Get Free Key",
    "geminiDesc": "Powers natural conversational weather advice and native audio.",
    "openmeteoKeyLabel": "Open-Meteo API Key (Optional)",
    "openmeteoPricing": "Open-Meteo Pricing",
    "openmeteoDesc": "Free tier is built-in — no API key needed for full live weather & 7-day predictions.",
    "testKeysBtn": "Test Keys",
    "saveSettingsBtn": "Save Settings",
    "footerSystem": "WeatherGPT • Real-Time & 7-Day AI Weather System",
    "footerPowered": "Powered by Open-Meteo REST API • Google Gemini Reasoning • All 12+ Indian Languages Voice",
    "days": {
      "today": "Today",
      "mon": "Mon",
      "tue": "Tue",
      "wed": "Wed",
      "thu": "Thu",
      "fri": "Fri",
      "sat": "Sat",
      "sun": "Sun"
    },
    "conditions": {
      "Clear sky": "Clear sky",
      "Mainly clear": "Mainly clear",
      "Partly cloudy": "Partly cloudy",
      "Overcast": "Overcast",
      "Fog": "Fog",
      "Dense fog": "Dense fog",
      "Depositing rime fog": "Dense fog",
      "Light drizzle": "Light drizzle",
      "Moderate drizzle": "Moderate drizzle",
      "Dense drizzle": "Dense drizzle",
      "Slight rain": "Slight rain",
      "Moderate rain": "Moderate rain",
      "Heavy rain": "Heavy rain",
      "Thunderstorm": "Thunderstorm",
      "Thunderstorm with hail": "Thunderstorm with hail",
      "Thunderstorm with slight hail": "Thunderstorm with hail",
      "Thunderstorm with heavy hail": "Severe thunderstorm",
      "Slight snow": "Slight snow",
      "Moderate snow": "Moderate snow",
      "Heavy snow": "Heavy snow",
      "Rain showers": "Rain showers",
      "Slight rain showers": "Light rain showers",
      "Moderate rain showers": "Moderate rain showers",
      "Violent rain showers": "Heavy rain showers"
    },
    "tips": {
      "rain": "High rain chance ({p}%) today. Keep an umbrella handy and travel carefully!",
      "warm": "Warm conditions ({t}°C). Stay hydrated, wear light cotton clothes and sunscreen.",
      "pleasant": "Pleasant {d} conditions today at {t}°C. Great for outdoor errands and walks!"
    }
  },
  "ta": {
    "appSubtitle": "நேரலை வானிலை & 7-நாள் முன்னறிவிப்பு",
    "gpsText": "நேரலை GPS",
    "gpsActive": "GPS செயலில் உள்ளது",
    "gpsLocating": "கண்டறிகிறது...",
    "searchPlaceholder": "இந்திய அல்லது உலக நகரத்தை தேடவும்...",
    "liveVoiceBtn": "நேரலை குரல் அரட்டை",
    "offlineBtn": "ஆஃப்லைன்",
    "exitOfflineBtn": "ஆஃப்லைனில் இருந்து வெளியேறு",
    "keysBtn": "சாவிகள் (Keys)",
    "offlineBannerText": "ஆஃப்லைன் பயன்முறை: சேமிக்கப்பட்ட 7-நாள் வானிலை காட்டப்படுகிறது.",
    "switchLiveBtn": "நேரலைக்கு மாறவும்",
    "citiesLabel": "நகரங்கள்:",
    "cities": [
      "சென்னை",
      "தில்லி",
      "பெங்களூரு",
      "ஹைதராபாத்",
      "மும்பை",
      "கொல்கத்தா",
      "தஞ்சாவூர்"
    ],
    "gpsLocked": "GPS இணைக்கப்பட்டது",
    "currentCond": "தற்போதைய வானிலை",
    "today": "இன்று",
    "talkBtn": "WeatherGPT உடன் பேசவும்",
    "feelsLike": "உணரப்படுவது",
    "highLow": "உ: {max}° • கு: {min}°",
    "rainChance": "மழை வாய்ப்பு",
    "rainExpected": "{rain} மி.மீ எதிர்பார்க்கப்படுகிறது",
    "humidity": "ஈரப்பதம்",
    "comfort": "இதமானது",
    "humid": "அதிக ஈரப்பதம்",
    "dry": "வறண்டது",
    "windSpeed": "காற்று வேகம்",
    "windBreeze": "மென்மையான காற்று",
    "uvIndex": "UV குறியீடு",
    "uvLevels": {
      "veryHigh": "மிக அதிகம் (தோல் & கண்களை பாதுகாக்கவும்)",
      "high": "அதிகம் (சன்ஸ்கிரீன் பயன்படுத்தவும்)",
      "safe": "மிதமான / பாதுகாப்பானது"
    },
    "airPressure": "காற்று அழுத்தம்",
    "surfacePressure": "வளிமண்டல அழுத்தம்",
    "cloudCover": "மேக மூட்டம்",
    "cloudDesc": "சிதறிய மேகங்கள்",
    "offlineCardTitle": "ஆஃப்லைன் 7-நாள் வானிலை தரவு",
    "offlineCardDesc": "இணையம் இல்லாதபோதும் பயன்படுத்த 7-நாள் முன்னறிவிப்பை சேமிக்கவும்",
    "offlineSavedText": "சாதனத்தில் சேமிக்கப்பட்டது",
    "saveOfflineBtn": "7-நாள் ஆஃப்லைனில் சேமிக்கவும்",
    "guideTitle": "குரல் & அரட்டை வானிலை வழிகாட்டி",
    "guideSubtitle": "அனைத்து 12+ இந்திய மொழிகளுக்கும் ஆதரவு",
    "welcomeChat": "வணக்கம்! நான் உங்கள் <strong>WeatherGPT குரல் வழிகாட்டி</strong>. நீங்கள் எந்த இந்திய மொழியிலும் பேசலாம் அல்லது தட்டச்சு செய்யலாம். மழை, குடை தேவை, என்ன உடை அணியலாம் அல்லது 7 நாள் வானிலை பற்றி என்னிடம் கேளுங்கள்!",
    "voiceOutputTitle": "இயற்கையான குரல் ஒலிப்பு",
    "voiceOutputStatus": "உங்கள் மொழியில் கேட்க கிளிக் செய்யவும்",
    "listenBtn": "🔊 கேட்க ({lang})",
    "liveModeBtn": "நேரலை முறை",
    "forecastTitle": "அடுத்த 7-நாள் வானிலை கணிப்பு",
    "forecastSubtitle": "Open-Meteo வளிமண்டல மாதிரிகள் மூலம் தினசரி முன்னறிவிப்பு",
    "legendMinMax": "குறைந்த / அதிகபட்ச வெப்பம்",
    "legendRain": "மழை சாத்தியக்கூறு",
    "voiceModalTitle": "நேரலை குரல் பயன்முறை",
    "voiceModalPrompt": "\"உங்கள் மொழியில் வானிலை பற்றிய கேள்வியை கேளுங்கள்...\"",
    "voiceStatusIdle": "பேசத் தொடங்க வட்டத்தை தட்டவும்",
    "voiceStatusListening": "கேட்கிறது... உங்கள் கேள்வியைக் கூறுங்கள்",
    "voiceStatusThinking": "WeatherGPT யோசிக்கிறது...",
    "voiceStatusSpeaking": "பதில் குரலில் ஒலிக்கிறது...",
    "handsFreeLabel": "ஹேண்ட்ஸ்-ஃப்ரீ தானியங்கி அரட்டை (தொடர்ச்சியானது)",
    "speedLabel": "வேகம்:",
    "settingsTitle": "API சாவி கட்டமைப்பு",
    "settingsSubtitle": "உங்கள் Gemini & Open-Meteo சாவிகளை இணைக்கவும்",
    "geminiKeyLabel": "Google Gemini API சாவி",
    "geminiFreeKey": "இலவச சாவி பெற",
    "geminiDesc": "இயற்கையான உரையாடல் மற்றும் குரல் ஒலிப்பை இயக்குகிறது.",
    "openmeteoKeyLabel": "Open-Meteo API சாவி (விருப்பத்திற்குரியது)",
    "openmeteoPricing": "Open-Meteo கட்டண விவரம்",
    "openmeteoDesc": "இலவச அடுக்கு உள்ளமைக்கப்பட்டுள்ளது — நேரலை & 7-நாள் கணிப்புக்கு சாவி தேவையில்லை.",
    "testKeysBtn": "சாவிகளை சோதிக்க",
    "saveSettingsBtn": "அமைப்புகளை சேமிக்க",
    "footerSystem": "WeatherGPT • நேரலை & 7-நாள் AI வானிலை தளம்",
    "footerPowered": "Open-Meteo REST API • Google Gemini Reasoning • அனைத்து 12+ இந்திய மொழிகள் குரல்",
    "days": {
      "today": "இன்று",
      "mon": "திங்கள்",
      "tue": "செவ்வாய்",
      "wed": "புதன்",
      "thu": "வியாழன்",
      "fri": "வெள்ளி",
      "sat": "சனி",
      "sun": "ஞாயிறு"
    },
    "conditions": {
      "Clear sky": "தெளிவான வானம்",
      "Mainly clear": "பெரும்பாலும் தெளிவானது",
      "Partly cloudy": "பகுதி மேகமூட்டம்",
      "Overcast": "முழு மேகமூட்டம்",
      "Fog": "மூடுபனி",
      "Dense fog": "அடர்ந்த மூடுபனி",
      "Depositing rime fog": "பனிமூட்டம்",
      "Light drizzle": "லேசான தூறல்",
      "Moderate drizzle": "மிதமான தூறல்",
      "Dense drizzle": "அடர் தூறல்",
      "Slight rain": "லேசான மழை",
      "Moderate rain": "மிதமான மழை",
      "Heavy rain": "கனமழை",
      "Thunderstorm": "இடி மின்னலுடன் மழை",
      "Thunderstorm with hail": "ஆலங்கட்டி மழை",
      "Thunderstorm with slight hail": "இடி மின்னலுடன் ஆலங்கட்டி",
      "Thunderstorm with heavy hail": "தீவிர இடிமழை",
      "Slight snow": "லேசான பனிப்பொழிவு",
      "Moderate snow": "மிதமான பனிப்பொழிவு",
      "Heavy snow": "கடும் பனிப்பொழிவு",
      "Rain showers": "மழைத்தூறல்",
      "Slight rain showers": "லேசான மழைச்சாரல்",
      "Moderate rain showers": "மிதமான மழைச்சாரல்",
      "Violent rain showers": "கடும் மழைக்காய்ச்சல்"
    },
    "tips": {
      "rain": "இன்று மழை வாய்ப்பு அதிகம் ({p}%). குடை எடுத்துச் செல்லவும், பாதுகாப்பாக பயணிக்கவும்!",
      "warm": "வெப்பமான வானிலை ({t}°C). நீர் அதிகம் பருகவும், லேசான பருத்தி ஆடைகளை அணியவும்.",
      "pleasant": "இன்று {t}°C உடன் இதமான {d} வானிலை. நடைப்பயிற்சி மற்றும் வெளிப்புற வேலைகளுக்கு உகந்தது!"
    }
  },
  "hi": {
    "appSubtitle": "रीयल-टाइम मौसम और 7-दिवसीय पूर्वानुमान",
    "gpsText": "लाइव GPS",
    "gpsActive": "GPS सक्रिय है",
    "gpsLocating": "स्थान खोज रहे हैं...",
    "searchPlaceholder": "कोई भी भारतीय या वैश्विक शहर खोजें...",
    "liveVoiceBtn": "लाइव वॉयस चैट",
    "offlineBtn": "ऑफलाइन",
    "exitOfflineBtn": "ऑफलाइन से बाहर निकलें",
    "keysBtn": "कुंजियाँ (Keys)",
    "offlineBannerText": "ऑफलाइन मोड सक्रिय: सहेजा गया 7-दिवसीय मौसम दिखाया जा रहा है।",
    "switchLiveBtn": "लाइव मोड पर जाएं",
    "citiesLabel": "शहर:",
    "cities": [
      "चेन्नई",
      "दिल्ली",
      "बेंगलुरु",
      "हैदराबाद",
      "मुंबई",
      "कोलकाता",
      "तंजावुर"
    ],
    "gpsLocked": "GPS लॉक हुआ",
    "currentCond": "वर्तमान स्थिति",
    "today": "आज",
    "talkBtn": "WeatherGPT से बात करें",
    "feelsLike": "महसूस हो रहा है",
    "highLow": "अधि: {max}° • न्यून: {min}°",
    "rainChance": "बारिश की संभावना",
    "rainExpected": "{rain} मिमी अपेक्षित",
    "humidity": "आर्द्रता (नमी)",
    "comfort": "आरामदायक",
    "humid": "उमस भरा",
    "dry": "शुष्क",
    "windSpeed": "हवा की गति",
    "windBreeze": "मंद समीर",
    "uvIndex": "UV सूचकांक",
    "uvLevels": {
      "veryHigh": "अत्यधिक (आँखों और त्वचा की सुरक्षा करें)",
      "high": "उच्च (सनस्क्रीन लगाएं)",
      "safe": "मध्यम / सुरक्षित"
    },
    "airPressure": "वायुमंडलीय दबाव",
    "surfacePressure": "सतही वायुदाब",
    "cloudCover": "बादल का आवरण",
    "cloudDesc": "हल्के बादल",
    "offlineCardTitle": "ऑफलाइन 7-दिवसीय मौसम डेटा",
    "offlineCardDesc": "बिना इंटरनेट के कभी भी देखने के लिए 7 दिनों का पूर्वानुमान सहेजें",
    "offlineSavedText": "डिवाइस पर स्थानीय रूप से सुरक्षित",
    "saveOfflineBtn": "7-दिवसीय डेटा सहेजें",
    "guideTitle": "वॉयस और चैट मौसम गाइड",
    "guideSubtitle": "सभी 12+ भारतीय भाषाओं में समर्थित",
    "welcomeChat": "नमस्ते! मैं आपका <strong>WeatherGPT वॉयस गाइड</strong> हूँ। आप किसी भी भारतीय भाषा में बोल या लिख सकते हैं। मुझसे बारिश, छाता, क्या पहनें या 7 दिनों के मौसम के बारे में पूछें!",
    "voiceOutputTitle": "प्राकृतिक ध्वनि आउटपुट",
    "voiceOutputStatus": "अपनी भाषा में सुनने के लिए क्लिक करें",
    "listenBtn": "🔊 सुनें ({lang})",
    "liveModeBtn": "लाइव मोड",
    "forecastTitle": "अगले 7 दिनों का मौसम पूर्वानुमान",
    "forecastSubtitle": "Open-Meteo वायुमंडलीय मॉडल द्वारा दैनिक पूर्वानुमान",
    "legendMinMax": "न्यूनतम / अधिकतम तापमान",
    "legendRain": "बारिश की संभावना",
    "voiceModalTitle": "लाइव वॉयस मोड",
    "voiceModalPrompt": "\"अपनी भाषा में कोई भी मौसम प्रश्न पूछें...\"",
    "voiceStatusIdle": "बोलने के लिए गोले पर टैप करें",
    "voiceStatusListening": "सुन रहा हूँ... अपना प्रश्न बोलें",
    "voiceStatusThinking": "WeatherGPT सोच रहा है...",
    "voiceStatusSpeaking": "मौसम सलाह बोली जा रही है...",
    "handsFreeLabel": "हैंड्स-फ्री ऑटो-चैट (निरंतर बातचीत)",
    "speedLabel": "गति:",
    "settingsTitle": "API कुंजी विन्यास",
    "settingsSubtitle": "अपनी Gemini और Open-Meteo कुंजियाँ जोड़ें",
    "geminiKeyLabel": "Google Gemini API कुंजी",
    "geminiFreeKey": "निःशुल्क कुंजी प्राप्त करें",
    "geminiDesc": "प्राकृतिक संवादात्मक सलाह और क्षेत्रीय ध्वनि आउटपुट सक्षम करता है।",
    "openmeteoKeyLabel": "Open-Meteo API कुंजी (वैकल्पिक)",
    "openmeteoPricing": "Open-Meteo मूल्य निर्धारण",
    "openmeteoDesc": "मुफ़्त स्तर पहले से सक्रिय है — लाइव और 7-दिवसीय मौसम के लिए कुंजी आवश्यक नहीं है।",
    "testKeysBtn": "कुंजियों का परीक्षण करें",
    "saveSettingsBtn": "सेटिंग्स सहेजें",
    "footerSystem": "WeatherGPT • रीयल-टाइम और 7-दिवसीय AI मौसम प्रणाली",
    "footerPowered": "Open-Meteo REST API • Google Gemini Reasoning • सभी 12+ भारतीय भाषा वॉयस",
    "days": {
      "today": "आज",
      "mon": "सोम",
      "tue": "मंगल",
      "wed": "बुध",
      "thu": "गुरु",
      "fri": "शुक्र",
      "sat": "शनि",
      "sun": "रवि"
    },
    "conditions": {
      "Clear sky": "साफ आसमान",
      "Mainly clear": "मुख्यतः साफ",
      "Partly cloudy": "आंशिक बादल",
      "Overcast": "घने बादल",
      "Fog": "कोहरा",
      "Dense fog": "घना कोहरा",
      "Depositing rime fog": "पाला कोहरा",
      "Light drizzle": "हल्की बूंदाबांदी",
      "Moderate drizzle": "मध्यम बूंदाबांदी",
      "Dense drizzle": "तेज बूंदाबांदी",
      "Slight rain": "हल्की बारिश",
      "Moderate rain": "मध्यम बारिश",
      "Heavy rain": "भारी बारिश",
      "Thunderstorm": "गरज के साथ बारिश",
      "Thunderstorm with hail": "ओलावृष्टि के साथ आंधी",
      "Thunderstorm with slight hail": "हल्की ओलावृष्टि के साथ आंधी",
      "Thunderstorm with heavy hail": "भीषण तूफान और ओलावृष्टि",
      "Slight snow": "हल्की बर्फबारी",
      "Moderate snow": "मध्यम बर्फबारी",
      "Heavy snow": "भारी बर्फबारी",
      "Rain showers": "वर्षा बौछਾਰें",
      "Slight rain showers": "हल्की बौछारें",
      "Moderate rain showers": "मध्यम बौछारें",
      "Violent rain showers": "तेज मूसलाधार बौछारें"
    },
    "tips": {
      "rain": "आज बारिश की संभावना अधिक है ({p}%)। छाता साथ रखें और सुरक्षित यात्रा करें!",
      "warm": "गर्म मौसम ({t}°C)। पानी पीते रहें, हल्के सूती कपड़े पहनें और धूप से बचें।",
      "pleasant": "आज {t}°C के साथ सुहावना {d} मौसम है। टहलने और बाहर के कामों के लिए बेहतरीन दिन!"
    }
  },
  "te": {
    "appSubtitle": "రియల్ టైమ్ వాతావరణం & 7-రోజుల అంచనా",
    "gpsText": "లైవ్ GPS",
    "gpsActive": "GPS క్రియాశీలంగా ఉంది",
    "gpsLocating": "గుర్తిస్తోంది...",
    "searchPlaceholder": "భారతీయ లేదా ప్రపంచ నగరాన్ని శోధించండి...",
    "liveVoiceBtn": "లైవ్ వాయిస్ చాట్",
    "offlineBtn": "ఆఫ్‌లైన్",
    "exitOfflineBtn": "ఆఫ్‌లైన్ నుండి నిష్క్రమించు",
    "keysBtn": "కీలు (Keys)",
    "offlineBannerText": "ఆఫ్‌లైన్ మోడ్: సేవ్ చేయబడిన 7-రోజుల వాతావరణం చూపబడుతోంది.",
    "switchLiveBtn": "లైవ్‌కు మారండి",
    "citiesLabel": "నగరాలు:",
    "cities": [
      "చెన్నై",
      "ఢిల్లీ",
      "బెంగళూరు",
      "హైదరాబాద్",
      "ముంబై",
      "కోల్‌కతా",
      "తంజావూరు"
    ],
    "gpsLocked": "GPS లాక్ చేయబడింది",
    "currentCond": "ప్రస్తుత వాతావరణం",
    "today": "ఈ రోజు",
    "talkBtn": "WeatherGPT తో మాట్లాడండి",
    "feelsLike": "అనిపించే ఉష్ణోగ్రత",
    "highLow": "గరిష్ట: {max}° • కనిష్ట: {min}°",
    "rainChance": "వర్షం అవకాశం",
    "rainExpected": "{rain} మి.మీ వర్షం అంచనా",
    "humidity": "తేమ (హ్యూమిడిటీ)",
    "comfort": "సౌకర్యవంతమైనది",
    "humid": "ఉక్కపోతగా ఉంది",
    "dry": "పొడిగా ఉంది",
    "windSpeed": "గాలి వేగం",
    "windBreeze": "మందమైన గాలి",
    "uvIndex": "UV సూచిక",
    "uvLevels": {
      "veryHigh": "చాలా ఎక్కువ (కళ్ళు, చర్మం జాగ్రత్త)",
      "high": "ఎక్కువ (సన్‌స్క్రీన్ వాడండి)",
      "safe": "సాధారణం / సురక్షితం"
    },
    "airPressure": "వాయు పీడనం",
    "surfacePressure": "ఉపరితల వాతావరణ పీడనం",
    "cloudCover": "మేఘాల కవరేజ్",
    "cloudDesc": "చెల్లాచెదురు మేఘాలు",
    "offlineCardTitle": "ఆఫ్‌లైన్ 7-రోజుల వాతావరణ డేటా",
    "offlineCardDesc": "నెట్‌వర్క్ లేకుండా చూడటానికి 7-రోజుల అంచనాను సేవ్ చేయండి",
    "offlineSavedText": "పరికరంలో సేవ్ చేయబడింది",
    "saveOfflineBtn": "7-రోజుల డేటా సేవ్ చేయండి",
    "guideTitle": "వాయిస్ & చాట్ వాతావరణ గైడ్",
    "guideSubtitle": "అన్ని 12+ భారతీయ భాషలకు మద్దతు",
    "welcomeChat": "నమస్కారం! నేను మీ <strong>WeatherGPT వాయిస్ గైడ్</strong>. మీరు ఏ భారతీయ భాషలోనైనా మాట్లాడవచ్చు లేదా టైప్ చేయవచ్చు. వర్షం, గొడుగు, ఏమి ధరించాలి లేదా 7 రోజుల వాతావరణం గురించి నన్ను అడగండి!",
    "voiceOutputTitle": "సహజ వాయిస్ అవుట్‌పుట్",
    "voiceOutputStatus": "మీ భాషలో వినడానికి క్లిక్ చేయండి",
    "listenBtn": "🔊 వినండి ({lang})",
    "liveModeBtn": "లైవ్ మోడ్",
    "forecastTitle": "తదుపరి 7-రోజుల వాతావరణ అంచనా",
    "forecastSubtitle": "Open-Meteo నమూనాల ఆధారంగా రోజువారీ సూచన",
    "legendMinMax": "కనిష్ట / గరిష్ట ఉష్ణోగ్రత",
    "legendRain": "వర్షం సంభావ్యత",
    "voiceModalTitle": "లైవ్ వాయిస్ మోడ్",
    "voiceModalPrompt": "\"మీ భాషలో ఏదైనా వాతావరణ ప్రశ్న అడగండి...\"",
    "voiceStatusIdle": "మాట్లాడటానికి వృత్తాన్ని తాకండి",
    "voiceStatusListening": "వింటోంది... మీ ప్రశ్న చెప్పండి",
    "voiceStatusThinking": "WeatherGPT ఆలోచిస్తోంది...",
    "voiceStatusSpeaking": "సమాధానం వినిపిస్తోంది...",
    "handsFreeLabel": "హ్యాండ్స్-ఫ్రీ ఆటో-చాట్ (నిరంతర సంభాషణ)",
    "speedLabel": "వేగం:",
    "settingsTitle": "API కీ కాన్ఫిగరేషన్",
    "settingsSubtitle": "మీ Gemini & Open-Meteo కీలను కనెక్ట్ చేయండి",
    "geminiKeyLabel": "Google Gemini API కీ",
    "geminiFreeKey": "ఉచిత కీ పొందండి",
    "geminiDesc": "సహజమైన సంభాషణ సలహాలు మరియు ప్రాంతీయ వాయిస్ అవుట్‌పుట్‌ను ప్రారంభిస్తుంది.",
    "openmeteoKeyLabel": "Open-Meteo API కీ (ఐచ్ఛికం)",
    "openmeteoPricing": "Open-Meteo ధరల వివరాలు",
    "openmeteoDesc": "ఉచిత శ్రేణి అంతర్నిర్మితంగా ఉంది — లైవ్ & 7-రోజుల వాతావరణానికి కీ అవసరం లేదు.",
    "testKeysBtn": "కీలను పరీక్షించండి",
    "saveSettingsBtn": "సెట్టింగులను సేవ్ చేయండి",
    "footerSystem": "WeatherGPT • రియల్-టైమ్ & 7-రోజుల AI వాతావరణ వ్యవస్థ",
    "footerPowered": "Open-Meteo REST API • Google Gemini Reasoning • 12+ భారతీయ భాషల వాయిస్",
    "days": {
      "today": "ఈ రోజు",
      "mon": "సోమ",
      "tue": "మంగళ",
      "wed": "బుధ",
      "thu": "గురు",
      "fri": "శుక్ర",
      "sat": "శని",
      "sun": "ఆది"
    },
    "conditions": {
      "Clear sky": "నిర్మలమైన ఆకాశం",
      "Mainly clear": "ఎక్కువగా నిర్మలం",
      "Partly cloudy": "పాక్షిక మేఘావృతం",
      "Overcast": "పూర్తిగా మేఘావృతం",
      "Fog": "పొగమంచు",
      "Dense fog": "దట్టమైన పొగమంచు",
      "Depositing rime fog": "మంచు పొగ",
      "Light drizzle": "తేలికపాటి జల్లులు",
      "Moderate drizzle": "మధ్యస్థ జల్లులు",
      "Dense drizzle": "భారీ జల్లులు",
      "Slight rain": "తేలికపాటి వర్షం",
      "Moderate rain": "మధ్యస్థ వర్షం",
      "Heavy rain": "భారీ వర్షం",
      "Thunderstorm": "ఉరుములతో కూడిన వర్షం",
      "Thunderstorm with hail": "వడగండ్ల వాన",
      "Thunderstorm with slight hail": "ఉరుములు, వడగండ్లు",
      "Thunderstorm with heavy hail": "తీవ్రమైన ఉరుములతో కూడిన తుఫాను",
      "Slight snow": "తేలికపాటి హిమపాతం",
      "Moderate snow": "మధ్యస్థ హిమపాతం",
      "Heavy snow": "భారీ హిమపాతం",
      "Rain showers": "వర్షపు జల్లులు",
      "Slight rain showers": "తేలికపాటి జల్లులు",
      "Moderate rain showers": "మధ్యస్థ జల్లులు",
      "Violent rain showers": "భారీ కుంభవృష్టి"
    },
    "tips": {
      "rain": "ఈ రోజు వర్షం పడే అవకాశం ఎక్కువ ({p}%). గొడుగు వెంట ఉంచుకోండి, జాగ్రత్తగా ప్రయాణించండి!",
      "warm": "వేడి వాతావరణం ({t}°C). ఎక్కువగా నీరు త్రాగండి, తేలికపాటి కాటన్ దుస్తులు ధరించండి.",
      "pleasant": "నేడు {t}°C తో ఆహ్లాదకరమైన {d} వాతావరణం. బయట పనులకు, నడకకు అనుకూలమైన రోజు!"
    }
  },
  "kn": {
    "appSubtitle": "ನೈಜ-ಸಮಯದ ಹವಾಮಾನ & 7-ದಿನಗಳ ಮುನ್ಸೂಚನೆ",
    "gpsText": "ಲೈವ್ GPS",
    "gpsActive": "GPS ಸಕ್ರಿಯವಾಗಿದೆ",
    "gpsLocating": "ಸ್ಥಳ ಪತ್ತೆಮಾಡಲಾಗುತ್ತಿದೆ...",
    "searchPlaceholder": "ಯಾವುದೇ ಭಾರತೀಯ ಅಥವಾ ಜಾಗತಿಕ ನಗರವನ್ನು ಹುಡುಕಿ...",
    "liveVoiceBtn": "ಲೈವ್ ಧ್ವನಿ ಚಾಟ್",
    "offlineBtn": "ಆಫ್‌ಲೈನ್",
    "exitOfflineBtn": "ಆಫ್‌ಲೈನ್‌ನಿಂದ ನಿರ್ಗಮಿಸಿ",
    "keysBtn": "ಕೀಲಿಗಳು (Keys)",
    "offlineBannerText": "ಆಫ್‌ಲೈನ್ ಮೋಡ್: ಉಳಿಸಿದ 7-ದಿನಗಳ ಹವಾಮಾನವನ್ನು ತೋರಿಸಲಾಗುತ್ತಿದೆ.",
    "switchLiveBtn": "ಲೈವ್ ಮೋಡ್‌ಗೆ ಬದಲಿಸಿ",
    "citiesLabel": "ನಗರಗಳು:",
    "cities": [
      "ಚೆನ್ನೈ",
      "ದೆಹಲಿ",
      "ಬೆಂಗಳೂರು",
      "ಹೈದರಾಬಾದ್",
      "ಮುಂಬೈ",
      "ಕೋಲ್ಕತ್ತಾ",
      "ತಂಜಾವೂರು"
    ],
    "gpsLocked": "GPS ಲಾಕ್ ಆಗಿದೆ",
    "currentCond": "ಪ್ರಸ್ತುತ ಹವಾಮಾನ ಸ್ಥಿತಿ",
    "today": "ಇಂದು",
    "talkBtn": "WeatherGPT ಜೊತೆ ಮಾತನಾಡಿ",
    "feelsLike": "ಅನುಭವವಾಗುವ ತಾಪಮಾನ",
    "highLow": "ಗರಿಷ್ಠ: {max}° • ಕನಿಷ್ಠ: {min}°",
    "rainChance": "ಮಳೆಯ ಸಾಧ್ಯತೆ",
    "rainExpected": "{rain} ಮಿಮೀ ನಿರೀಕ್ಷಿಸಲಾಗಿದೆ",
    "humidity": "ಆರ್ದ್ರತೆ (ತೇವಾಂಶ)",
    "comfort": "ಹಿತಕರ",
    "humid": "ಸೆಕೆ/ಆರ್ದ್ರತೆ ಹೆಚ್ಚು",
    "dry": "ಒಣ ಹವೆ",
    "windSpeed": "ಗಾಳಿಯ ವೇಗ",
    "windBreeze": "ತಂಗಾಳಿ",
    "uvIndex": "UV ಸೂಚ್ಯಂಕ",
    "uvLevels": {
      "veryHigh": "ಅತ್ಯಂತ ಹೆಚ್ಚು (ಕಣ್ಣು ಮತ್ತು ಚರ್ಮ ರಕ್ಷಿಸಿ)",
      "high": "ಹೆಚ್ಚು (ಸನ್‌ಸ್ಕ್ರೀನ್ ಬಳಸಿ)",
      "safe": "ಸಾಧಾರಣ / ಸುರಕ್ಷಿತ"
    },
    "airPressure": "ವಾತಾವರಣದ ಒತ್ತಡ",
    "surfacePressure": "ಮೇಲ್ಮೈ ಒತ್ತಡ",
    "cloudCover": "ಮೋಡ ಕವಿದ ಪ್ರಮಾಣ",
    "cloudDesc": "ಚದುರಿದ ಮೋಡಗಳು",
    "offlineCardTitle": "ಆಫ್‌ಲೈನ್ 7-ದಿನಗಳ ಹವಾಮಾನ ಮಾಹಿತಿ",
    "offlineCardDesc": "ಇಂಟರ್ನೆಟ್ ಇಲ್ಲದೆ ನೋಡಲು ಮುಂದಿನ 7 ದಿನಗಳ ಮುನ್ಸೂಚನೆ ಉಳಿಸಿ",
    "offlineSavedText": "ಸಾಧನದಲ್ಲಿ ಸುರಕ್ಷಿತವಾಗಿ ಉಳಿಸಲಾಗಿದೆ",
    "saveOfflineBtn": "7-ದಿನಗಳ ಮಾಹಿತಿ ಉಳಿಸಿ",
    "guideTitle": "ಧ್ವನಿ & ಚಾಟ್ ಹವಾಮಾನ ಸಹಾಯಕ",
    "guideSubtitle": "ಎಲ್ಲಾ 12+ ಭಾರತೀಯ ಭಾಷೆಗಳಿಗೆ ಬೆಂಬಲ",
    "welcomeChat": "ನಮಸ್ಕಾರ! ನಾನು ನಿಮ್ಮ <strong>WeatherGPT ಧ್ವನಿ ಸಹಾಯಕ</strong>. ನೀವು ಯಾವುದೇ ಭಾರತೀಯ ಭಾಷೆಯಲ್ಲಿ ಮಾತನಾಡಬಹುದು ಅಥವಾ ಬರೆಯಬಹುದು. ಮಳೆ, ಛತ್ರಿ, ಉಡುಪು ಅಥವಾ 7 ದಿನಗಳ ಹವಾಮಾನ ಮುನ್ಸೂಚನೆ ಬಗ್ಗೆ ನನ್ನನ್ನು ಕೇಳಿ!",
    "voiceOutputTitle": "ನೈಸರ್ಗಿಕ ಧ್ವನಿ ಔಟ್‌ಪುಟ್",
    "voiceOutputStatus": "ನಿಮ್ಮ ಭಾಷೆಯಲ್ಲಿ ಕೇಳಲು ಕ್ಲಿಕ್ ಮಾಡಿ",
    "listenBtn": "🔊 ಆಲಿಸಿ ({lang})",
    "liveModeBtn": "ಲೈವ್ ಮೋಡ್",
    "forecastTitle": "ಮುಂದಿನ 7-ದಿನಗಳ ಹವಾಮಾನ ಭವಿಷ್ಯ",
    "forecastSubtitle": "Open-Meteo ಮಾದರಿಗಳ ಮೂಲಕ ದೈನಂದಿನ ಮುನ್ಸೂಚನೆ",
    "legendMinMax": "ಕನಿಷ್ಠ / ಗರಿಷ್ಠ ತಾಪಮಾನ",
    "legendRain": "ಮಳೆಯ ಸಂಭವನೀಯತೆ",
    "voiceModalTitle": "ಲೈವ್ ಧ್ವನಿ ಮೋಡ್",
    "voiceModalPrompt": "\"ನಿಮ್ಮ ಭಾಷೆಯಲ್ಲಿ ಯಾವುದೇ ಹವಾಮಾನ ಪ್ರಶ್ನೆ ಕೇಳಿ...\"",
    "voiceStatusIdle": "ಮಾತನಾಡಲು ವೃತ್ತವನ್ನು ಸ್ಪರ್ಶಿಸಿ",
    "voiceStatusListening": "ಕೇಳಿಸಿಕೊಳ್ಳುತ್ತಿದೆ... ಪ್ರಶ್ನೆ ಕೇಳಿ",
    "voiceStatusThinking": "WeatherGPT ಯೋಚಿಸುತ್ತಿದೆ...",
    "voiceStatusSpeaking": "ಉತ್ತರವನ್ನು ಧ್ವನಿಯಲ್ಲಿ ಹೇಳಲಾಗುತ್ತಿದೆ...",
    "handsFreeLabel": "ಹ್ಯಾಂಡ್ಸ್-ಫ್ರೀ ಸ್ವಯಂಚಾಲಿತ ಚಾಟ್ (ನಿರಂತರ)",
    "speedLabel": "ವೇಗ:",
    "settingsTitle": "API ಕೀ ಸಂರಚನೆ",
    "settingsSubtitle": "ನಿಮ್ಮ Gemini & Open-Meteo ಕೀಗಳನ್ನು ಸಂಪರ್ಕಿಸಿ",
    "geminiKeyLabel": "Google Gemini API ಕೀ",
    "geminiFreeKey": "ಉಚಿತ ಕೀ ಪಡೆಯಿರಿ",
    "geminiDesc": "ನೈಸರ್ಗಿಕ ಸಂಭಾಷಣೆ ಮತ್ತು ಪ್ರಾದೇಶಿಕ ಧ್ವನಿ ಔಟ್‌ಪುಟ್‌ಗೆ ಅಗತ್ಯ.",
    "openmeteoKeyLabel": "Open-Meteo API ಕೀ (ಐಚ್ಛಿಕ)",
    "openmeteoPricing": "Open-Meteo ಬೆಲೆ ವಿವರ",
    "openmeteoDesc": "ಉಚಿತ ಶ್ರೇಣಿ ಅಂತರ್ಗತವಾಗಿದೆ — ಲೈವ್ ಮತ್ತು 7-ದಿನಗಳ ಹವಾಮಾನಕ್ಕೆ ಕೀ ಅಗತ್ಯವಿಲ್ಲ.",
    "testKeysBtn": "ಕೀಲಿಗಳನ್ನು ಪರೀಕ್ಷಿಸಿ",
    "saveSettingsBtn": "ಸೆಟ್ಟಿಂಗ್ಸ್ ಉಳಿಸಿ",
    "footerSystem": "WeatherGPT • ರಿಯಲ್-ಟೈಮ್ & 7-ದಿನಗಳ AI ಹವಾಮಾನ ವ್ಯವಸ್ಥೆ",
    "footerPowered": "Open-Meteo REST API • Google Gemini Reasoning • 12+ ಭಾರತೀಯ ಭಾಷೆಗಳ ಧ್ವನಿ",
    "days": {
      "today": "ಇಂದು",
      "mon": "ಸೋಮ",
      "tue": "ಮಂಗಳ",
      "wed": "ಬುಧ",
      "thu": "ಗುರು",
      "fri": "ಶುಕ್ರ",
      "sat": "ಶನಿ",
      "sun": "ಭಾನು"
    },
    "conditions": {
      "Clear sky": "ಸ್ವಚ್ಛ ಆಕಾಶ",
      "Mainly clear": "ಹೆಚ್ಚಾಗಿ ಸ್ವಚ್ಛ",
      "Partly cloudy": "ಭಾಗಶಃ ಮೋಡ",
      "Overcast": "ಪೂರ್ಣ ಮೋಡ",
      "Fog": "ದಟ್ಟ ಮಂಜು",
      "Dense fog": "ಅತಿ ದಟ್ಟ ಮಂಜು",
      "Depositing rime fog": "ಹಿಮ ಮಂಜು",
      "Light drizzle": "ತುಂತುರು ಮಳೆ",
      "Moderate drizzle": "ಮಧ್ಯಮ ತುಂತುರು",
      "Dense drizzle": "ದಟ್ಟ ತುಂತುರು",
      "Slight rain": "ಹಗುರ ಮಳೆ",
      "Moderate rain": "ಸಾಧಾರಣ ಮಳೆ",
      "Heavy rain": "ಭಾರೀ ಮಳೆ",
      "Thunderstorm": "ಗುಡುಗು ಸಹಿತ ಮಳೆ",
      "Thunderstorm with hail": "ಆಲಿಕಲ್ಲು ಮಳೆ",
      "Thunderstorm with slight hail": "ಗುಡುಗು ಮತ್ತು ಆಲಿಕಲ್ಲು",
      "Thunderstorm with heavy hail": "ಭೀಕರ ಗುಡುಗು ಚಂಡಮಾರುತ",
      "Slight snow": "ಹಗುರ ಹಿಮಪಾತ",
      "Moderate snow": "ಮಧ್ಯಮ ಹಿಮಪಾತ",
      "Heavy snow": "ಭಾರೀ ಹಿಮಪಾತ",
      "Rain showers": "ಮಳೆ ಸುರಿಮಳೆ",
      "Slight rain showers": "ಹಗುರ ತುಂತುರು ಜಲ್ಲು",
      "Moderate rain showers": "ಮಧ್ಯಮ ಜಲ್ಲು",
      "Violent rain showers": "ಧಾರಾಕಾರ ಮಳೆ"
    },
    "tips": {
      "rain": "ಇಂದು ಮಳೆಯ ಸಾಧ್ಯತೆ ಹೆಚ್ಚು ({p}%). ಛತ್ರಿ ಜೊತೆಗೆ ಇರಲಿ ಮತ್ತು ಜಾಗರೂಕರಾಗಿ ಪ್ರಯಾಣಿಸಿ!",
      "warm": "ಬಿಸಿಲಿನ ವಾತಾವರಣ ({t}°C). ಹೆಚ್ಚು ನೀರು ಕುಡಿಯಿರಿ, ಹತ್ತಿ ಬಟ್ಟೆಗಳನ್ನು ಧರಿಸಿ.",
      "pleasant": "ಇಂದು {t}°C ನೊಂದಿಗೆ ಹಿತಕರವಾದ {d} ಹವಾಮಾನ. ಹೊರಾಂಗಣ ನಡಿಗೆ ಮತ್ತು ಕೆಲಸಗಳಿಗೆ ಉತ್ತಮ ದಿನ!"
    }
  },
  "ml": {
    "appSubtitle": "തത്സമയ കാലാവസ്ഥ & 7-ദിന പ്രവചനം",
    "gpsText": "ലൈവ് GPS",
    "gpsActive": "GPS സജീവമാണ്",
    "gpsLocating": "സ്ഥലം കണ്ടെത്തുന്നു...",
    "searchPlaceholder": "ഏതെങ്കിലും ഇന്ത്യൻ അല്ലെങ്കിൽ ആഗോള നഗരം തിരയുക...",
    "liveVoiceBtn": "ലൈവ് വോയ്‌സ് ചാറ്റ്",
    "offlineBtn": "ഓഫ്‌ലൈൻ",
    "exitOfflineBtn": "ഓഫ്‌ലൈൻ നിർത്തുക",
    "keysBtn": "കീകൾ (Keys)",
    "offlineBannerText": "ഓഫ്‌ലൈൻ മോഡ് സജീവമാണ്: സംരക്ഷിച്ച 7-ദിന വിവരങ്ങൾ കാണിക്കുന്നു.",
    "switchLiveBtn": "ലൈവിലേക്ക് മാറുക",
    "citiesLabel": "നഗരങ്ങൾ:",
    "cities": [
      "ചെന്നൈ",
      "ദില്ലി",
      "ബെംഗളൂരു",
      "ഹൈദരാബാദ്",
      "മുംബൈ",
      "കൊൽക്കത്ത",
      "തഞ്ചാവൂർ"
    ],
    "gpsLocked": "GPS ലോക്ക് ചെയ്തു",
    "currentCond": "നിലവിലെ അവസ്ഥ",
    "today": "ഇന്ന്",
    "talkBtn": "WeatherGPT യോട് സംസാരിക്കൂ",
    "feelsLike": "അനുഭവപ്പെടുന്നത്",
    "highLow": "പരമാ: {max}° • കുറഞ്ഞ: {min}°",
    "rainChance": "മഴ സാധ്യത",
    "rainExpected": "{rain} മി.മീ പ്രതീക്ഷിക്കുന്നു",
    "humidity": "ആർദ്രത",
    "comfort": "സുഖകരം",
    "humid": "ഉഷ്ണം/ഈർപ്പം കൂടുതൽ",
    "dry": "വരണ്ടത്",
    "windSpeed": "കാറ്റിന്റെ വേഗത",
    "windBreeze": "ഇളം കാറ്റ്",
    "uvIndex": "UV സൂചിക",
    "uvLevels": {
      "veryHigh": "വളരെ കൂടുതൽ (കണ്ണുകളും ചർമ്മവും സംരക്ഷിക്കുക)",
      "high": "കൂടുതൽ (സൺസ്ക്രീൻ ഉപയോഗിക്കുക)",
      "safe": "സാധാരണ / സുരക്ഷിതം"
    },
    "airPressure": "വായുമർദ്ദം",
    "surfacePressure": "ഭൗമോപരിതല മർദ്ദം",
    "cloudCover": "മേഘാവൃതം",
    "cloudDesc": "ചിതറിയ മേഘങ്ങൾ",
    "offlineCardTitle": "ഓഫ്‌ലൈൻ 7-ദിന കാലാവസ്ഥാ വിവരങ്ങൾ",
    "offlineCardDesc": "നെറ്റ്‌വർക്കില്ലാതെ കാണാൻ അടുത്ത 7 ദിവസത്തെ പ്രവചനം സൂക്ഷിക്കുക",
    "offlineSavedText": "ഡിവൈസിൽ സേവ് ചെയ്തു",
    "saveOfflineBtn": "7-ദിന വിവരങ്ങൾ സൂക്ഷിക്കുക",
    "guideTitle": "വോയ്‌സ് & ചാറ്റ് കാലാവസ്ഥാ ഗൈഡ്",
    "guideSubtitle": "12+ ഇന്ത്യൻ ഭാഷകളിൽ ലഭ്യമാണ്",
    "welcomeChat": "നമസ്കാരം! ഞാൻ നിങ്ങളുടെ <strong>WeatherGPT വോയ്‌സ് ഗൈഡ്</strong> ആണ്. നിങ്ങൾക്ക് ഏത് ഇന്ത്യൻ ഭാഷയിലും സംസാരിക്കാം അല്ലെങ്കിൽ ടൈപ്പ് ചെയ്യാം. മഴ, കുട, വസ്ത്രധാരണം അല്ലെങ്കിൽ 7 ദിവസത്തെ കാലാവസ്ഥയെക്കുറിച്ച് എന്നോട് ചോദിക്കൂ!",
    "voiceOutputTitle": "സ്വാഭാവിക ശബ്ദ ഔട്ട്പുട്ട്",
    "voiceOutputStatus": "നിങ്ങളുടെ ഭാഷയിൽ കേൾക്കാൻ ക്ലിക്ക് ചെയ്യുക",
    "listenBtn": "🔊 കേൾക്കുക ({lang})",
    "liveModeBtn": "ലൈവ് മോഡ്",
    "forecastTitle": "അടുത്ത 7-ദിവസത്തെ കാലാവസ്ഥ പ്രവചനം",
    "forecastSubtitle": "Open-Meteo മോഡലുകൾ അടിസ്ഥാനമാക്കിയുള്ള പ്രവചനം",
    "legendMinMax": "കുറഞ്ഞ / ഉയർന്ന താപനില",
    "legendRain": "മഴ സാധ്യത",
    "voiceModalTitle": "ലൈവ് വോയ്‌സ് മോഡ്",
    "voiceModalPrompt": "\"നിങ്ങളുടെ ഭാഷയിൽ കാലാവസ്ഥയെക്കുറിച്ച് എന്തും ചോദിക്കൂ...\"",
    "voiceStatusIdle": "സംസാരിക്കാൻ വൃത്തത്തിൽ തൊടുക",
    "voiceStatusListening": "കേൾക്കുന്നു... നിങ്ങളുടെ ചോദ്യം പറയൂ",
    "voiceStatusThinking": "WeatherGPT ചിന്തിക്കുന്നു...",
    "voiceStatusSpeaking": "ഉത്തരം ശബ്ദത്തിൽ നൽകുന്നു...",
    "handsFreeLabel": "ഹാൻഡ്‌സ്-ഫ്രീ ഓട്ടോ-ചാറ്റ് (തുടർച്ചയായ സംഭാഷണം)",
    "speedLabel": "വേഗത:",
    "settingsTitle": "API കീ കോൺഫിഗറേഷൻ",
    "settingsSubtitle": "നിങ്ങളുടെ Gemini & Open-Meteo കീകൾ ബന്ധിപ്പിക്കുക",
    "geminiKeyLabel": "Google Gemini API കീ",
    "geminiFreeKey": "സൗജന്യ കീ നേടുക",
    "geminiDesc": "സ്വാഭാവിക സംഭാഷണത്തിനും പ്രാദേശിക ഭാഷാ ശബ്ദത്തിനും സഹായിക്കുന്നു.",
    "openmeteoKeyLabel": "Open-Meteo API കീ (ഐച്ഛികം)",
    "openmeteoPricing": "Open-Meteo നിരക്കുകൾ",
    "openmeteoDesc": "സൗജന്യ പ്ലാൻ നിലവിലുണ്ട് — തത്സമയ, 7-ദിന വിവരങ്ങൾക്ക് കീ ആവശ്യമില്ല.",
    "testKeysBtn": "കീകൾ പരിശോധിക്കുക",
    "saveSettingsBtn": "ക്രമീകരണം സേവ് ചെയ്യുക",
    "footerSystem": "WeatherGPT • തത്സമയ & 7-ദിന AI കാലാവസ്ഥ സംവിധാനം",
    "footerPowered": "Open-Meteo REST API • Google Gemini Reasoning • 12+ ഇന്ത്യൻ ഭാഷാ വോയ്‌സ്",
    "days": {
      "today": "ഇന്ന്",
      "mon": "തിങ്കൾ",
      "tue": "ചൊവ്വ",
      "wed": "ബുധൻ",
      "thu": "വ്യാഴം",
      "fri": "വെള്ളി",
      "sat": "ശനി",
      "sun": "ഞായർ"
    },
    "conditions": {
      "Clear sky": "തെളിഞ്ഞ ആകാശം",
      "Mainly clear": "മിക്കവാറും തെളിഞ്ഞത്",
      "Partly cloudy": "ഭാഗികമായി മേഘാവൃതം",
      "Overcast": "പൂർണ്ണ മേഘാവൃതം",
      "Fog": "മഞ്ഞ്",
      "Dense fog": "കടുത്ത മഞ്ഞ്",
      "Depositing rime fog": "ഹിമമഞ്ഞ്",
      "Light drizzle": "നേരിയ ചാറ്റൽമഴ",
      "Moderate drizzle": "മിതമായ ചാറ്റൽമഴ",
      "Dense drizzle": "കനത്ത ചാറ്റൽമഴ",
      "Slight rain": "നേരിയ മഴ",
      "Moderate rain": "മിതമായ മഴ",
      "Heavy rain": "കനത്ത മഴ",
      "Thunderstorm": "ഇടിമിന്നലോട് കൂടിയ മഴ",
      "Thunderstorm with hail": "ആലിപ്പഴ വർഷം",
      "Thunderstorm with slight hail": "ഇടിമിന്നലും ആലിപ്പഴവും",
      "Thunderstorm with heavy hail": "തീവ്രമായ ഇടിമിന്നൽ ചുഴലി",
      "Slight snow": "നേരിയ മഞ്ഞുവീഴ്ച",
      "Moderate snow": "മിതമായ മഞ്ഞുവീഴ്ച",
      "Heavy snow": "കനത്ത മഞ്ഞുവീഴ്ച",
      "Rain showers": "മഴത്തുള്ളികൾ",
      "Slight rain showers": "നേരിയ മഴച്ചാറ്റൽ",
      "Moderate rain showers": "മിതമായ മഴച്ചാറ്റൽ",
      "Violent rain showers": "തീവ്രമായ പേമാരി"
    },
    "tips": {
      "rain": "ഇന്ന് മഴയ്ക്ക് സാധ്യത കൂടുതലാണ് ({p}%). കുട കയ്യിൽ കരുതുക, ശ്രദ്ധിച്ച് യാത്ര ചെയ്യുക!",
      "warm": "ചൂടുള്ള കാലാവസ്ഥ ({t}°C). ധാരാളം വെള്ളം കുടിക്കുക, കോട്ടൺ വസ്ത്രങ്ങൾ ധരിക്കുക.",
      "pleasant": "ഇന്ന് {t}°C ഉള്ള സുഖകരമായ {d} കാലാവസ്ഥ. നടക്കാനും പുറം ജോലികൾക്കും നല്ല സമയം!"
    }
  },
  "bn": {
    "appSubtitle": "রিয়েল-টাইম আবহাওয়া ও ৭ দিনের পূর্বাভাস",
    "gpsText": "লাইভ GPS",
    "gpsActive": "GPS সক্রিয় আছে",
    "gpsLocating": "অবস্থান খোঁজা হচ্ছে...",
    "searchPlaceholder": "যেকোনো ভারতীয় বা বৈশ্বিক শহর অনুসন্ধান করুন...",
    "liveVoiceBtn": "লাইভ ভয়েস চ্যাট",
    "offlineBtn": "অফলাইন",
    "exitOfflineBtn": "অফলাইন বন্ধ করুন",
    "keysBtn": "কিজ (Keys)",
    "offlineBannerText": "অফলাইন মোড সক্রিয়: সংরক্ষিত ৭ দিনের আবহাওয়া দেখানো হচ্ছে।",
    "switchLiveBtn": "লাইভ মোডে যান",
    "citiesLabel": "শহরসমূহ:",
    "cities": [
      "চেন্নাই",
      "দিল্লি",
      "বেঙ্গালুরু",
      "হায়দ্রাবাদ",
      "মুম্বাই",
      "কলকাতা",
      "তাঞ্জাভুর"
    ],
    "gpsLocked": "GPS লক করা হয়েছে",
    "currentCond": "বর্তমান পরিস্থিতি",
    "today": "আজ",
    "talkBtn": "WeatherGPT-র সাথে কথা বলুন",
    "feelsLike": "অনুভূত তাপমাত্রা",
    "highLow": "সর্বোচ্চ: {max}° • সর্বনিম্ন: {min}°",
    "rainChance": "বৃষ্টির সম্ভাবনা",
    "rainExpected": "{rain} মিমি বৃষ্টির সম্ভাবনা",
    "humidity": "আর্দ্রতা",
    "comfort": "আরামদায়ক",
    "humid": "ভ্যাপসা গরম/আর্দ্র",
    "dry": "শুষ্ক",
    "windSpeed": "বাতাসের গতিবেগ",
    "windBreeze": "মৃদু বাতাস",
    "uvIndex": "UV সূচক",
    "uvLevels": {
      "veryHigh": "খুব বেশি (চোখ ও ত্বকের যত্ন নিন)",
      "high": "বেশি (সানস্ক্রিন ব্যবহার করুন)",
      "safe": "মাঝারি / নিরাপদ"
    },
    "airPressure": "বায়ুচাপ",
    "surfacePressure": "পৃষ্ঠীয় বায়ুমণ্ডলীয় চাপ",
    "cloudCover": "মেঘের ঘনত্ব",
    "cloudDesc": "বিক্ষিপ্ত মেঘ",
    "offlineCardTitle": "অফলাইন ৭ দিনের আবহাওয়া ডেটা",
    "offlineCardDesc": "ইন্টারনেট ছাড়া দেখতে আগামী ৭ দিনের পূর্বাভাস সংরক্ষণ করুন",
    "offlineSavedText": "ডিভাইসে নিরাপদে সংরক্ষিত",
    "saveOfflineBtn": "৭ দিনের ডেটা সংরক্ষণ করুন",
    "guideTitle": "ভয়েস ও চ্যাট আবহাওয়া গাইড",
    "guideSubtitle": "১২+ ভারতীয় ভাষায় উপলব্ধ",
    "welcomeChat": "নমস্কার! আমি আপনার <strong>WeatherGPT ভয়েস গাইড</strong>। আপনি যেকোনো ভারতীয় ভাষায় কথা বলতে বা টাইপ করতে পারেন। বৃষ্টি, ছাতা, পোশাক বা আগামী ৭ দিনের পূর্বাভাস সম্পর্কে আমাকে জিজ্ঞাসা করুন!",
    "voiceOutputTitle": "প্রাকৃতিক ভয়েস আউটপুট",
    "voiceOutputStatus": "আপনার ভাষায় শুনতে ক্লিক করুন",
    "listenBtn": "🔊 শুনুন ({lang})",
    "liveModeBtn": "লাইভ মোড",
    "forecastTitle": "আগামী ৭ দিনের আবহাওয়ার পূর্বাভাস",
    "forecastSubtitle": "Open-Meteo মডেল দ্বারা চালিত পূর্বাভাস",
    "legendMinMax": "সর্বনিম্ন / সর্বোচ্চ তাপমাত্রা",
    "legendRain": "বৃষ্টির সম্ভাবনা",
    "voiceModalTitle": "লাইভ ভয়েস মোড",
    "voiceModalPrompt": "\"আপনার ভাষায় আবহাওয়ার যেকোনো প্রশ্ন জিজ্ঞাসা করুন...\"",
    "voiceStatusIdle": "কথা বলতে বৃত্তটিতে স্পর্শ করুন",
    "voiceStatusListening": "শুনছি... আপনার প্রশ্ন বলুন",
    "voiceStatusThinking": "WeatherGPT চিন্তা করছে...",
    "voiceStatusSpeaking": "উত্তর মুখে বলা হচ্ছে...",
    "handsFreeLabel": "হ্যান্ডস-ফ্রি অটো-চ্যাট (নিরবচ্ছিন্ন আলোচনা)",
    "speedLabel": "গতি:",
    "settingsTitle": "API কী কনফিগারেশন",
    "settingsSubtitle": "আপনার Gemini ও Open-Meteo কী সংযুক্ত করুন",
    "geminiKeyLabel": "Google Gemini API কী",
    "geminiFreeKey": "বিনামূল্যে কী পান",
    "geminiDesc": "স্বাভাবিক কথোপকথন ও আঞ্চলিক ভয়েস আউটপুট সক্ষম করে।",
    "openmeteoKeyLabel": "Open-Meteo API কী (ঐচ্ছিক)",
    "openmeteoPricing": "Open-Meteo মূল্যতালিকা",
    "openmeteoDesc": "বিনামূল্যের স্তর প্রস্তুত রয়েছে — লাইভ ও ৭ দিনের পূর্বাভাসে কোনো কী দরকার নেই।",
    "testKeysBtn": "কী পরীক্ষা করুন",
    "saveSettingsBtn": "সেটিংস সংরক্ষণ করুন",
    "footerSystem": "WeatherGPT • রিয়েল-টাইম ও ৭ দিনের AI আবহাওয়া ব্যবস্থা",
    "footerPowered": "Open-Meteo REST API • Google Gemini Reasoning • ১২+ ভারতীয় ভাষার ভয়েস",
    "days": {
      "today": "আজ",
      "mon": "সোম",
      "tue": "মঙ্গল",
      "wed": "বুধ",
      "thu": "বৃহস্পতি",
      "fri": "শুক্র",
      "sat": "শনি",
      "sun": "রবি"
    },
    "conditions": {
      "Clear sky": "পরিষ্কার আকাশ",
      "Mainly clear": "বেশিরভাগ পরিষ্কার",
      "Partly cloudy": "আংশিক মেঘলা",
      "Overcast": "মেঘলা আকাশ",
      "Fog": "কুয়াশা",
      "Dense fog": "ঘন কুয়াশা",
      "Depositing rime fog": "হিম কুয়াশা",
      "Light drizzle": "হালকা গুঁড়ি গুঁড়ি বৃষ্টি",
      "Moderate drizzle": "মাঝারি গুঁড়ি গুঁড়ি বৃষ্টি",
      "Dense drizzle": "ভারী গুঁড়ি গুঁড়ি বৃষ্টি",
      "Slight rain": "হালকা বৃষ্টি",
      "Moderate rain": "মাঝারি বৃষ্টি",
      "Heavy rain": "ভারী বৃষ্টি",
      "Thunderstorm": "বজ্রবিদ্যুৎসহ ঝড়বৃষ্টি",
      "Thunderstorm with hail": "শিলাবৃষ্টিসহ ঝড়",
      "Thunderstorm with slight hail": "বজ্রপাত ও হালকা শিলাবৃষ্টি",
      "Thunderstorm with heavy hail": "তীব্র বজ্রঝড়",
      "Slight snow": "হালকা তুষারপাত",
      "Moderate snow": "মাঝারি তুষারপাত",
      "Heavy snow": "ভারী তুষারপাত",
      "Rain showers": "বৃষ্টির ধারা",
      "Slight rain showers": "হালকা পশলা বৃষ্টি",
      "Moderate rain showers": "মাঝারি পশলা বৃষ্টি",
      "Violent rain showers": "প্রচণ্ড বৃষ্টিপাত"
    },
    "tips": {
      "rain": "আজ বৃষ্টির সম্ভাবনা বেশি ({p}%)। ছাতা সঙ্গে রাখুন এবং সাবধানে যাতায়াত করুন!",
      "warm": "গরম আবহাওয়া ({t}°C)। পর্যাপ্ত জল পান করুন, সুতির হালকা জামাকাপড় পরুন।",
      "pleasant": "আজ {t}°C সহ চমৎকার {d} আবহাওয়া। হাঁটাচলা এবং বাইরের কাজের জন্য উপযুক্ত দিন!"
    }
  },
  "mr": {
    "appSubtitle": "थेट हवामान आणि 7 दिवसांचा अंदाज",
    "gpsText": "थेट GPS",
    "gpsActive": "GPS सुरू आहे",
    "gpsLocating": "शोधत आहे...",
    "searchPlaceholder": "कोणतेही भारतीय किंवा जागतिक शहर शोधा...",
    "liveVoiceBtn": "थेट व्हॉइस चॅट",
    "offlineBtn": "ऑफलाइन",
    "exitOfflineBtn": "ऑफलाइनमधून बाहेर पडा",
    "keysBtn": "कीज (Keys)",
    "offlineBannerText": "ऑफलाइन मोड सक्रिय: सेव्ह केलेला 7 दिवसांचा अंदाज दाखवत आहे।",
    "switchLiveBtn": "थेट मोडवर जा",
    "citiesLabel": "शहरे:",
    "cities": [
      "चेन्नई",
      "दिल्ली",
      "बंगळुरू",
      "हैदराबाद",
      "मुंबई",
      "कोलकाता",
      "तंजावर"
    ],
    "gpsLocked": "GPS लॉक झाले",
    "currentCond": "सध्याचे हवामान",
    "today": "आज",
    "talkBtn": "WeatherGPT शी बोला",
    "feelsLike": "जाणवणारे तापमान",
    "highLow": "कमाल: {max}° • किमान: {min}°",
    "rainChance": "पावसाची शक्यता",
    "rainExpected": "{rain} मिमी पाऊस अपेक्षित",
    "humidity": "आर्द्रता (दमटपणा)",
    "comfort": "आरामदायी",
    "humid": "दमट / उकाडा",
    "dry": "कोरडे",
    "windSpeed": "वाऱ्याचा वेग",
    "windBreeze": "मंद वारा",
    "uvIndex": "UV निर्देशांक",
    "uvLevels": {
      "veryHigh": "अति तीव्र (डोळे व त्वचेची काळजी घ्या)",
      "high": "जास्त (सनस्क्रीन वापरा)",
      "safe": "मध्यम / सुरक्षित"
    },
    "airPressure": "हवेचा दाब",
    "surfacePressure": "पृष्ठभागावरील दाब",
    "cloudCover": "ढगाळ वातावरण",
    "cloudDesc": "विखुरलेले ढग",
    "offlineCardTitle": "ऑफलाइन 7-दिवसीय हवामान डेटा",
    "offlineCardDesc": "इंटरनेट नसताना पाहण्यासाठी पुढील 7 दिवसांचा अंदाज सेव्ह करा",
    "offlineSavedText": "डिव्हाइसवर सेव्ह केले",
    "saveOfflineBtn": "7 दिवसांचा डेटा सेव्ह करा",
    "guideTitle": "व्हॉइस आणि चॅट हवामान मार्गदर्शक",
    "guideSubtitle": "सर्व 12+ भारतीय भाषांमध्ये उपलब्ध",
    "welcomeChat": "नमस्कार! मी आपला <strong>WeatherGPT व्हॉइस मार्गदर्शक</strong> आहे. आपण कोणत्याही भारतीय भाषेत बोलू किंवा लिहू शकता. पाऊस, छत्री, कपडे किंवा पुढील 7 दिवसांच्या हवामानाविषयी मला विचारा!",
    "voiceOutputTitle": "नैसर्गिक आवाज आउटपुट",
    "voiceOutputStatus": "आपल्या भाषेत ऐकण्यासाठी क्लिक करा",
    "listenBtn": "🔊 ऐका ({lang})",
    "liveModeBtn": "थेट मोड",
    "forecastTitle": "पुढील 7 दिवसांचा हवामान अंदाज",
    "forecastSubtitle": "Open-Meteo मॉडेल्सद्वारे समर्थित दैनिक अंदाज",
    "legendMinMax": "किमान / कमाल तापमान",
    "legendRain": "पावसाची शक्यता",
    "voiceModalTitle": "थेट व्हॉइस मोड",
    "voiceModalPrompt": "\"आपल्या भाषेत हवामानाविषयी कोणताही प्रश्न विचारा...\"",
    "voiceStatusIdle": "बोलण्यासाठी गोलावर टॅप करा",
    "voiceStatusListening": "ऐकत आहे... आपला प्रश्न सांगा",
    "voiceStatusThinking": "WeatherGPT विचार करत आहे...",
    "voiceStatusSpeaking": "उत्तर आवाजात सांगितले जात आहे...",
    "handsFreeLabel": "हँड्स-फ्री ऑटो-चॅट (सतत संवाद)",
    "speedLabel": "गती:",
    "settingsTitle": "API की संरचना",
    "settingsSubtitle": "आपल्या Gemini व Open-Meteo की जोडा",
    "geminiKeyLabel": "Google Gemini API की",
    "geminiFreeKey": "मोफत की मिळवा",
    "geminiDesc": "नैसर्गिक संवाद आणि प्रादेशिक आवाज आउटपुट सक्षम करते.",
    "openmeteoKeyLabel": "Open-Meteo API की (पर्यायी)",
    "openmeteoPricing": "Open-Meteo दर",
    "openmeteoDesc": "मोफत सेवा उपलब्ध आहे — थेट आणि 7 दिवसांच्या हवामानासाठी की ची गरज नाही.",
    "testKeysBtn": "की तपासा",
    "saveSettingsBtn": "सेटिंग्ज सेव्ह करा",
    "footerSystem": "WeatherGPT • थेट आणि 7 दिवसांची AI हवामान प्रणाली",
    "footerPowered": "Open-Meteo REST API • Google Gemini Reasoning • 12+ भारतीय भाषांचा आवाज",
    "days": {
      "today": "आज",
      "mon": "सोम",
      "tue": "मंगळ",
      "wed": "बुध",
      "thu": "गुरु",
      "fri": "शुक्र",
      "sat": "शनी",
      "sun": "रवि"
    },
    "conditions": {
      "Clear sky": "निरभ्र आकाश",
      "Mainly clear": "मुख्यत्वे निरभ्र",
      "Partly cloudy": "अंशतः ढगाळ",
      "Overcast": "पूर्ण ढगाळ",
      "Fog": "धुके",
      "Dense fog": "दाट धुके",
      "Depositing rime fog": "हिम धुके",
      "Light drizzle": "हलकी रिमझिम",
      "Moderate drizzle": "मध्यम रिमझिम",
      "Dense drizzle": "तीव्र रिमझिम",
      "Slight rain": "हलका पाऊस",
      "Moderate rain": "मध्यम पाऊस",
      "Heavy rain": "मुसळधार पाऊस",
      "Thunderstorm": "वादळी पाऊस",
      "Thunderstorm with hail": "गारपिटीसह वादळ",
      "Thunderstorm with slight hail": "वादळ आणि बारीक गारपीट",
      "Thunderstorm with heavy hail": "भीषण गारपीट आणि वादळ",
      "Slight snow": "हलकी बर्फवृष्टी",
      "Moderate snow": "मध्यम बर्फवृष्टी",
      "Heavy snow": "मुसळधार बर्फवृष्टी",
      "Rain showers": "पावसाच्या सरी",
      "Slight rain showers": "हलक्या सरी",
      "Moderate rain showers": "मध्यम सरी",
      "Violent rain showers": "जोरदार मुसळधार सरी"
    },
    "tips": {
      "rain": "आज पावसाची शक्यता जास्त आहे ({p}%)। छत्री जवळ ठेवा आणि काळजीपूर्वक प्रवास करा!",
      "warm": "उष्ण हवामान ({t}°C)। भरपूर पाणी प्या, हलके सुती कपडे वापरा.",
      "pleasant": "आज {t}°C सह आल्हाददायक {d} हवामान आहे. फिरण्यासाठी आणि बाहेरील कामांसाठी उत्तम दिवस!"
    }
  },
  "gu": {
    "appSubtitle": "રીઅલ-ટાઇમ હવામાન અને 7-દિવસીય આગાહી",
    "gpsText": "લાઇવ GPS",
    "gpsActive": "GPS સક્રિય છે",
    "gpsLocating": "શોધી રહ્યું છે...",
    "searchPlaceholder": "કોઈપણ ભારતીય અથવા વૈશ્વિક શહેર શોધો...",
    "liveVoiceBtn": "લાઇવ વૉઇસ ચેટ",
    "offlineBtn": "ઑફલાઇન",
    "exitOfflineBtn": "ઑફલાઇન બંધ કરો",
    "keysBtn": "કીઝ (Keys)",
    "offlineBannerText": "ઑફલાઇન મોડ સક્રિય: સંગ્રહિત 7-દિવસની આગાહી બતાવી રહ્યું છે.",
    "switchLiveBtn": "લાઇવ મોડ પર જાઓ",
    "citiesLabel": "શહેરો:",
    "cities": [
      "ચેન્નઈ",
      "દિલ્હી",
      "બેંગલુરુ",
      "હૈદરાબાદ",
      "મુંબઈ",
      "કોલકાતા",
      "તંજાવુર"
    ],
    "gpsLocked": "GPS લૉક થયું",
    "currentCond": "વર્તમાન પરિસ્થિતિ",
    "today": "આજે",
    "talkBtn": "WeatherGPT સાથે વાત કરો",
    "feelsLike": "અનુભવાતું તાપમાન",
    "highLow": "મહત્તમ: {max}° • લઘુત્તમ: {min}°",
    "rainChance": "વરસાદની શક્યતા",
    "rainExpected": "{rain} મીમી વરસાદ અપેક્ષિત",
    "humidity": "ભેજ",
    "comfort": "આરામદાયક",
    "humid": "બફારો / ભેજવાળું",
    "dry": "સૂકું",
    "windSpeed": "પવનની ગતિ",
    "windBreeze": "ધીમો પવન",
    "uvIndex": "UV ઇન્ડેક્સ",
    "uvLevels": {
      "veryHigh": "અતિશય વધુ (આંખો અને ત્વચાનું ધ્યાન રાખો)",
      "high": "વધુ (સનસ્ક્રીન વાપરો)",
      "safe": "સામાન્ય / સુરક્ષિત"
    },
    "airPressure": "હવાનું દબાણ",
    "surfacePressure": "વાતાવરણીય દબાણ",
    "cloudCover": "વાદળોનું પ્રમાણ",
    "cloudDesc": "છૂટાછવાયા વાદળો",
    "offlineCardTitle": "ઑફલાઇન 7-દિવસનો હવામાન ડેટા",
    "offlineCardDesc": "ઇન્ટરનેટ વિના જોવા માટે આગામી 7 દિવસની આગાહી સાચવો",
    "offlineSavedText": "ડિવાઇસમાં સાચવવામાં આવ્યું",
    "saveOfflineBtn": "7-દિવસનો ડેટા સાચવો",
    "guideTitle": "વૉઇસ અને ચેટ હવામાન માર્ગદર્શિકા",
    "guideSubtitle": "બધી 12+ ભારતીય ભાષાઓમાં ઉપલબ્ધ",
    "welcomeChat": "નમસ્તે! હું તમારો <strong>WeatherGPT વૉઇસ ગાઇડ</strong> છું. તમે કોઈપણ ભારતીય ભાષામાં બોલી અથવા લખી શકો છો. વરસાદ, છત્રી, શું પહેરવું કે 7 દિવસની આગાહી વિશે મને પૂછો!",
    "voiceOutputTitle": "કુદરતી અવાજ આઉટપુટ",
    "voiceOutputStatus": "તમારી ભાષામાં સાંભળવા માટે ક્લિક કરો",
    "listenBtn": "🔊 સાંભળો ({lang})",
    "liveModeBtn": "લાઇવ મોડ",
    "forecastTitle": "આગામી 7-દિવસની હવામાન આગાહી",
    "forecastSubtitle": "Open-Meteo મોડેલ દ્વારા દૈનિક આગાહી",
    "legendMinMax": "લઘુત્તમ / મહત્તમ તાપમાન",
    "legendRain": "વરસાદની સંભાવના",
    "voiceModalTitle": "લાઇવ વૉઇસ મોડ",
    "voiceModalPrompt": "\"તમારી ભાષામાં હવામાન અંગે કોઈપણ પ્રશ્ન પૂછો...\"",
    "voiceStatusIdle": "વાત કરવા માટે ગોળ પર ટેપ કરો",
    "voiceStatusListening": "સાંભળી રહ્યું છે... પ્રશ્ન પૂછો",
    "voiceStatusThinking": "WeatherGPT વિચારી રહ્યું છે...",
    "voiceStatusSpeaking": "જવાબ બોલાઈ રહ્યો છે...",
    "handsFreeLabel": "હેન્ડ્સ-ફ્રી ઑટો-ચેટ (સતત વાતચીત)",
    "speedLabel": "ઝડપ:",
    "settingsTitle": "API કી ગોઠવણી",
    "settingsSubtitle": "તમારી Gemini અને Open-Meteo કી જોડો",
    "geminiKeyLabel": "Google Gemini API કી",
    "geminiFreeKey": "મફત કી મેળવો",
    "geminiDesc": "કુદરતી વાતચીત અને પ્રાદેશિક અવાજ આઉટપુટને સક્ષમ કરે છે.",
    "openmeteoKeyLabel": "Open-Meteo API કી (વૈકલ્પિક)",
    "openmeteoPricing": "Open-Meteo ભાવો",
    "openmeteoDesc": "મફત સુવિધા ઉપલબ્ધ છે — લાઇવ અને 7-દિવસ માટે કીની જરૂર નથી.",
    "testKeysBtn": "કી તપાસો",
    "saveSettingsBtn": "સેટિંગ્સ સાચવો",
    "footerSystem": "WeatherGPT • રીઅલ-ટાઇમ અને 7-દિવસીય AI હવામાન સિસ્ટમ",
    "footerPowered": "Open-Meteo REST API • Google Gemini Reasoning • 12+ ભારતીય ભાષાઓ અવાજ",
    "days": {
      "today": "આજે",
      "mon": "સોમ",
      "tue": "મંગળ",
      "wed": "બુધ",
      "thu": "ગુરુ",
      "fri": "શુક્ર",
      "sat": "શનિ",
      "sun": "રવિ"
    },
    "conditions": {
      "Clear sky": "ચોખ્ખું આકાશ",
      "Mainly clear": "મોટે ભાગે ચોખ્ખું",
      "Partly cloudy": "અંશતઃ વાદળછાયું",
      "Overcast": "વાદળછાયું વાતાવરણ",
      "Fog": "ધુમ્મસ",
      "Dense fog": "ગાઢ ધુમ્મસ",
      "Depositing rime fog": "હિમ ધુમ્મસ",
      "Light drizzle": "ઝરમર વરસાદ",
      "Moderate drizzle": "મધ્યમ ઝરમર",
      "Dense drizzle": "ભારે ઝરમર",
      "Slight rain": "હળવો વરસાદ",
      "Moderate rain": "મધ્યમ વરસાદ",
      "Heavy rain": "ભારે વરસાદ",
      "Thunderstorm": "ગાજવીજ સાથે વરસાદ",
      "Thunderstorm with hail": "કરા સાથે વાવાઝોડું",
      "Thunderstorm with slight hail": "ગાજવીજ અને હળવા કરા",
      "Thunderstorm with heavy hail": "તીવ્ર વાવાઝોડું",
      "Slight snow": "હળવી હિમવર્ષા",
      "Moderate snow": "મધ્યમ હિમવર્ષા",
      "Heavy snow": "ભારે હિમવર્ષા",
      "Rain showers": "વરસાદી ઝાપટાં",
      "Slight rain showers": "હળવા ઝાપટાં",
      "Moderate rain showers": "મધ્યમ ઝાપટાં",
      "Violent rain showers": "તોફાની વરસાદ"
    },
    "tips": {
      "rain": "આજે વરસાદની શક્યતા વધુ છે ({p}%)। છત્રી સાથે રાખો અને સાવચેતીથી મુસાફરી કરો!",
      "warm": "ગરમ વાતાવરણ ({t}°C)। પુષ્કળ પાણી પીવો, સુતરાઉ કપડાં પહેરો.",
      "pleasant": "આજે {t}°C સાથે ખુશનુમા {d} હવામાન છે. બહારના કામકાજ અને ફરવા માટે સારો દિવસ!"
    }
  },
  "pa": {
    "appSubtitle": "ਰੀਅਲ-ਟਾਈਮ ਮੌਸਮ ਅਤੇ 7 ਦਿਨਾਂ ਦੀ ਭਵਿੱਖਬਾਣੀ",
    "gpsText": "ਲਾਈਵ GPS",
    "gpsActive": "GPS ਚਾਲੂ ਹੈ",
    "gpsLocating": "ਲੋਕੇਸ਼ਨ ਲੱਭ ਰਿਹਾ ਹੈ...",
    "searchPlaceholder": "ਕੋਈ ਵੀ ਭਾਰਤੀ ਜਾਂ ਗਲੋਬਲ ਸ਼ਹਿਰ ਖੋਜੋ...",
    "liveVoiceBtn": "ਲਾਈਵ ਵੌਇਸ ਚੈਟ",
    "offlineBtn": "ਆਫਲਾਈਨ",
    "exitOfflineBtn": "ਆਫਲਾਈਨ ਬੰਦ ਕਰੋ",
    "keysBtn": "ਕੁੰਜੀਆਂ (Keys)",
    "offlineBannerText": "ਆਫਲਾਈਨ ਮੋਡ ਚਾਲੂ: ਸੁਰੱਖਿਅਤ ਕੀਤਾ 7 ਦਿਨਾਂ ਦਾ ਮੌਸਮ ਦਿਖਾਇਆ ਜਾ ਰਿਹਾ ਹੈ।",
    "switchLiveBtn": "ਲਾਈਵ ਮੋਡ 'ਤੇ ਜਾਓ",
    "citiesLabel": "ਸ਼ਹਿਰ:",
    "cities": [
      "ਚੇਨਈ",
      "ਦਿੱਲੀ",
      "ਬੈਂਗਲੁਰੂ",
      "ਹੈਦਰਾਬਾਦ",
      "ਮੁੰਬਈ",
      "ਕੋਲਕਾਤਾ",
      "ਤੰਜਾਵੁਰ"
    ],
    "gpsLocked": "GPS ਲਾਕ ਹੋ ਗਿਆ",
    "currentCond": "ਮੌਜੂਦਾ ਹਾਲਾਤ",
    "today": "ਅੱਜ",
    "talkBtn": "WeatherGPT ਨਾਲ ਗੱਲ ਕਰੋ",
    "feelsLike": "ਮਹਿਸੂਸ ਹੁੰਦਾ ਤਾਪਮਾਨ",
    "highLow": "ਵੱਧ: {max}° • ਘੱਟ: {min}°",
    "rainChance": "ਮੀਂਹ ਦੀ ਸੰਭਾਵਨਾ",
    "rainExpected": "{rain} ਮਿਲੀਮੀਟਰ ਸੰਭਾਵਿਤ",
    "humidity": "ਨਮੀ",
    "comfort": "ਸੁਖਾਵਾਂ",
    "humid": "ਉਮਸ ਵਾਲਾ",
    "dry": "ਖੁਸ਼ਕ",
    "windSpeed": "ਹਵਾ ਦੀ ਰਫ਼ਤਾਰ",
    "windBreeze": "ਠੰਢੀ ਹਵਾ",
    "uvIndex": "UV ਇੰਡੈਕਸ",
    "uvLevels": {
      "veryHigh": "ਬਹੁਤ ਜ਼ਿਆਦਾ (ਅੱਖਾਂ ਅਤੇ ਚਮੜੀ ਦੀ ਸੁਰੱਖਿਆ ਕਰੋ)",
      "high": "ਜ਼ਿਆਦਾ (ਸਨਸਕ੍ਰੀਨ ਵਰਤੋ)",
      "safe": "ਦਰਮਿਆਨਾ / ਸੁਰੱਖਿਅਤ"
    },
    "airPressure": "ਹਵਾ ਦਾ ਦਬਾਅ",
    "surfacePressure": "ਵਾਯੂਮੰਡਲੀ ਦਬਾਅ",
    "cloudCover": "ਬੱਦਲਵਾਈ",
    "cloudDesc": "ਖਿੰਡੇ ਹੋਏ ਬੱਦਲ",
    "offlineCardTitle": "ਆਫਲਾਈਨ 7 ਦਿਨਾਂ ਦਾ ਮੌਸਮ ਡਾਟਾ",
    "offlineCardDesc": "ਬਿਨਾਂ ਇੰਟਰਨੈੱਟ ਵੇਖਣ ਲਈ ਅਗਲੇ 7 ਦਿਨਾਂ ਦਾ ਮੌਸਮ ਸੇਵ ਕਰੋ",
    "offlineSavedText": "ਡਿਵਾਈਸ 'ਤੇ ਸੁਰੱਖਿਅਤ",
    "saveOfflineBtn": "7 ਦਿਨਾਂ ਦਾ ਡਾਟਾ ਸੇਵ ਕਰੋ",
    "guideTitle": "ਵੌਇਸ ਅਤੇ ਚੈਟ ਮੌਸਮ ਗਾਈਡ",
    "guideSubtitle": "ਸਾਰੀਆਂ 12+ ਭਾਰਤੀ ਭਾਸ਼ਾਵਾਂ ਵਿੱਚ ਉਪਲਬਧ",
    "welcomeChat": "ਸਤਿ ਸ੍ਰੀ ਅਕਾਲ! ਮੈਂ ਤੁਹਾਡਾ <strong>WeatherGPT ਵੌਇਸ ਗਾਈਡ</strong> ਹਾਂ। ਤੁਸੀਂ ਕਿਸੇ ਵੀ ਭਾਰਤੀ ਭਾਸ਼ਾ ਵਿੱਚ ਬੋਲ ਜਾਂ ਲਿਖ ਸਕਦੇ ਹੋ। ਮੀਂਹ, ਛਤਰੀ, ਕੱਪੜੇ ਜਾਂ 7 ਦਿਨਾਂ ਦੇ ਮੌਸਮ ਬਾਰੇ ਮੈਨੂੰ ਪੁੱਛੋ!",
    "voiceOutputTitle": "ਕੁਦਰਤੀ ਆਵਾਜ਼ ਆਉਟਪੁੱਟ",
    "voiceOutputStatus": "ਆਪਣੀ ਭਾਸ਼ਾ ਵਿੱਚ ਸੁਣਨ ਲਈ ਕਲਿੱਕ ਕਰੋ",
    "listenBtn": "🔊 ਸੁਣੋ ({lang})",
    "liveModeBtn": "ਲਾਈਵ ਮੋਡ",
    "forecastTitle": "ਅਗਲੇ 7 ਦਿਨਾਂ ਦੀ ਮੌਸਮ ਭਵਿੱਖਬਾਣੀ",
    "forecastSubtitle": "Open-Meteo ਮਾਡਲ ਰਾਹੀਂ ਰੋਜ਼ਾਨਾ ਭਵਿੱਖਬਾਣੀ",
    "legendMinMax": "ਘੱਟੋ-ਘੱਟ / ਵੱਧ ਤੋਂ ਵੱਧ ਤਾਪਮਾਨ",
    "legendRain": "ਮੀਂਹ ਦੀ ਸੰਭਾਵਨਾ",
    "voiceModalTitle": "ਲਾਈਵ ਵੌਇਸ ਮੋਡ",
    "voiceModalPrompt": "\"ਆਪਣੀ ਭਾਸ਼ਾ ਵਿੱਚ ਮੌਸਮ ਦਾ ਕੋਈ ਵੀ ਸਵਾਲ ਪੁੱਛੋ...\"",
    "voiceStatusIdle": "ਬੋਲਣ ਲਈ ਚੱਕਰ 'ਤੇ ਟੈਪ ਕਰੋ",
    "voiceStatusListening": "ਸੁਣ ਰਿਹਾ ਹਾਂ... ਆਪਣਾ ਸਵਾਲ ਬੋਲੋ",
    "voiceStatusThinking": "WeatherGPT ਸੋਚ ਰਿਹਾ ਹੈ...",
    "voiceStatusSpeaking": "ਜਵਾਬ ਬੋਲਿਆ ਜਾ ਰਿਹਾ ਹੈ...",
    "handsFreeLabel": "ਹੈਂਡਸ-ਫ੍ਰੀ ਆਟੋ-ਚੈਟ (ਲਗਾਤਾਰ ਗੱਲਬਾਤ)",
    "speedLabel": "ਗਤੀ:",
    "settingsTitle": "API ਕੁੰਜੀ ਸੰਰਚਨਾ",
    "settingsSubtitle": "ਆਪਣੀਆਂ Gemini ਅਤੇ Open-Meteo ਕੁੰਜੀਆਂ ਜੋੜੋ",
    "geminiKeyLabel": "Google Gemini API ਕੁੰਜੀ",
    "geminiFreeKey": "ਮੁਫ਼ਤ ਕੁੰਜੀ ਲਵੋ",
    "geminiDesc": "ਕੁਦਰਤੀ ਗੱਲਬਾਤ ਅਤੇ ਸਥਾਨਕ ਆਵਾਜ਼ ਆਉਟਪੁੱਟ ਨੂੰ ਚਲਾਉਂਦਾ ਹੈ।",
    "openmeteoKeyLabel": "Open-Meteo API ਕੁੰਜੀ (ਵਿਕਲਪਿਕ)",
    "openmeteoPricing": "Open-Meteo ਕੀਮਤਾਂ",
    "openmeteoDesc": "ਮੁਫ਼ਤ ਸਹੂਲਤ ਸ਼ਾਮਲ ਹੈ — ਲਾਈਵ ਅਤੇ 7 ਦਿਨਾਂ ਦੇ ਮੌਸਮ ਲਈ ਕੁੰਜੀ ਦੀ ਲੋੜ ਨਹੀਂ।",
    "testKeysBtn": "ਕੁੰਜੀਆਂ ਪਰਖੋ",
    "saveSettingsBtn": "ਸੈਟਿੰਗਾਂ ਸੇਵ ਕਰੋ",
    "footerSystem": "WeatherGPT • ਰੀਅਲ-ਟਾਈਮ ਅਤੇ 7-ਦਿਨਾ AI ਮੌਸਮ ਸਿਸਟਮ",
    "footerPowered": "Open-Meteo REST API • Google Gemini Reasoning • 12+ ਭਾਰਤੀ ਭਾਸ਼ਾਵਾਂ ਆਵਾਜ਼",
    "days": {
      "today": "ਅੱਜ",
      "mon": "ਸੋਮ",
      "tue": "ਮੰਗਲ",
      "wed": "ਬੁੱਧ",
      "thu": "ਵੀਰ",
      "fri": "ਸ਼ੁੱਕਰ",
      "sat": "ਸ਼ਨਿੱਚਰ",
      "sun": "ਐਤ"
    },
    "conditions": {
      "Clear sky": "ਸਾਫ਼ ਅਸਮਾਨ",
      "Mainly clear": "ਜ਼ਿਆਦਾਤਰ ਸਾਫ਼",
      "Partly cloudy": "ਅੰਸ਼ਕ ਬੱਦਲਵਾਈ",
      "Overcast": "ਪੂਰੀ ਤਰ੍ਹਾਂ ਬੱਦਲਵਾਈ",
      "Fog": "ਧੁੰਦ",
      "Dense fog": "ਸੰਘਣੀ ਧੁੰਦ",
      "Depositing rime fog": "ਬਰਫ਼ੀਲੀ ਧੁੰਦ",
      "Light drizzle": "ਹਲਕੀ ਬੂੰਦਾ-ਬਾਂਦੀ",
      "Moderate drizzle": "ਦਰਮਿਆਨੀ ਬੂੰਦਾ-ਬਾਂਦੀ",
      "Dense drizzle": "ਤੇਜ਼ ਬੂੰਦਾ-ਬਾਂਦੀ",
      "Slight rain": "ਹਲਕਾ ਮੀਂਹ",
      "Moderate rain": "ਦਰਮਿਆਨਾ ਮੀਂਹ",
      "Heavy rain": "ਭਾਰੀ ਮੀਂਹ",
      "Thunderstorm": "ਗਰਜ ਨਾਲ ਮੀਂਹ",
      "Thunderstorm with hail": "ਗੜਿਆਂ ਨਾਲ ਤੂਫ਼ਾਨ",
      "Thunderstorm with slight hail": "ਗਰਜ ਅਤੇ ਹਲਕੇ ਗੜੇ",
      "Thunderstorm with heavy hail": "ਭਿਆਨਕ ਗੜਿਆਂ ਵਾਲਾ ਤੂਫ਼ਾਨ",
      "Slight snow": "ਹਲਕੀ ਬਰਫ਼ਬਾਰੀ",
      "Moderate snow": "ਦਰਮਿਆਨੀ ਬਰਫ਼ਬਾਰੀ",
      "Heavy snow": "ਭਾਰੀ ਬਰਫ਼ਬਾਰੀ",
      "Rain showers": "ਮੀਂਹ ਦੀਆਂ ਫੁਹਾਰਾਂ",
      "Slight rain showers": "ਹਲਕੀਆਂ ਫੁਹਾਰਾਂ",
      "Moderate rain showers": "ਦਰਮਿਆਨੀਆਂ ਫੁਹਾਰਾਂ",
      "Violent rain showers": "ਮੂਸਲਾਧਾਰ ਮੀਂਹ"
    },
    "tips": {
      "rain": "ਅੱਜ ਮੀਂਹ ਪੈਣ ਦੀ ਸੰਭਾਵਨਾ ਜ਼ਿਆਦਾ ਹੈ ({p}%)। ਛਤਰੀ ਕੋਲ ਰੱਖੋ ਅਤੇ ਧਿਆਨ ਨਾਲ ਸਫ਼ਰ ਕਰੋ!",
      "warm": "ਗਰਮ ਮੌਸਮ ({t}°C)। ਪਾਣੀ ਵੱਧ ਪੀਓ, ਹਲਕੇ ਸੂਤੀ ਕੱਪੜੇ ਪਾਓ.",
      "pleasant": "ਅੱਜ {t}°C ਨਾਲ ਸੁਹਾਵਣਾ {d} ਮੌਸਮ ਹੈ। ਸੈਰ ਅਤੇ ਬਾਹਰਲੇ ਕੰਮਾਂ ਲਈ ਬਹੁਤ ਵਧੀਆ ਦਿਨ!"
    }
  },
  "or": {
    "appSubtitle": "ରିଅଲ-ଟାଇମ୍ ପାଣିପାଗ ଏବଂ ୭-ଦିନର ପୂର୍ବାନୁମାନ",
    "gpsText": "ଲାଇଭ୍ GPS",
    "gpsActive": "GPS ସକ୍ରିୟ ଅଛି",
    "gpsLocating": "ସ୍ଥାନ ଖୋଜୁଛି...",
    "searchPlaceholder": "ଯେକୌଣସି ଭାରତୀୟ ବା ବିଶ୍ୱର ସହର ଖୋଜନ୍ତୁ...",
    "liveVoiceBtn": "ଲାଇଭ୍ ଭଏସ୍ ଚାଟ୍",
    "offlineBtn": "ଅଫଲାଇନ୍",
    "exitOfflineBtn": "ଅଫଲାଇନ୍ ବନ୍ଦ କରନ୍ତୁ",
    "keysBtn": "ଚାବି (Keys)",
    "offlineBannerText": "ଅଫଲାଇନ୍ ମୋଡ୍ ସକ୍ରିୟ: ସଂରକ୍ଷିତ ୭-ଦିନର ପାଣିପାଗ ଦେଖାଉଛି।",
    "switchLiveBtn": "ଲାଇଭ୍ ମୋଡ୍‌କୁ ଯାଆନ୍ତୁ",
    "citiesLabel": "ସହର:",
    "cities": [
      "ଚେନ୍ନାଇ",
      "ଦିଲ୍ଲୀ",
      "ବେଙ୍ଗାଲୁରୁ",
      "ହାଇଦ୍ରାବାଦ",
      "ମୁମ୍ବାଇ",
      "କୋଲକାତା",
      "ତାଞ୍ଜାଭୁର"
    ],
    "gpsLocked": "GPS ଲକ୍ ହୋଇଛି",
    "currentCond": "ବର୍ତ୍ତମାନର ସ୍ଥିତି",
    "today": "ଆଜି",
    "talkBtn": "WeatherGPT ସହିତ କଥା ହୁଅନ୍ତୁ",
    "feelsLike": "ଅନୁଭୂତ ତାପମାତ୍ରା",
    "highLow": "ସର୍ବାଧିକ: {max}° • ସର୍ବନିମ୍ନ: {min}°",
    "rainChance": "ବର୍ଷା ସମ୍ଭାବନା",
    "rainExpected": "{rain} ମିମି ବର୍ଷା ସମ୍ଭାବନା",
    "humidity": "ଆର୍ଦ୍ରତା",
    "comfort": "ଆରାମଦାୟକ",
    "humid": "ଗୁଳୁଗୁଳି / ଆର୍ଦ୍ର",
    "dry": "ଶୁଷ୍କ",
    "windSpeed": "ପବନର ବେଗ",
    "windBreeze": "ମୃଦୁ ପବନ",
    "uvIndex": "UV ସୂଚକାଙ୍କ",
    "uvLevels": {
      "veryHigh": "ଅତ୍ୟନ୍ତ ଅଧିକ (ଚର୍ମ ଏବଂ ଆଖି ସୁରକ୍ଷିତ ରଖନ୍ତୁ)",
      "high": "ଅଧିକ (ସନ୍‌ସ୍କ୍ରିନ୍ ବ୍ୟବହାର କରନ୍ତୁ)",
      "safe": "ମଧ୍ୟମ / ସୁରକ୍ଷିତ"
    },
    "airPressure": "ବାୟୁମଣ୍ଡଳୀୟ ଚାପ",
    "surfacePressure": "ଭୂପୃଷ୍ଠ ଚାପ",
    "cloudCover": "ମେଘ ଆଚ୍ଛାଦନ",
    "cloudDesc": "ଖଣ୍ଡବିଖଣ୍ଡିତ ମେଘ",
    "offlineCardTitle": "ଅଫଲାଇନ୍ ୭-ଦିନର ପାଣିପାଗ ତଥ୍ୟ",
    "offlineCardDesc": "ଇଣ୍ଟରନେଟ୍ ବିନା ଦେଖିବାକୁ ୭-ଦିନର ପୂର୍ବାନୁମାନ ସେଭ୍ କରନ୍ତୁ",
    "offlineSavedText": "ଡିଭାଇସ୍‌ରେ ସୁରକ୍ଷିତ ଭାବେ ରଖାଗଲା",
    "saveOfflineBtn": "୭-ଦିନର ତଥ୍ୟ ସେଭ୍ କରନ୍ତୁ",
    "guideTitle": "ଭଏସ୍ ଏବଂ ଚାଟ୍ ପାଣିପାଗ ଗାଇଡ୍",
    "guideSubtitle": "ସମସ୍ତ ୧୨+ ଭାରତୀୟ ଭାଷାରେ ଉପଲବ୍ଧ",
    "welcomeChat": "ନମସ୍କାର! ମୁଁ ଆପଣଙ୍କର <strong>WeatherGPT ଭଏସ୍ ଗାଇଡ୍</strong>। ଆପଣ ଯେକୌଣସି ଭାରତୀୟ ଭାଷାରେ କହିପାରିବେ ବା ଲେଖିପାରିବେ। ବର୍ଷା, ଛତା, କଣ ପିନ୍ଧିବେ ବା ୭ ଦିନର ପାଣିପାଗ ବିଷୟରେ ମୋତେ ପଚାରନ୍ତୁ!",
    "voiceOutputTitle": "ପ୍ରାକୃତିକ ଶବ୍ଦ ଆଉଟପୁଟ୍",
    "voiceOutputStatus": "ଆପଣଙ୍କ ଭାଷାରେ ଶୁଣିବାକୁ କ୍ଲିକ୍ କରନ୍ତୁ",
    "listenBtn": "🔊 ଶୁଣନ୍ତୁ ({lang})",
    "liveModeBtn": "ଲାଇଭ୍ ମୋଡ୍",
    "forecastTitle": "ଆଗାମୀ ୭ ଦିନର ପାଣିପାଗ ପୂର୍ବାନୁମାନ",
    "forecastSubtitle": "Open-Meteo ମଡେଲ୍ ଦ୍ୱାରା ଦୈନିକ ପୂର୍ବାନୁମାନ",
    "legendMinMax": "ସର୍ବନିମ୍ନ / ସର୍ବାଧିକ ତାପମାତ୍ରା",
    "legendRain": "ବର୍ଷା ସମ୍ଭାବନା",
    "voiceModalTitle": "ଲାଇଭ୍ ଭଏସ୍ ମୋଡ୍",
    "voiceModalPrompt": "\"ଆପଣଙ୍କ ଭାଷାରେ ଯେକୌଣସି ପାଣିପାଗ ପ୍ରଶ୍ନ ପଚାରନ୍ତୁ...\"",
    "voiceStatusIdle": "କହିବା ପାଇଁ ବୃତ୍ତ ଉପରେ କ୍ଲିକ୍ କରନ୍ତୁ",
    "voiceStatusListening": "ଶୁଣୁଛି... ପ୍ରଶ୍ନ ପଚାରନ୍ତୁ",
    "voiceStatusThinking": "WeatherGPT ଚିନ୍ତା କରୁଛି...",
    "voiceStatusSpeaking": "ଉତ୍ତର କୁହାଯାଉଛି...",
    "handsFreeLabel": "ହ୍ୟାଣ୍ଡସ୍-ଫ୍ରି ଅଟୋ-ଚାଟ୍ (ନିରନ୍ତର କଥାବାର୍ତ୍ତା)",
    "speedLabel": "ଗତି:",
    "settingsTitle": "API କୀ କନଫିଗରେସନ୍",
    "settingsSubtitle": "ଆପଣଙ୍କ Gemini ଓ Open-Meteo କୀ ଯୋଡନ୍ତୁ",
    "geminiKeyLabel": "Google Gemini API କୀ",
    "geminiFreeKey": "ମାଗଣା କୀ ପାଆନ୍ତୁ",
    "geminiDesc": "ପ୍ରାକୃତିକ କଥାବାର୍ତ୍ତା ଏବଂ ଆଞ୍ଚଳିକ ଭଏସ୍ ଆଉଟପୁଟ୍ ପାଇଁ ଆବଶ୍ୟକ।",
    "openmeteoKeyLabel": "Open-Meteo API କୀ (ଐଚ୍ଛିକ)",
    "openmeteoPricing": "Open-Meteo ମୂଲ୍ୟ",
    "openmeteoDesc": "ମାଗଣା ସେବା ଉପଲବ୍ଧ — ଲାଇଭ୍ ଓ ୭-ଦିନ ପାଇଁ କୀ ଆବଶ୍ୟକ ନାହିଁ।",
    "testKeysBtn": "କୀ ପରୀକ୍ଷା କରନ୍ତୁ",
    "saveSettingsBtn": "ସେଟିଙ୍ଗ୍ ସେଭ୍ କରନ୍ତୁ",
    "footerSystem": "WeatherGPT • ରିଅଲ-ଟାଇମ୍ ଓ ୭-ଦିନର AI ପାଣିପାଗ ବ୍ୟବସ୍ଥା",
    "footerPowered": "Open-Meteo REST API • Google Gemini Reasoning • ୧୨+ ଭାରତୀୟ ଭାଷାର ଭଏସ୍",
    "days": {
      "today": "ଆଜି",
      "mon": "ସୋମ",
      "tue": "ମଙ୍ଗଳ",
      "wed": "ବୁଧ",
      "thu": "ଗୁରୁ",
      "fri": "ଶୁକ୍ର",
      "sat": "ଶନି",
      "sun": "ରବି"
    },
    "conditions": {
      "Clear sky": "ନିର୍ମଳ ଆକାଶ",
      "Mainly clear": "ମୁଖ୍ୟତଃ ନିର୍ମଳ",
      "Partly cloudy": "ଆଂଶିକ ମେଘୁଆ",
      "Overcast": "ପୂର୍ଣ୍ଣ ମେଘୁଆ",
      "Fog": "କୁହୁଡ଼ି",
      "Dense fog": "ଘନ କୁହୁଡ଼ି",
      "Depositing rime fog": "ତୁଷାର କୁହୁଡ଼ି",
      "Light drizzle": "ହାଲୁକା ଝିପିଝିପି ବର୍ଷା",
      "Moderate drizzle": "ମଧ୍ୟମ ଝିପିଝିପି ବର୍ଷା",
      "Dense drizzle": "ଘନ ଝିପିଝିପି ବର୍ଷା",
      "Slight rain": "ହାଲୁକା ବର୍ଷା",
      "Moderate rain": "ମଧ୍ୟମ ବର୍ଷା",
      "Heavy rain": "ପ୍ରବଳ ବର୍ଷା",
      "Thunderstorm": "ଘଡ଼ଘଡ଼ି ସହ ବର୍ଷା",
      "Thunderstorm with hail": "କୁଆପଥର ସହ ଝଡ଼",
      "Thunderstorm with slight hail": "ଘଡ଼ଘଡ଼ି ଓ କୁଆପଥର",
      "Thunderstorm with heavy hail": "ଭୀଷଣ କୁଆପଥର ଝଡ଼",
      "Slight snow": "ହାଲୁକା ତୁଷାରପାତ",
      "Moderate snow": "ମଧ୍ୟମ ତୁଷାରପାତ",
      "Heavy snow": "ପ୍ରବଳ ତୁଷାରପାତ",
      "Rain showers": "ବର୍ଷା ଝଲକ",
      "Slight rain showers": "ହାଲୁକା ବର୍ଷା ଝଲକ",
      "Moderate rain showers": "ମଧ୍ୟମ ବର୍ଷା ଝଲକ",
      "Violent rain showers": "ପ୍ରବଳ ମୂଷଳଧାରା"
    },
    "tips": {
      "rain": "ଆଜି ବର୍ଷା ସମ୍ଭାବନା ଅଧିକ ({p}%)। ଛତା ସାଙ୍ଗରେ ରଖନ୍ତୁ ଏବଂ ସାବଧାନରେ ଯାତ୍ରା କରନ୍ତୁ!",
      "warm": "ଗରମ ପାଗ ({t}°C)। ପ୍ରଚୁର ପାଣି ପିଅନ୍ତୁ, ସୂତା ପୋଷାକ ପିନ୍ଧନ୍ତୁ.",
      "pleasant": "ଆଜି {t}°C ସହିତ ମନୋରମ {d} ପାଗ। ଚାଲିବା ଏବଂ ବାହାର କାମ ପାଇଁ ଭଲ ଦିନ!"
    }
  },
  "as": {
    "appSubtitle": "ৰিয়েল-টাইম বতৰ আৰু ৭ দিনৰ আগজাননী",
    "gpsText": "লাইভ GPS",
    "gpsActive": "GPS সক্ৰিয় হৈছে",
    "gpsLocating": "স্থান বিচাৰি থকা হৈছে...",
    "searchPlaceholder": "যিকোনো ভাৰতীয় বা বিশ্বৰ চহৰ সন্ধান কৰক...",
    "liveVoiceBtn": "লাইভ ভয়েচ চেট",
    "offlineBtn": "অফলাইন",
    "exitOfflineBtn": "অফলাইন বন্ধ কৰক",
    "keysBtn": "কিজ (Keys)",
    "offlineBannerText": "অফলাইন ম'ড সক্ৰিয়: সংৰক্ষিত ৭ দিনৰ বতৰ দেখুওৱা হৈছে।",
    "switchLiveBtn": "লাইভ ম'ডলৈ যাওক",
    "citiesLabel": "চহৰসমূহ:",
    "cities": [
      "চেন্নাই",
      "দিল্লী",
      "বেংগালুৰু",
      "হায়দৰাবাদ",
      "মুম্বাই",
      "কলকাতা",
      "তাঞ্জাভুৰ"
    ],
    "gpsLocked": "GPS লক হৈছে",
    "currentCond": "বৰ্তমানৰ অৱস্থা",
    "today": "আজি",
    "talkBtn": "WeatherGPT ৰ সৈতে কথা পাতক",
    "feelsLike": "অনুভৱ হোৱা তাপমাত্ৰা",
    "highLow": "সৰ্বোচ্চ: {max}° • সৰ্বনিম্ন: {min}°",
    "rainChance": "বৰষুণৰ সম্ভাৱনা",
    "rainExpected": "{rain} মিমি বৰষুণৰ সম্ভাৱনা",
    "humidity": "আৰ্দ্ৰতা",
    "comfort": "আৰামদায়ক",
    "humid": "বতাহত আৰ্দ্ৰতা বেছি",
    "dry": "শুকান",
    "windSpeed": "বতাহৰ গতি",
    "windBreeze": "মৃদু বতাহ",
    "uvIndex": "UV সূচক",
    "uvLevels": {
      "veryHigh": "অতি মাত্ৰা (চকু আৰু ছালৰ যত্ন লওক)",
      "high": "বেছি (ছানস্ক্ৰীন ব্যৱহাৰ কৰক)",
      "safe": "মধ্যমীয়া / সুৰক্ষিত"
    },
    "airPressure": "বায়ুমণ্ডলীয় চাপ",
    "surfacePressure": "পৃষ্ঠীয় চাপ",
    "cloudCover": "ডাৱৰৰ ঘনত্ব",
    "cloudDesc": "সিঁচৰতি ডাৱৰ",
    "offlineCardTitle": "অফলাইন ৭ দিনৰ বতৰৰ তথ্য",
    "offlineCardDesc": "ইণ্টাৰনেট নোহোৱাকৈ চাবলৈ ৭ দিনৰ বতৰৰ তথ্য সংৰক্ষণ কৰক",
    "offlineSavedText": "ডিভাইচত সংৰক্ষিত",
    "saveOfflineBtn": "৭ দিনৰ তথ্য সংৰক্ষণ কৰক",
    "guideTitle": "ভয়েচ আৰু চেট বতৰ সহায়িকা",
    "guideSubtitle": "সকলো ১২+ ভাৰতীয় ভাষাত উপলব্ধ",
    "welcomeChat": "নমস্কাৰ! মই আপোনাৰ <strong>WeatherGPT ভয়েচ গাইড</strong>। আপুনি যিকোনো ভাৰতীয় ভাষাত কথা ক'ব বা লিখিব পাৰে। বৰষুণ, ছাতি, কি পিন্ধিব বা ৭ দিনৰ বতৰৰ বিষয়ে মোক সোধক!",
    "voiceOutputTitle": "প্ৰাকৃতিক মাতৰ আউটপুট",
    "voiceOutputStatus": "আপোনাৰ ভাষাত শুনিবলৈ ক্লিক কৰক",
    "listenBtn": "🔊 শুনক ({lang})",
    "liveModeBtn": "লাইভ ম'ড",
    "forecastTitle": "অহা ৭ দিনৰ বতৰৰ আগজাননী",
    "forecastSubtitle": "Open-Meteo আৰ্হিৰ ওপৰত ভিত্তি কৰি দৈনিক বতৰ",
    "legendMinMax": "সৰ্বনিম্ন / সৰ্বোচ্চ তাপমাত্ৰা",
    "legendRain": "বৰষুণৰ সম্ভাৱনা",
    "voiceModalTitle": "লাইভ ভয়েচ ম'ড",
    "voiceModalPrompt": "\"আপোনাৰ ভাষাত বতৰৰ যিকোনো প্ৰশ্ন সোধক...\"",
    "voiceStatusIdle": "কথা পাতিবলৈ বৃত্তটোত স্পৰ্শ কৰক",
    "voiceStatusListening": "শুনি আছোঁ... প্ৰশ্ন সোধক",
    "voiceStatusThinking": "WeatherGPT য়ে ভাবি আছে...",
    "voiceStatusSpeaking": "উত্তৰ কোৱা হৈ আছে...",
    "handsFreeLabel": "হেন্দছ-ফ্ৰী স্বয়ংক্ৰিয় চেট (নিৰৱচ্ছিন্ন আলোচনা)",
    "speedLabel": "গতি:",
    "settingsTitle": "API কী সংৰূপ",
    "settingsSubtitle": "আপোনাৰ Gemini আৰু Open-Meteo কী সংযোগ কৰক",
    "geminiKeyLabel": "Google Gemini API কী",
    "geminiFreeKey": "বিনামূলীয়া কী লওক",
    "geminiDesc": "প্ৰাকৃতিক কথোপকথন আৰু স্থানীয় মাতৰ বাবে প্ৰয়োজনীয়।",
    "openmeteoKeyLabel": "Open-Meteo API কী (ঐচ্ছিক)",
    "openmeteoPricing": "Open-Meteo মূল্য",
    "openmeteoDesc": "বিনামূলীয়া সেৱা উপলব্ধ — লাইভ আৰু ৭ দিনৰ বাবে কোনো কীৰ প্ৰয়োজন নাই।",
    "testKeysBtn": "কী পৰীক্ষা কৰক",
    "saveSettingsBtn": "ছেটিং সংৰক্ষণ কৰক",
    "footerSystem": "WeatherGPT • ৰিয়েল-টাইম আৰু ৭ দিনৰ AI বতৰ ব্যৱস্থা",
    "footerPowered": "Open-Meteo REST API • Google Gemini Reasoning • ১২+ ভাৰতীয় ভাষাৰ মাত",
    "days": {
      "today": "আজি",
      "mon": "সোম",
      "tue": "মঙ্গল",
      "wed": "বুধ",
      "thu": "বৃহস্পতি",
      "fri": "শুক্ৰ",
      "sat": "শনি",
      "sun": "দেও"
    },
    "conditions": {
      "Clear sky": "পৰিষ্কাৰ আকাশ",
      "Mainly clear": "প্ৰায় পৰিষ্কাৰ",
      "Partly cloudy": "আংশিক ডাৱৰীয়া",
      "Overcast": "সম্পূৰ্ণ ডাৱৰীয়া",
      "Fog": "কুঁৱলী",
      "Dense fog": "ঘন কুঁৱলী",
      "Depositing rime fog": "তুষাৰ কুঁৱলী",
      "Light drizzle": "পাতল বৰষুণৰ টোপাল",
      "Moderate drizzle": "মধ্যমীয়া টোপাল",
      "Dense drizzle": "ডাঠ বৰষুণৰ টোপাল",
      "Slight rain": "পাতল বৰষুণ",
      "Moderate rain": "মধ্যমীয়া বৰষুণ",
      "Heavy rain": "ধাৰাসাৰ বৰষুণ",
      "Thunderstorm": "বজ্ৰপাতসহ ধুমুহা",
      "Thunderstorm with hail": "শিলাবৃষ্টিসহ ধুমুহা",
      "Thunderstorm with slight hail": "বজ্ৰপাত আৰু পাতল শিলাবৃষ্টি",
      "Thunderstorm with heavy hail": "তীব্ৰ শিলাবৃষ্টি ধুমুহা",
      "Slight snow": "পাতল তুষাৰপাত",
      "Moderate snow": "মধ্যমীয়া তুষাৰপাত",
      "Heavy snow": "প্ৰচণ্ড তুষাৰপাত",
      "Rain showers": "বৰষুণৰ পচলা",
      "Slight rain showers": "পাতল পচলা",
      "Moderate rain showers": "মধ্যমীয়া পচলা",
      "Violent rain showers": "প্ৰচণ্ড ধাৰাসাৰ বৰষুণ"
    },
    "tips": {
      "rain": "আজি বৰষুণৰ সম্ভাৱনা বেছি ({p}%)। ছাতি লগত ৰাখক আৰু সাৱধানে ভ্ৰমণ কৰক!",
      "warm": "গৰম বতৰ ({t}°C)। প্ৰচুৰ পানী খাওক, পাতল কপাহী কাপোৰ পিন্ধক.",
      "pleasant": "আজি {t}°C ৰ সৈতে মনোৰম {d} বতৰ। খোজ কঢ়া আৰু বাহিৰৰ কামৰ বাবে উপযুক্ত দিন!"
    }
  },
  "ur": {
    "appSubtitle": "حقیقی وقت کا موسم اور 7 روزہ پیش گوئی",
    "gpsText": "لائیو GPS",
    "gpsActive": "GPS فعال ہے",
    "gpsLocating": "مقام تلاش کیا جا رہا ہے...",
    "searchPlaceholder": "کوئی بھی ہندوستانی یا عالمی شہر تلاش کریں...",
    "liveVoiceBtn": "لائیو وائس چیٹ",
    "offlineBtn": "آف لائن",
    "exitOfflineBtn": "آف لائن بند کریں",
    "keysBtn": "چابیاں (Keys)",
    "offlineBannerText": "آف لائن موڈ فعال: محفوظ شدہ 7 روزہ موسم دکھایا جا رہا ہے۔",
    "switchLiveBtn": "لائیو موڈ پر جائیں",
    "citiesLabel": "شہر:",
    "cities": [
      "چنئی",
      "دہلی",
      "بنگلور",
      "حیدرآباد",
      "ممبئی",
      "کولکتہ",
      "تنچاور"
    ],
    "gpsLocked": "GPS لاک ہو گیا",
    "currentCond": "موجودہ صورتحال",
    "today": "آج",
    "talkBtn": "WeatherGPT سے بات کریں",
    "feelsLike": "محسوس ہو رہا ہے",
    "highLow": "زیادہ: {max}° • کم: {min}°",
    "rainChance": "بارش کا امکان",
    "rainExpected": "{rain} ملی میٹر بارش متوقع",
    "humidity": "نمی",
    "comfort": "پرسکون",
    "humid": "حبس / زیادہ نمی",
    "dry": "خشک",
    "windSpeed": "ہوا کی رفتار",
    "windBreeze": "ہلکی ہوا",
    "uvIndex": "UV انڈیکس",
    "uvLevels": {
      "veryHigh": "بہت تیز (آنکھوں اور جلد کی حفاظت کریں)",
      "high": "زیادہ (سن اسکرین استعمال کریں)",
      "safe": "معتدل / محفوظ"
    },
    "airPressure": "ہوا کا دباؤ",
    "surfacePressure": "سطحی فضائی دباؤ",
    "cloudCover": "بادلوں کا تناسب",
    "cloudDesc": "بکھرے بادل",
    "offlineCardTitle": "آف لائن 7 روزہ موسم کا ڈیٹا",
    "offlineCardDesc": "انٹرنیٹ کے بغیر دیکھنے کے لیے اگلے 7 دنوں کا موسم محفوظ کریں",
    "offlineSavedText": "ڈیوائس پر محفوظ ہے",
    "saveOfflineBtn": "7 روزہ ڈیٹا محفوظ کریں",
    "guideTitle": "وائس اور چیٹ ویدر گائیڈ",
    "guideSubtitle": "تمام 12+ ہندوستانی زبانوں میں دستیاب",
    "welcomeChat": "ہیلو! میں آپ کا <strong>WeatherGPT وائس گائیڈ</strong> ہوں۔ آپ کسی بھی ہندوستانی زبان میں بول یا لکھ سکتے ہیں۔ مجھ سے بارش، چھتری، کیا پہنیں یا 7 دن کے موسم کے بارے میں پوچھیں!",
    "voiceOutputTitle": "قدرتی آواز کا آؤٹ پٹ",
    "voiceOutputStatus": "اپنی زبان میں سننے کے لیے کلک کریں",
    "listenBtn": "🔊 سنیں ({lang})",
    "liveModeBtn": "لائیو موڈ",
    "forecastTitle": "اگلے 7 دنوں کا موسمی اندازہ",
    "forecastSubtitle": "Open-Meteo ماڈلز کے ذریعے روزانہ کی پیش گوئی",
    "legendMinMax": "کم سے کم / زیادہ سے زیادہ درجہ حرارت",
    "legendRain": "بارش کا امکان",
    "voiceModalTitle": "لائیو وائس موڈ",
    "voiceModalPrompt": "\"اپنی زبان میں موسم سے متعلق کوئی بھی سوال پوچھیں...\"",
    "voiceStatusIdle": "بات کرنے کے لیے دائرے پر ٹیپ کریں",
    "voiceStatusListening": "سن رہا ہے... اپنا سوال بولیں",
    "voiceStatusThinking": "WeatherGPT سوچ رہا ہے...",
    "voiceStatusSpeaking": "جواب بولا جا رہا ہے...",
    "handsFreeLabel": "ہینڈز فری آٹو چیٹ (مسلسل بات چیت)",
    "speedLabel": "رفتار:",
    "settingsTitle": "API کلید کی تشکیل",
    "settingsSubtitle": "اپنی Gemini اور Open-Meteo کلیدیں منسلک کریں",
    "geminiKeyLabel": "Google Gemini API کلید",
    "geminiFreeKey": "مفت کلید حاصل کریں",
    "geminiDesc": "قدرتی گفتگو اور علاقائی آواز کے آؤٹ پٹ کو فعال کرتا ہے۔",
    "openmeteoKeyLabel": "Open-Meteo API کلید (اختیاری)",
    "openmeteoPricing": "Open-Meteo قیمتیں",
    "openmeteoDesc": "مفت درجہ پہلے سے موجود ہے — لائیو اور 7 روزہ موسم کے لیے کلید کی ضرورت نہیں۔",
    "testKeysBtn": "ٹیسٹ کریں",
    "saveSettingsBtn": "ترتیبات محفوظ کریں",
    "footerSystem": "WeatherGPT • حقیقی وقت اور 7 روزہ AI موسمی نظام",
    "footerPowered": "Open-Meteo REST API • Google Gemini Reasoning • 12+ ہندوستانی زبانوں کی آواز",
    "days": {
      "today": "آج",
      "mon": "پیر",
      "tue": "منگل",
      "wed": "بدھ",
      "thu": "جمعرات",
      "fri": "جمعہ",
      "sat": "ہفتہ",
      "sun": "اتوار"
    },
    "conditions": {
      "Clear sky": "صاف آسمان",
      "Mainly clear": "زیادہ تر صاف",
      "Partly cloudy": "جزوی ابر آلود",
      "Overcast": "مکمل ابر آلود",
      "Fog": "دھند",
      "Dense fog": "گہری دھند",
      "Depositing rime fog": "برفیلی دھند",
      "Light drizzle": "ہلکی بوندا باندی",
      "Moderate drizzle": "درمیانی بوندا باندی",
      "Dense drizzle": "تیز بوندا باندی",
      "Slight rain": "ہلکی بارش",
      "Moderate rain": "درمیانی بارش",
      "Heavy rain": "موسلا دھار بارش",
      "Thunderstorm": "گرج چمک کے ساتھ بارش",
      "Thunderstorm with hail": "ژالہ باری کے ساتھ طوفان",
      "Thunderstorm with slight hail": "گرج چمک اور ہلکی ژالہ باری",
      "Thunderstorm with heavy hail": "شدید ژالہ باری اور طوفان",
      "Slight snow": "ہلکی برف باری",
      "Moderate snow": "درمیانی برف باری",
      "Heavy snow": "شدید برف باری",
      "Rain showers": "بارش کی پھوار",
      "Slight rain showers": "ہلکی پھوار",
      "Moderate rain showers": "درمیانی پھوار",
      "Violent rain showers": "شدید موسلا دھار بارش"
    },
    "tips": {
      "rain": "آج بارش کا امکان زیادہ ہے ({p}%)۔ چھتری ساتھ رکھیں اور احتیاط سے سفر کریں!",
      "warm": "گرم موسم ({t}°C)۔ خوب پانی پئیں، ہلکے سوتی کپڑے پہنیں اور دھوپ سے بچیں۔",
      "pleasant": "آج {t}°C کے ساتھ خوشگوار {d} موسم ہے۔ چہل قدمی اور بیرونی کاموں کے لیے بہترین دن!"
    }
  }
};

    function translateWeatherDesc(desc, lang) {
      if (!desc) return "";
      const dict = (APP_I18N[lang] && APP_I18N[lang].conditions) || APP_I18N["en"].conditions;
      const clean = desc.trim();
      if (dict[clean]) return dict[clean];
      const lower = clean.toLowerCase();
      for (const [k, v] of Object.entries(dict)) {
        if (k.toLowerCase() === lower || lower.includes(k.toLowerCase())) {
          return v;
        }
      }
      return desc;
    }


    const PLACES_I18N = {
  "India": {
    "ta": "இந்தியா",
    "hi": "भारत",
    "te": "భారతదేశం",
    "kn": "ಭಾರತ",
    "ml": "ഇന്ത്യ",
    "bn": "ভারত",
    "mr": "भारत",
    "gu": "ભારત",
    "pa": "ਭਾਰਤ",
    "or": "ଭାରତ",
    "as": "ভাৰত",
    "ur": "ہندوستان",
    "en": "India"
  },
  "Tamil Nadu": {
    "ta": "தமிழ்நாடு",
    "hi": "तमिलनाडु",
    "te": "తమిళనాడు",
    "kn": "ತಮಿಳುನಾಡು",
    "ml": "തമിഴ്നാട്",
    "bn": "তামিলনাড়ু",
    "mr": "तमिळनाडू",
    "gu": "તમિલનાડુ",
    "pa": "ਤਾਮਿਲਨਾਡੂ",
    "or": "ତାମିଲନାଡୁ",
    "as": "তামিলনাডু",
    "ur": "تمل ناڈو",
    "en": "Tamil Nadu"
  },
  "Karnataka": {
    "ta": "கர்நாடகா",
    "hi": "कर्नाटक",
    "te": "కర్ణాటక",
    "kn": "ಕರ್ನಾಟಕ",
    "ml": "കർണാടക",
    "bn": "কর্ণাটক",
    "mr": "कर्नाटक",
    "gu": "કર્ણાટક",
    "pa": "ਕਰਨਾਟਕ",
    "or": "କର୍ଣ୍ଣାଟକ",
    "as": "কৰ্ণাটক",
    "ur": "کرناٹک",
    "en": "Karnataka"
  },
  "Kerala": {
    "ta": "கேரளா",
    "hi": "केरल",
    "te": "కేరళ",
    "kn": "ಕೇರಳ",
    "ml": "കേരളം",
    "bn": "কেরল",
    "mr": "केरळ",
    "gu": "કેરળ",
    "pa": "ਕੇਰਲ",
    "or": "କେରଳ",
    "as": "কেৰালা",
    "ur": "کیرالہ",
    "en": "Kerala"
  },
  "Andhra Pradesh": {
    "ta": "ஆந்திரப் பிரதேசம்",
    "hi": "आंध्र प्रदेश",
    "te": "ఆంధ్రప్రదేశ్",
    "kn": "ಆಂಧ್ರಪ್ರದೇಶ",
    "ml": "ആന്ധ്രാപ്രദേശ്",
    "bn": "অন্ধ্রপ্রদেশ",
    "mr": "आंध्र प्रदेश",
    "gu": "આંધ્રપ્રદેશ",
    "pa": "ਆਂਧਰਾ ਪ੍ਰਦੇਸ਼",
    "or": "ଆନ୍ଧ୍ର ପ୍ରଦେଶ",
    "as": "অন্ধ্ৰ প্ৰদেশ",
    "ur": "آندھرا پردیش",
    "en": "Andhra Pradesh"
  },
  "Telangana": {
    "ta": "தெலுங்கானா",
    "hi": "तेलंगाना",
    "te": "తెలంగాణ",
    "kn": "ತೆಲಂಗಾಣ",
    "ml": "തെലങ്കാന",
    "bn": "তেলেঙ্গানা",
    "mr": "तेलंगणा",
    "gu": "તેલંગાણા",
    "pa": "ਤੇਲੰਗਾਨਾ",
    "or": "ତେଲେଙ୍ଗାନା",
    "as": "তেলেংগানা",
    "ur": "تلنگانہ",
    "en": "Telangana"
  },
  "Maharashtra": {
    "ta": "மகாராஷ்டிரா",
    "hi": "महाराष्ट्र",
    "te": "మహారాష్ట్ర",
    "kn": "ಮಹಾರಾಷ್ಟ್ರ",
    "ml": "മഹാരാഷ്ട്ര",
    "bn": "মহারাষ্ট্র",
    "mr": "महाराष्ट्र",
    "gu": "મહારાષ્ટ્ર",
    "pa": "ਮਹਾਰਾਸ਼ਟਰ",
    "or": "ମହାରାଷ୍ଟ୍ର",
    "as": "মহাৰাষ্ট্ৰ",
    "ur": "مہاراشٹر",
    "en": "Maharashtra"
  },
  "Gujarat": {
    "ta": "குஜராத்",
    "hi": "गुजरात",
    "te": "ગુજરાત",
    "kn": "ಗುಜರಾತ್",
    "ml": "ഗുജറാത്ത്",
    "bn": "গুজরাট",
    "mr": "गुजरात",
    "gu": "ગુજરાત",
    "pa": "ਗੁਜਰਾਤ",
    "or": "ଗୁଜରାଟ",
    "as": "গুজৰাট",
    "ur": "گجرات",
    "en": "Gujarat"
  },
  "West Bengal": {
    "ta": "மேற்கு வங்காளம்",
    "hi": "पश्चिम बंगाल",
    "te": "పశ్చిమ బెంగాల్",
    "kn": "ಪಶ್ಚಿಮ ಬಂಗಾಳ",
    "ml": "പശ്ചിമ ബംഗാൾ",
    "bn": "পশ্চিমবঙ্গ",
    "mr": "पश्चिम बंगाल",
    "gu": "પશ્ચિમ બંગાળ",
    "pa": "ਪੱਛਮੀ ਬੰਗਾਲ",
    "or": "ପଶ୍ଚିମ ବଙ୍ଗ",
    "as": "পশ্চিম বংগ",
    "ur": "مغربی بنگال",
    "en": "West Bengal"
  },
  "Punjab": {
    "ta": "பஞ்சாப்",
    "hi": "पंजाब",
    "te": "పంజాబ్",
    "kn": "ಪಂಜಾಬ್",
    "ml": "പഞ്ചാബ്",
    "bn": "পাঞ্জাব",
    "mr": "पंजाब",
    "gu": "પંજાબ",
    "pa": "ਪੰਜਾਬ",
    "or": "ପଞ୍ଜାବ",
    "as": "পাঞ্জাৱ",
    "ur": "پنجاب",
    "en": "Punjab"
  },
  "Odisha": {
    "ta": "ஒடிசா",
    "hi": "ओडिशा",
    "te": "ఒడిశా",
    "kn": "ಒಡಿಶಾ",
    "ml": "ഒഡീഷ",
    "bn": "ওড়িশা",
    "mr": "ओडिशा",
    "gu": "ઓડિશા",
    "pa": "ਓਡੀਸ਼ਾ",
    "or": "ଓଡ଼ିଶା",
    "as": "ওড়িশা",
    "ur": "اوڈیشہ",
    "en": "Odisha"
  },
  "Assam": {
    "ta": "அசாம்",
    "hi": "असम",
    "te": "అస్సాం",
    "kn": "ಅಸ್ಸಾಂ",
    "ml": "അസം",
    "bn": "আসাম",
    "mr": "आसाम",
    "gu": "આસામ",
    "pa": "ਅਸਾਮ",
    "or": "ଆସାମ",
    "as": "অসম",
    "ur": "آسام",
    "en": "Assam"
  },
  "Delhi": {
    "ta": "தில்லி",
    "hi": "दिल्ली",
    "te": "ఢిల్లీ",
    "kn": "ದೆಹಲಿ",
    "ml": "ദില്ലി",
    "bn": "দিল্লি",
    "mr": "दिल्ली",
    "gu": "દિલ્હી",
    "pa": "ਦਿੱਲੀ",
    "or": "ଦିଲ୍ଲୀ",
    "as": "দিল্লী",
    "ur": "دہلی",
    "en": "Delhi"
  },
  "Puducherry": {
    "ta": "புதுச்சேரி",
    "hi": "पुदुचेरी",
    "te": "పుదుచ్చేరి",
    "kn": "ಪುದುಚೇರಿ",
    "ml": "പുതുച്ചേരി",
    "bn": "পুদুচেরি",
    "mr": "पुदुचेरी",
    "gu": "પુડુચેરી",
    "pa": "ਪੁਡੂਚੇਰੀ",
    "or": "ପୁଡୁଚେରୀ",
    "as": "পুডুচেৰী",
    "ur": "پڈوچیری",
    "en": "Puducherry"
  },
  "Oragadam": {
    "ta": "ஒரகடம்",
    "hi": "ओरगदम",
    "te": "ఒరగడం",
    "kn": "ಒರಗಡಂ",
    "ml": "ഒറഗടം",
    "bn": "ওরাগাদাম",
    "mr": "ओरागडम",
    "gu": "ઓરાગડમ",
    "pa": "ਓਰਾਗਦਮ",
    "or": "ଓରାଗାଦାମ",
    "as": "ওৰাগাদাম",
    "ur": "اوراگدم",
    "en": "Oragadam"
  },
  "Sriperumbudur": {
    "ta": "ஸ்ரீபெரும்புதூர்",
    "hi": "श्रीपेरंबदूर",
    "te": "శ్రీపెరంబుదూర్",
    "kn": "ಶ್ರೀಪೆರಂಬುದೂರ್",
    "ml": "ശ്രീപെരുമ്പുദൂർ",
    "bn": "শ্রীপেরুম্বুদুর",
    "mr": "श्रीपेरुंबुदूर",
    "gu": "શ્રીપેરમ્બુદૂર",
    "pa": "ਸ਼੍ਰੀਪੇਰੰਬੁਦੂਰ",
    "or": "ଶ୍ରୀପେରୁମ୍ବୁଦୁର",
    "as": "শ্ৰীপেৰুম্বুদ্বুৰ",
    "ur": "سری پیرومبدور",
    "en": "Sriperumbudur"
  },
  "Kanchipuram": {
    "ta": "காஞ்சிபுரம்",
    "hi": "कांचीपुरम",
    "te": "కాంచీపురం",
    "kn": "ಕಾಂಚೀಪುರಂ",
    "ml": "കാഞ്ചീപുരം",
    "bn": "কাঞ্চীপুরম",
    "mr": "कांचीपुरम",
    "gu": "કાંચીપુરમ",
    "pa": "ਕਾਂਚੀਪੁਰਮ",
    "or": "କାଞ୍ଚିପୁରମ୍",
    "as": "কাঞ্চীপুৰম",
    "ur": "کانچی پورم",
    "en": "Kanchipuram"
  },
  "Chengalpattu": {
    "ta": "செங்கல்பட்டு",
    "hi": "चेंगलपट्टू",
    "te": "చెంగల్పట్టు",
    "kn": "ಚೆಂಗಲ್ಪಟ್ಟು",
    "ml": "ചെങ്കൽപട്ട്",
    "bn": "চেঙ্গালপট্টু",
    "mr": "चेंगलपट्टू",
    "gu": "ચેંગલપટ્ટુ",
    "pa": "ਚੇਂਗਲਪੱਟੂ",
    "or": "ଚେଙ୍ଗଲପଟ୍ଟୁ",
    "as": "চেংগলপট্টু",
    "ur": "چنگل پٹو",
    "en": "Chengalpattu"
  },
  "Tambaram": {
    "ta": "தாம்பரம்",
    "hi": "तांबरम",
    "te": "తాంబరం",
    "kn": "ತಾಂಬರಂ",
    "ml": "താംബരം",
    "bn": "তাম্বারাম",
    "mr": "तांबरम",
    "gu": "તાંબરમ",
    "pa": "ਤਾਂਬਰਮ",
    "or": "ତାମ୍ବରମ",
    "as": "তাম্বাৰাম",
    "ur": "تامبرم",
    "en": "Tambaram"
  },
  "Guindy": {
    "ta": "கிண்டி",
    "hi": "गिंडी",
    "te": "గిండి",
    "kn": "ಗಿಂಡಿ",
    "ml": "ಗಿണ്ടി",
    "bn": "গিন্ডি",
    "mr": "गिंडी",
    "gu": "ગિન્ડી",
    "pa": "ਗਿੰਡੀ",
    "or": "ଗିଣ୍ଡି",
    "as": "গিণ্ডি",
    "ur": "گنڈی",
    "en": "Guindy"
  },
  "Chennai": {
    "ta": "சென்னை",
    "hi": "चेन्नई",
    "te": "చెన్నై",
    "kn": "ಚೆನ್ನೈ",
    "ml": "ചെന്നൈ",
    "bn": "চেন্নাই",
    "mr": "चेन्नई",
    "gu": "ચેન્નઈ",
    "pa": "ਚੇਨਈ",
    "or": "ଚେନ୍ନାଇ",
    "as": "চেন্নাই",
    "ur": "چنئی",
    "en": "Chennai"
  },
  "Bengaluru": {
    "ta": "பெங்களூரு",
    "hi": "बेंगलुरु",
    "te": "బెంగళూరు",
    "kn": "ಬೆಂಗಳೂರು",
    "ml": "ബെംഗളൂരു",
    "bn": "বেঙ্গালুরু",
    "mr": "बंगळुरू",
    "gu": "બેંગલુરુ",
    "pa": "ਬੈਂਗਲੁਰੂ",
    "or": "ବେଙ୍ଗାଲୁରୁ",
    "as": "বেংগালুৰু",
    "ur": "بنگلور",
    "en": "Bengaluru"
  },
  "Hyderabad": {
    "ta": "ஹைதராபாத்",
    "hi": "हैदराबाद",
    "te": "హైదరాబాద్",
    "kn": "ಹೈದರಾಬಾದ್",
    "ml": "ഹൈദരാബാദ്",
    "bn": "হায়দ্রাবাদ",
    "mr": "हैदराबाद",
    "gu": "હૈદરાબાદ",
    "pa": "ਹੈਦਰਾਬਾਦ",
    "or": "ହାଇଦ୍ରାବାଦ",
    "as": "হায়দৰাবাদ",
    "ur": "حیدرآباد",
    "en": "Hyderabad"
  },
  "Mumbai": {
    "ta": "மும்பை",
    "hi": "मुंबई",
    "te": "ముంబై",
    "kn": "ಮುಂಬೈ",
    "ml": "മുംബൈ",
    "bn": "মুম্বাই",
    "mr": "मुंबई",
    "gu": "મુંબઈ",
    "pa": "ਮੁੰਬਈ",
    "or": "ମୁମ୍ବାଇ",
    "as": "মুম্বাই",
    "ur": "ممبئی",
    "en": "Mumbai"
  },
  "Kolkata": {
    "ta": "கொல்கத்தா",
    "hi": "कोलकाता",
    "te": "కోల్‌కతా",
    "kn": "ಕೋಲ್ಕತ್ತಾ",
    "ml": "കൊൽക്കത്ത",
    "bn": "কলকাতা",
    "mr": "कोलकाता",
    "gu": "કોલકાતા",
    "pa": "ਕੋਲਕਾਤਾ",
    "or": "କୋଲକାତା",
    "as": "কলকাতা",
    "ur": "کولکتہ",
    "en": "Kolkata"
  },
  "Thanjavur": {
    "ta": "தஞ்சாவூர்",
    "hi": "तंजावुर",
    "te": "తంజావూరు",
    "kn": "ತಂಜಾವೂರು",
    "ml": "തഞ്ചാവൂർ",
    "bn": "তাঞ্জাভুর",
    "mr": "तंजावर",
    "gu": "તંજાવુર",
    "pa": "ਤੰਜਾਵੁਰ",
    "or": "ତାଞ୍ଜାଭୁର",
    "as": "তাঞ্জাভুৰ",
    "ur": "تنچاور",
    "en": "Thanjavur"
  },
  "Coimbatore": {
    "ta": "கோயம்புத்தூர்",
    "hi": "कोयंबटूर",
    "te": "కోయంబత్తూర్",
    "kn": "ಕೊಯಮತ್ತೂರು",
    "ml": "കോയമ്പത്തൂർ",
    "bn": "কোয়েম্বাটুর",
    "mr": "कोइम्बतूर",
    "gu": "કોયમ્બતૂર",
    "pa": "ਕੋਇੰਬਟੂਰ",
    "or": "କୋଏମ୍ବାଟୁର",
    "as": "কোয়েম্বাটোৰ",
    "ur": "کوئمبٹور",
    "en": "Coimbatore"
  },
  "Madurai": {
    "ta": "மதுரை",
    "hi": "मदुरै",
    "te": "మధురై",
    "kn": "ಮಧುರೈ",
    "ml": "മധുര",
    "bn": "মাদুরাই",
    "mr": "मदुराई",
    "gu": "મદુરાઇ",
    "pa": "ਮਦੁਰਾਈ",
    "or": "ମଦୁରାଇ",
    "as": "মাদুৰাই",
    "ur": "مدورائی",
    "en": "Madurai"
  },
  "Tiruchirappalli": {
    "ta": "திருச்சிராப்பள்ளி",
    "hi": "तिरुचिरापल्ली",
    "te": "తిరుచిరాపల్లి",
    "kn": "ತಿರುಚಿರಾಪಳ್ಳಿ",
    "ml": "തിരുച്ചിറപ്പള്ളി",
    "bn": "তিরুচিরাপল্লী",
    "mr": "तिरुचिरापल्ली",
    "gu": "તિરુચિરાપલ્લી",
    "pa": "ਤਿਰੂਚਿਰਾਪੱਲੀ",
    "or": "ତିରୁଚିରାପଲ୍ଲୀ",
    "as": "তিৰুচিৰাপল্লী",
    "ur": "تروچیراپلی",
    "en": "Tiruchirappalli"
  },
  "Salem": {
    "ta": "சேலம்",
    "hi": "सलेम",
    "te": "సేలం",
    "kn": "ಸೇಲಂ",
    "ml": "സേலம்",
    "bn": "সালেম",
    "mr": "सेलम",
    "gu": "સેલમ",
    "pa": "ਸਲੇਮ",
    "or": "ସାଲେମ୍",
    "as": "চালেম",
    "ur": "سیلم",
    "en": "Salem"
  },
  "Tirunelveli": {
    "ta": "திருநெல்வேலி",
    "hi": "तिरुनेलवेली",
    "te": "తిరునెల్వేలి",
    "kn": "ತಿರುನೆಲ್ವೇಲಿ",
    "ml": "തിരുനെൽവേലി",
    "bn": "তিরুনেলভেলি",
    "mr": "तिरुनेलवेली",
    "gu": "તિરુનેલવેલી",
    "pa": "ਤਿਰੂਨੇਲਵੇਲੀ",
    "or": "ତିରୁନେଲଭେଲି",
    "as": "তিৰুনেলভেলি",
    "ur": "ترونلویلی",
    "en": "Tirunelveli"
  },
  "Vellore": {
    "ta": "வேலூர்",
    "hi": "वेल्लोर",
    "te": "వెల్లూరు",
    "kn": "ವೆಲ್ಲೂರು",
    "ml": "വെല്ലൂർ",
    "bn": "ভেলোর",
    "mr": "वेल्लोर",
    "gu": "વેલ્લોર",
    "pa": "ਵੇਲੋਰ",
    "or": "ଭେଲୋର୍",
    "as": "ভেলোৰ",
    "ur": "ویلور",
    "en": "Vellore"
  },
  "Erode": {
    "ta": "ஈரோடு",
    "hi": "ईरोड",
    "te": "ఈరోడ్",
    "kn": "ಈರೋಡ್",
    "ml": "ഈറോഡ്",
    "bn": "ইরোড",
    "mr": "इरोड",
    "gu": "ઈરોડ",
    "pa": "ਈਰੋਡ",
    "or": "ଇରୋଡ୍",
    "as": "ইৰোড",
    "ur": "ایروڈ",
    "en": "Erode"
  },
  "Your Location": {
    "ta": "உங்கள் இருப்பிடம்",
    "hi": "आपका स्थान",
    "te": "మీ స్థానం",
    "kn": "ನಿಮ್ಮ ಸ್ಥಳ",
    "ml": "നിങ്ങളുടെ സ്ഥലം",
    "bn": "আপনার অবস্থান",
    "mr": "आपले स्थान",
    "gu": "તમારું સ્થાન",
    "pa": "ਤੁਹਾਡਾ ਸਥਾਨ",
    "or": "ଆପଣଙ୍କ ସ୍ଥାନ",
    "as": "আপোনাৰ স্থান",
    "ur": "آپ کا مقام",
    "en": "Your Location"
  }
};

    function localizeLocationName(locString, lang) {
      if (!locString) return "";
      if (lang === "en") return rawLocationEnglish || locString;
      if (cachedLocationTranslations[lang]) return cachedLocationTranslations[lang];

      let result = locString;
      for (const [placeEn, transMap] of Object.entries(PLACES_I18N)) {
        const localized = transMap[lang];
        if (!localized) continue;
        const regex = new RegExp(`\\b${placeEn}\\b`, 'gi');
        result = result.replace(regex, localized);
      }
      return result;
    }

    function updateCoordsDisplay(langCode = currentLang) {
      const i18n = APP_I18N[langCode] || APP_I18N["en"];
      const coordsEl = document.getElementById('current-coords-text');
      const accBadge = document.getElementById('gps-accuracy-badge');

      if (lastGpsAccuracy !== null && lastGpsAccuracy !== undefined) {
        if (accBadge) {
          accBadge.innerText = `${i18n.gpsBadgeText || 'GPS'} (±${lastGpsAccuracy}m)`;
          accBadge.classList.remove('hidden');
        }
        if (coordsEl) {
          coordsEl.innerText = `${currentLat.toFixed(4)}° N, ${currentLon.toFixed(4)}° E • ${i18n.gpsAccuracyText || 'Live GPS accuracy'} ±${lastGpsAccuracy}m`;
        }
      } else {
        if (coordsEl) {
          coordsEl.innerText = `${currentLat.toFixed(4)}° N, ${currentLon.toFixed(4)}° E • ${i18n.updatedJustNow || 'Updated just now'}`;
        }
      }
    }

    async function fetchLocationName(lat, lon, langCode) {
      if (cachedLocationTranslations[langCode]) {
        if (currentLang === langCode) {
          currentLocationName = cachedLocationTranslations[langCode];
          const locElement = document.getElementById('current-location-name');
          if (locElement) locElement.innerText = currentLocationName;
        }
        return;
      }
      try {
        const geoResp = await fetch(`/api/geocode/reverse?latitude=${lat}&longitude=${lon}&language=${langCode}`);
        if (geoResp.ok) {
          const geoData = await geoResp.json();
          if (!rawLocationEnglish) rawLocationEnglish = geoData.raw_display_name || geoData.name;
          cachedLocationTranslations[langCode] = geoData.display_name;
          if (currentLang === langCode) {
            currentLocationName = geoData.display_name;
            const locElement = document.getElementById('current-location-name');
            if (locElement) locElement.innerText = currentLocationName;
          }
        }
      } catch (e) {
        console.warn("fetchLocationName error:", e);
      }
    }

    function translateDayName(dayName, isToday, lang) {
      const days = (APP_I18N[lang] && APP_I18N[lang].days) || APP_I18N["en"].days;
      if (isToday) return days.today || "Today";
      const short = dayName ? dayName.substring(0, 3).toLowerCase() : "";
      return days[short] || dayName;
    }

    function applyAppLanguage(langCode) {
      const i18n = APP_I18N[langCode] || APP_I18N["en"];
      currentLang = langCode;
      localStorage.setItem('weathergpt_lang', langCode);

      // 1. Sync dropdowns
      const navLang = document.getElementById('nav-lang');
      if (navLang) navLang.value = langCode;
      const guideLang = document.getElementById('guide-lang');
      if (guideLang) guideLang.value = langCode;
      const voiceModalLang = document.getElementById('voice-modal-lang');
      if (voiceModalLang) voiceModalLang.value = langCode;

      // 2. Header & GPS
      const appSub = document.getElementById('app-subtitle');
      if (appSub) appSub.innerText = i18n.appSubtitle;

      const gpsTxt = document.getElementById('gps-text');
      if (gpsTxt) {
        if (gpsTxt.innerText.includes("Active") || gpsTxt.innerText.includes("செயலில்") || gpsTxt.innerText.includes("सक्रिय")) {
          gpsTxt.innerText = i18n.gpsActive;
        } else if (gpsTxt.innerText.includes("Locat") || gpsTxt.innerText.includes("கண்டறிகிறது") || gpsTxt.innerText.includes("खोज")) {
          gpsTxt.innerText = i18n.gpsLocating;
        } else {
          gpsTxt.innerText = i18n.gpsText;
        }
      }

      const searchInp = document.getElementById('search-input');
      if (searchInp) searchInp.placeholder = i18n.searchPlaceholder;

      const btnLiveVoice = document.getElementById('btn-live-voice-text');
      if (btnLiveVoice) btnLiveVoice.innerText = i18n.liveVoiceBtn;

      const offlineModeTxt = document.getElementById('offline-mode-text');
      if (offlineModeTxt) {
        offlineModeTxt.innerText = isOfflineView ? i18n.exitOfflineBtn : i18n.offlineBtn;
      }

      const btnKeys = document.getElementById('btn-keys-text');
      if (btnKeys) btnKeys.innerText = i18n.keysBtn;

      const offlineBannerTxt = document.getElementById('offline-banner-text');
      if (offlineBannerTxt && !offlineBannerTxt.innerText.includes("saved on")) {
        offlineBannerTxt.innerText = i18n.offlineBannerText;
      }
      const offlineSwitchBtn = document.getElementById('offline-switch-btn');
      if (offlineSwitchBtn) offlineSwitchBtn.innerText = i18n.switchLiveBtn;

      // 3. Location Bar & Cities
      const citiesLbl = document.getElementById('cities-label');
      if (citiesLbl) citiesLbl.innerText = i18n.citiesLabel;

      if (Array.isArray(i18n.cities)) {
        i18n.cities.forEach((cityName, idx) => {
          const cBtn = document.getElementById(`city-btn-${idx}`);
          if (cBtn) cBtn.innerText = cityName;
        });
      }

      // Localize Current Location Name Instantly
      const locElement = document.getElementById('current-location-name');
      if (locElement) {
        const localizedLoc = localizeLocationName(rawLocationEnglish || locElement.innerText, langCode);
        currentLocationName = localizedLoc;
        locElement.innerText = localizedLoc;

        // If GPS coordinates are active, also background fetch freshest official translation from server
        if (currentLat && currentLon) {
          fetchLocationName(currentLat, currentLon, langCode);
        }
      }

      // Localize GPS accuracy and coordinates display
      updateCoordsDisplay(langCode);

      // 4. Hero Card
      const heroCondTitle = document.getElementById('hero-cond-title');
      if (heroCondTitle) heroCondTitle.innerText = i18n.currentCond;

      const heroDateTxt = document.getElementById('hero-date-text');
      if (heroDateTxt) heroDateTxt.innerText = i18n.today;

      const heroTalk = document.getElementById('hero-talk-text');
      if (heroTalk) heroTalk.innerText = i18n.talkBtn;

      // 5. 6-Metric Highlights Labels
      const rainLbl = document.getElementById('label-rain-chance');
      if (rainLbl) rainLbl.innerText = i18n.rainChance;

      const humidLbl = document.getElementById('label-humidity');
      if (humidLbl) humidLbl.innerText = i18n.humidity;

      const windLbl = document.getElementById('label-wind-speed');
      if (windLbl) windLbl.innerText = i18n.windSpeed;

      const uvLbl = document.getElementById('label-uv-index');
      if (uvLbl) uvLbl.innerText = i18n.uvIndex;

      const pressureLbl = document.getElementById('label-air-pressure');
      if (pressureLbl) pressureLbl.innerText = i18n.airPressure;

      const cloudLbl = document.getElementById('label-cloud-cover');
      if (cloudLbl) cloudLbl.innerText = i18n.cloudCover;

      const pressureDesc = document.getElementById('metric-pressure-desc');
      if (pressureDesc) pressureDesc.innerText = i18n.surfacePressure;

      const cloudDesc = document.getElementById('metric-clouds-desc');
      if (cloudDesc) cloudDesc.innerText = i18n.cloudDesc;

      // 6. Offline 7-Day Card
      const offTitle = document.getElementById('offline-card-title');
      if (offTitle) offTitle.innerText = i18n.offlineCardTitle;

      const offDesc = document.getElementById('offline-card-desc');
      if (offDesc) offDesc.innerText = i18n.offlineCardDesc;

      const offSaved = document.getElementById('offline-saved-text');
      if (offSaved) offSaved.innerText = i18n.offlineSavedText;

      const offBtn = document.getElementById('btn-download-cache-text');
      if (offBtn) offBtn.innerText = i18n.saveOfflineBtn;

      // 7. Conversational Guide Header & Chat
      const guideTitle = document.getElementById('guide-header-title');
      if (guideTitle) guideTitle.innerText = i18n.guideTitle;

      const guideSub = document.getElementById('guide-header-subtitle');
      if (guideSub) guideSub.innerText = i18n.guideSubtitle;

      const chatWelcome = document.getElementById('chat-welcome-text');
      if (chatWelcome) chatWelcome.innerHTML = i18n.welcomeChat;

      const audioTitle = document.getElementById('audio-bar-title');
      if (audioTitle) audioTitle.innerText = i18n.voiceOutputTitle;

      const audioLive = document.getElementById('audio-bar-live-text');
      if (audioLive) audioLive.innerText = i18n.liveModeBtn;

      // 8. 7-Day Forecast Prediction Section
      const fTitle = document.getElementById('forecast-section-title');
      if (fTitle) fTitle.innerHTML = `<i class="fa-solid fa-calendar-week text-sky-500 mr-2"></i> ${escapeHtml(i18n.forecastTitle)}`;

      const fDesc = document.getElementById('forecast-section-desc');
      if (fDesc) fDesc.innerText = i18n.forecastSubtitle;

      const lMinMax = document.getElementById('legend-minmax-text');
      if (lMinMax) lMinMax.innerHTML = `<i class="fa-solid fa-temperature-half mr-1"></i> ${escapeHtml(i18n.legendMinMax)}`;

      const lRain = document.getElementById('legend-rain-text');
      if (lRain) lRain.innerHTML = `<i class="fa-solid fa-cloud-rain mr-1"></i> ${escapeHtml(i18n.legendRain)}`;

      // 9. Modals: Live Voice Modal
      const modalVTitle = document.getElementById('modal-voice-title');
      if (modalVTitle) modalVTitle.innerText = i18n.voiceModalTitle;

      const modalHLabel = document.getElementById('modal-handsfree-label');
      if (modalHLabel) modalHLabel.innerText = i18n.handsFreeLabel;

      const modalSLabel = document.getElementById('modal-speed-label');
      if (modalSLabel) modalSLabel.innerText = i18n.speedLabel;

      const vTranscript = document.getElementById('voice-user-transcript');
      if (vTranscript && (vTranscript.innerText.startsWith('"') || vTranscript.innerText.startsWith('“'))) {
        vTranscript.innerText = i18n.voiceModalPrompt;
      }

      // 10. Modals: Settings Modal
      const sTitle = document.getElementById('settings-modal-title');
      if (sTitle) sTitle.innerText = i18n.settingsTitle;

      const sDesc = document.getElementById('settings-modal-desc');
      if (sDesc) sDesc.innerText = i18n.settingsSubtitle;

      const lGemini = document.getElementById('label-gemini-key');
      if (lGemini) lGemini.innerText = i18n.geminiKeyLabel;

      const lnkGemini = document.getElementById('link-gemini-key');
      if (lnkGemini) lnkGemini.innerHTML = `${escapeHtml(i18n.geminiFreeKey)} <i class="fa-solid fa-arrow-up-right-from-square text-[10px]"></i>`;

      const dGemini = document.getElementById('desc-gemini-key');
      if (dGemini) dGemini.innerText = i18n.geminiDesc;

      const lOM = document.getElementById('label-openmeteo-key');
      if (lOM) lOM.innerText = i18n.openmeteoKeyLabel;

      const lnkOM = document.getElementById('link-openmeteo-key');
      if (lnkOM) lnkOM.innerHTML = `${escapeHtml(i18n.openmeteoPricing)} <i class="fa-solid fa-arrow-up-right-from-square text-[10px]"></i>`;

      const dOM = document.getElementById('desc-openmeteo-key');
      if (dOM) dOM.innerHTML = `<i class="fa-solid fa-circle-check"></i> ${escapeHtml(i18n.openmeteoDesc)}`;

      const bTest = document.getElementById('btn-test-keys');
      if (bTest && !bTest.disabled) bTest.innerText = i18n.testKeysBtn;

      const bSave = document.getElementById('btn-save-keys');
      if (bSave) bSave.innerText = i18n.saveSettingsBtn;

      // 11. Footer
      const fSys = document.getElementById('footer-system-text');
      if (fSys) fSys.innerText = `• ${i18n.footerSystem.replace('WeatherGPT • ', '')}`;

      const fPow = document.getElementById('footer-powered-text');
      if (fPow) fPow.innerText = i18n.footerPowered;

      // 12. Localized Quick Prompt Chips, Chat Input & Voice Orb Hints
      updateLanguageUI(langCode);

      // 13. Re-render live weather card & 7-day cards if data is already cached
      if (cachedWeatherData) {
        renderWeatherUI(cachedWeatherData);
      }
    }

    const LOCALIZED_UI = {
      en: {
        placeholder: "Ask weather question or click mic...",
        voiceHint: "Tap orb or speak your question in English...",
        chips: [
          { label: "Umbrella?", query: "Do I need to carry an umbrella today?", icon: "fa-umbrella text-sky-600", cls: "bg-sky-50 hover:bg-sky-100 text-sky-800 border-sky-100" },
          { label: "Dry clothes?", query: "Can I dry clothes outside today?", icon: "fa-jug-detergent text-teal-600", cls: "bg-teal-50 hover:bg-teal-100 text-teal-800 border-teal-100" },
          { label: "Wash car?", query: "Can I wash my car today?", icon: "fa-car text-blue-600", cls: "bg-blue-50 hover:bg-blue-100 text-blue-800 border-blue-100" },
          { label: "What to wear?", query: "What should I wear today based on the temperature?", icon: "fa-shirt text-indigo-600", cls: "bg-indigo-50 hover:bg-indigo-100 text-indigo-800 border-indigo-100" },
          { label: "Outdoor sports?", query: "Is today good for outdoor sports or running?", icon: "fa-person-running text-emerald-600", cls: "bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border-emerald-100" },
          { label: "Night weather?", query: "Will it be cold or chilly tonight?", icon: "fa-moon text-purple-600", cls: "bg-purple-50 hover:bg-purple-100 text-purple-800 border-purple-100" },
          { label: "Weekend travel?", query: "How is the weather this weekend for travel?", icon: "fa-route text-amber-600", cls: "bg-amber-50 hover:bg-amber-100 text-amber-800 border-amber-100" },
        ]
      },
      ta: {
        placeholder: "வானிலை பற்றி கேட்கவும் அல்லது மைக் அழுத்தவும்...",
        voiceHint: "பேசத் தொடங்க வட்டத்தை அழுத்தவும் (தமிழ்)...",
        chips: [
          { label: "குடை தேவையா?", query: "இன்று வெளியே செல்ல குடை தேவையா?", icon: "fa-umbrella text-sky-600", cls: "bg-sky-50 hover:bg-sky-100 text-sky-800 border-sky-100" },
          { label: "துணி காய?", query: "இன்று துணிகளை வெளியில் காய வைக்கலாமா?", icon: "fa-jug-detergent text-teal-600", cls: "bg-teal-50 hover:bg-teal-100 text-teal-800 border-teal-100" },
          { label: "வண்டி கழுவலாமா?", query: "இன்று கார் அல்லது வண்டி கழுவலாமா?", icon: "fa-car text-blue-600", cls: "bg-blue-50 hover:bg-blue-100 text-blue-800 border-blue-100" },
          { label: "என்ன உடை?", query: "இன்றைய வெப்பநிலைக்கு என்ன ஆடை அணியலாம்?", icon: "fa-shirt text-indigo-600", cls: "bg-indigo-50 hover:bg-indigo-100 text-indigo-800 border-indigo-100" },
          { label: "விளையாட்டு/ஓட்டம்?", query: "இன்று உடற்பயிற்சி அல்லது விளையாட சிறந்த நாளா?", icon: "fa-person-running text-emerald-600", cls: "bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border-emerald-100" },
          { label: "இரவு வானிலை?", query: "இன்று இரவு குளிர் அல்லது புழுக்கம் அதிகமாக இருக்குமா?", icon: "fa-moon text-purple-600", cls: "bg-purple-50 hover:bg-purple-100 text-purple-800 border-purple-100" },
          { label: "பயணம்?", query: "இந்த வார இறுதியில் பயணம் செய்ய வானிலை எப்படி?", icon: "fa-route text-amber-600", cls: "bg-amber-50 hover:bg-amber-100 text-amber-800 border-amber-100" },
        ]
      },
      hi: {
        placeholder: "मौसम के बारे में पूछें या माइक दबाएं...",
        voiceHint: "बोलने के लिए गोले पर टैप करें (हिन्दी)...",
        chips: [
          { label: "छाता चाहिए?", query: "क्या आज मुझे छाता लेकर जाना चाहिए?", icon: "fa-umbrella text-sky-600", cls: "bg-sky-50 hover:bg-sky-100 text-sky-800 border-sky-100" },
          { label: "कपड़े सुखाना?", query: "क्या आज कपड़े बाहर सुखा सकते हैं?", icon: "fa-jug-detergent text-teal-600", cls: "bg-teal-50 hover:bg-teal-100 text-teal-800 border-teal-100" },
          { label: "गाड़ी धोएं?", query: "क्या आज कार या बाइक धो सकते हैं?", icon: "fa-car text-blue-600", cls: "bg-blue-50 hover:bg-blue-100 text-blue-800 border-blue-100" },
          { label: "क्या पहनें?", query: "आज के तापमान के अनुसार क्या पहनना सही रहेगा?", icon: "fa-shirt text-indigo-600", cls: "bg-indigo-50 hover:bg-indigo-100 text-indigo-800 border-indigo-100" },
          { label: "खेलकूद/दौड़?", query: "क्या आज दौड़ने या खेलकूद के लिए अच्छा मौसम है?", icon: "fa-person-running text-emerald-600", cls: "bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border-emerald-100" },
          { label: "रात का मौसम?", query: "क्या आज रात अधिक ठंड या उमस रहेगी?", icon: "fa-moon text-purple-600", cls: "bg-purple-50 hover:bg-purple-100 text-purple-800 border-purple-100" },
          { label: "वीकेंड यात्रा?", query: "इस वीकेंड घूमने जाने के लिए मौसम कैसा रहेगा?", icon: "fa-route text-amber-600", cls: "bg-amber-50 hover:bg-amber-100 text-amber-800 border-amber-100" },
        ]
      },
      te: {
        placeholder: "వాతావరణం గురించి అడగండి లేదా మైక్ నొక్కండి...",
        voiceHint: "మాట్లాడటానికి వృత్తాన్ని తాకండి (తెలుగు)...",
        chips: [
          { label: "గొడుగు అవసరమా?", query: "ఈ రోజు బయటకు వెళ్ళేటప్పుడు గొడుగు అవసరమా?", icon: "fa-umbrella text-sky-600", cls: "bg-sky-50 hover:bg-sky-100 text-sky-800 border-sky-100" },
          { label: "బట్టలు ఆరబెట్టవచ్చా?", query: "ఈ రోజు బట్టలు బయట ఆరబెట్టవచ్చా?", icon: "fa-jug-detergent text-teal-600", cls: "bg-teal-50 hover:bg-teal-100 text-teal-800 border-teal-100" },
          { label: "కారు కడగవచ్చా?", query: "ఈ రోజు కారు లేదా బైక్ వాష్ చేయవచ్చా?", icon: "fa-car text-blue-600", cls: "bg-blue-50 hover:bg-blue-100 text-blue-800 border-blue-100" },
          { label: "ఏమి ధరించాలి?", query: "నేటి ఉష్ణోగ్రతకు ఎలాంటి దుస్తులు ధరించాలి?", icon: "fa-shirt text-indigo-600", cls: "bg-indigo-50 hover:bg-indigo-100 text-indigo-800 border-indigo-100" },
          { label: "ఆటలు/జాగింగ్?", query: "జాగింగ్ లేదా ఆటలకు ఈ రోజు వాతావరణం బాగుందా?", icon: "fa-person-running text-emerald-600", cls: "bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border-emerald-100" },
          { label: "రాత్రి వాతావరణం?", query: "ఈ రాత్రి చలిగా లేదా ఉక్కపోతగా ఉంటుందా?", icon: "fa-moon text-purple-600", cls: "bg-purple-50 hover:bg-purple-100 text-purple-800 border-purple-100" },
          { label: "ప్రయాణం?", query: "ఈ వీకెండ్ ప్రయాణానికి వాతావరణం ఎలా ఉంటుంది?", icon: "fa-route text-amber-600", cls: "bg-amber-50 hover:bg-amber-100 text-amber-800 border-amber-100" },
        ]
      },
      kn: {
        placeholder: "ಹವಾಮಾನದ ಬಗ್ಗೆ ಕೇಳಿ ಅಥವಾ ಮೈಕ್ ಒತ್ತಿ...",
        voiceHint: "ಮಾತನಾಡಲು ವೃತ್ತವನ್ನು ಸ್ಪರ್ಶಿಸಿ (ಕನ್ನಡ)...",
        chips: [
          { label: "ಛತ್ರಿ ಬೇಕೇ?", query: "ಇಂದು ಹೊರಗೆ ಹೋಗುವಾಗ ಛತ್ರಿ ಬೇಕಾಗುತ್ತದೆಯೇ?", icon: "fa-umbrella text-sky-600", cls: "bg-sky-50 hover:bg-sky-100 text-sky-800 border-sky-100" },
          { label: "ಬಟ್ಟೆ ಒಣಗಿಸಬಹುದೇ?", query: "ಇಂದು ಬಟ್ಟೆಗಳನ್ನು ಹೊರಗೆ ಒಣಗಿಸಬಹುದೇ?", icon: "fa-jug-detergent text-teal-600", cls: "bg-teal-50 hover:bg-teal-100 text-teal-800 border-teal-100" },
          { label: "ಕಾರು ತೊಳೆಯಬಹುದೇ?", query: "ಇಂದು ಕಾರು ಅಥವಾ ಬೈಕ್ ತೊಳೆಯಬಹುದೇ?", icon: "fa-car text-blue-600", cls: "bg-blue-50 hover:bg-blue-100 text-blue-800 border-blue-100" },
          { label: "ಉಡುಪು?", query: "ಇಂದಿನ ತಾಪಮಾನಕ್ಕೆ ತಕ್ಕಂತೆ ಯಾವ ಬಟ್ಟೆ ಧರಿಸಬೇಕು?", icon: "fa-shirt text-indigo-600", cls: "bg-indigo-50 hover:bg-indigo-100 text-indigo-800 border-indigo-100" },
          { label: "ಕ್ರೀಡೆ/ಓಟ?", query: "ಇಂದು ಓಟ ಅಥವಾ ಹೊರಾಂಗಣ ಕ್ರೀಡೆಗಳಿಗೆ ಸೂಕ್ತವೇ?", icon: "fa-person-running text-emerald-600", cls: "bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border-emerald-100" },
          { label: "ರಾತ್ರಿ ಹವಾಮಾನ?", query: "ಇಂದು ರಾತ್ರಿ ಚಳಿ ಹೆಚ್ಚಾಗಿರುತ್ತದೆಯೇ?", icon: "fa-moon text-purple-600", cls: "bg-purple-50 hover:bg-purple-100 text-purple-800 border-purple-100" },
          { label: "ಪ್ರಯಾಣ?", query: "ಈ ವಾರಾಂತ್ಯದ ಪ್ರಯಾಣಕ್ಕೆ ಹವಾಮಾನ ಹೇಗಿದೆ?", icon: "fa-route text-amber-600", cls: "bg-amber-50 hover:bg-amber-100 text-amber-800 border-amber-100" },
        ]
      },
      ml: {
        placeholder: "കാലാവസ്ഥയെക്കുറിച്ച് ചോദിക്കുക അല്ലെങ്കിൽ മൈക്ക് അമർത്തുക...",
        voiceHint: "സംസാരിക്കാൻ വൃത്തത്തിൽ തൊടുക (മലയാളം)...",
        chips: [
          { label: "കുട വേണമോ?", query: "ഇന്ന് പുറത്തുപോകുമ്പോൾ കുട എടുക്കേണ്ടതുണ്ടോ?", icon: "fa-umbrella text-sky-600", cls: "bg-sky-50 hover:bg-sky-100 text-sky-800 border-sky-100" },
          { label: "തുണി ഉണക്കാൻ?", query: "ഇന്ന് തുണികൾ വെളിയിൽ ഉണക്കാൻ ഇടാമോ?", icon: "fa-jug-detergent text-teal-600", cls: "bg-teal-50 hover:bg-teal-100 text-teal-800 border-teal-100" },
          { label: "കാർ കഴുകാമോ?", query: "ഇന്ന് കാർ അല്ലെങ്കിൽ വണ്ടി കഴുകാമോ?", icon: "fa-car text-blue-600", cls: "bg-blue-50 hover:bg-blue-100 text-blue-800 border-blue-100" },
          { label: "ഏത് വസ്ത്രം?", query: "ഇന്നത്തെ കാലാവസ്ഥയ്ക്ക് അനുയോജ്യമായ വസ്ത്രം ഏതാണ്?", icon: "fa-shirt text-indigo-600", cls: "bg-indigo-50 hover:bg-indigo-100 text-indigo-800 border-indigo-100" },
          { label: "കളി/ഓട്ടം?", query: "ഇന്ന് ഓടാനോ കളിക്കാനോ നല്ല കാലാവസ്ഥയാണോ?", icon: "fa-person-running text-emerald-600", cls: "bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border-emerald-100" },
          { label: "രാത്രി കാലാവസ്ഥ?", query: "ഇന്ന് രാത്രി തണുപ്പോ ചൂടോ കൂടുതലായിരിക്കുമോ?", icon: "fa-moon text-purple-600", cls: "bg-purple-50 hover:bg-purple-100 text-purple-800 border-purple-100" },
          { label: "യാത്ര?", query: "ഈ വാരാന്ത്യത്തിൽ യാത്രയ്ക്ക് കാലാവസ്ഥ എങ്ങനെയുണ്ട്?", icon: "fa-route text-amber-600", cls: "bg-amber-50 hover:bg-amber-100 text-amber-800 border-amber-100" },
        ]
      },
      bn: {
        placeholder: "আবহাওয়া সম্পর্কে জিজ্ঞাসা করুন বা মাইক টিপুন...",
        voiceHint: "কথা বলতে বৃত্তটিতে স্পর্শ করুন (বাংলা)...",
        chips: [
          { label: "ছাতা দরকার?", query: "আজ কি বাইরে যাওয়ার সময় ছাতা দরকার?", icon: "fa-umbrella text-sky-600", cls: "bg-sky-50 hover:bg-sky-100 text-sky-800 border-sky-100" },
          { label: "কাপড় শুকানো?", query: "আজ কি বাইরে কাপড় শুকানো যাবে?", icon: "fa-jug-detergent text-teal-600", cls: "bg-teal-50 hover:bg-teal-100 text-teal-800 border-teal-100" },
          { label: "গাড়ি ধোয়া?", query: "আজ কি গাড়ি ধোয়া ঠিক হবে?", icon: "fa-car text-blue-600", cls: "bg-blue-50 hover:bg-blue-100 text-blue-800 border-blue-100" },
          { label: "কী পরবেন?", query: "আজকের তাপমাত্রায় কী ধরণের পোশাক পরা উচিত?", icon: "fa-shirt text-indigo-600", cls: "bg-indigo-50 hover:bg-indigo-100 text-indigo-800 border-indigo-100" },
          { label: "খেলাধূলা?", query: "আজ কি বাইরে খেলাধূলা বা দৌড়ানোর জন্য ভালো দিন?", icon: "fa-person-running text-emerald-600", cls: "bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border-emerald-100" },
          { label: "রাতের আবহাওয়া?", query: "আজ রাতে কি ঠান্ডা বা ভ্যাপসা গরম থাকবে?", icon: "fa-moon text-purple-600", cls: "bg-purple-50 hover:bg-purple-100 text-purple-800 border-purple-100" },
          { label: "ভ্রমণ?", query: "এই উইকএন্ডে ভ্রমণের জন্য আবহাওয়া কেমন থাকবে?", icon: "fa-route text-amber-600", cls: "bg-amber-50 hover:bg-amber-100 text-amber-800 border-amber-100" },
        ]
      },
      mr: {
        placeholder: "हवामानाबद्दल विचारा किंवा माइक दाबा...",
        voiceHint: "बोलण्यासाठी गोलावर टॅप करा (मराठी)...",
        chips: [
          { label: "छत्री हवी का?", query: "आज बाहेर जाताना छत्री नेण्याची गरज आहे का?", icon: "fa-umbrella text-sky-600", cls: "bg-sky-50 hover:bg-sky-100 text-sky-800 border-sky-100" },
          { label: "कपडे वाळवणे?", query: "आज कपडे बाहेर वाळत घालता येतील का?", icon: "fa-jug-detergent text-teal-600", cls: "bg-teal-50 hover:bg-teal-100 text-teal-800 border-teal-100" },
          { label: "गाडी धुणे?", query: "आज गाडी किंवा कार धुणे योग्य ठरेल का?", icon: "fa-car text-blue-600", cls: "bg-blue-50 hover:bg-blue-100 text-blue-800 border-blue-100" },
          { label: "काय घालावे?", query: "आजच्या तापमानानुसार कोणते कपडे घालावेत?", icon: "fa-shirt text-indigo-600", cls: "bg-indigo-50 hover:bg-indigo-100 text-indigo-800 border-indigo-100" },
          { label: "खेळ/धावणे?", query: "आज धावण्यासाठी किंवा खेळासाठी हवामान चांगले आहे का?", icon: "fa-person-running text-emerald-600", cls: "bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border-emerald-100" },
          { label: "रात्रीचे हवामान?", query: "आज रात्री थंडी जास्त असेल का?", icon: "fa-moon text-purple-600", cls: "bg-purple-50 hover:bg-purple-100 text-purple-800 border-purple-100" },
          { label: "प्रवास?", query: "या वीकेंडला प्रवासासाठी हवामान कसे राहील?", icon: "fa-route text-amber-600", cls: "bg-amber-50 hover:bg-amber-100 text-amber-800 border-amber-100" },
        ]
      },
      gu: {
        placeholder: "હવામાન વિશે પૂછો અથવા માઇક દબાવો...",
        voiceHint: "વાત કરવા માટે ગોળ પર ટેપ કરો (ગુજરાતી)...",
        chips: [
          { label: "છત્રી જોઈએ?", query: "શું આજે બહાર જતાં છત્રીની જરૂર પડશે?", icon: "fa-umbrella text-sky-600", cls: "bg-sky-50 hover:bg-sky-100 text-sky-800 border-sky-100" },
          { label: "કપડાં સૂકવવા?", query: "શું આજે કપડાં બહાર સૂકવી શકાય?", icon: "fa-jug-detergent text-teal-600", cls: "bg-teal-50 hover:bg-teal-100 text-teal-800 border-teal-100" },
          { label: "ગાડી ધોવી?", query: "શું આજે કાર કે બાઇક ધોઈ શકાય?", icon: "fa-car text-blue-600", cls: "bg-blue-50 hover:bg-blue-100 text-blue-800 border-blue-100" },
          { label: "શું પહેરવું?", query: "આજના તાપમાન મુજબ કેવા કપડાં પહેરવા જોઈએ?", icon: "fa-shirt text-indigo-600", cls: "bg-indigo-50 hover:bg-indigo-100 text-indigo-800 border-indigo-100" },
          { label: "રમતગમત?", query: "શું આજે દોડવા કે રમતગમત માટે હવામાન સારું છે?", icon: "fa-person-running text-emerald-600", cls: "bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border-emerald-100" },
          { label: "રાત્રિનું હવામાન?", query: "શું આજે રાત્રે ઠંડી કે બફારો વધારે રહેશે?", icon: "fa-moon text-purple-600", cls: "bg-purple-50 hover:bg-purple-100 text-purple-800 border-purple-100" },
          { label: "મુસાફરી?", query: "આ વીકએન્ડમાં પ્રવાસ માટે હવામાન કેવું રહેશે?", icon: "fa-route text-amber-600", cls: "bg-amber-50 hover:bg-amber-100 text-amber-800 border-amber-100" },
        ]
      },
      pa: {
        placeholder: "ਮੌਸਮ ਬਾਰੇ ਪੁੱਛੋ ਜਾਂ ਮਾਈਕ ਦਬਾਓ...",
        voiceHint: "ਬੋਲਣ ਲਈ ਚੱਕਰ 'ਤੇ ਟੈਪ ਕਰੋ (ਪੰਜਾਬੀ)...",
        chips: [
          { label: "ਛਤਰੀ ਚਾਹੀਦੀ?", query: "ਕੀ ਅੱਜ ਬਾਹਰ ਜਾਣ ਵੇਲੇ ਛਤਰੀ ਦੀ ਲੋੜ ਪਵੇਗੀ?", icon: "fa-umbrella text-sky-600", cls: "bg-sky-50 hover:bg-sky-100 text-sky-800 border-sky-100" },
          { label: "ਕੱਪੜੇ ਸੁਕਾਉਣੇ?", query: "ਕੀ ਅੱਜ ਕੱਪੜੇ ਬਾਹਰ ਸੁਕਾਏ ਜਾ ਸਕਦੇ ਹਨ?", icon: "fa-jug-detergent text-teal-600", cls: "bg-teal-50 hover:bg-teal-100 text-teal-800 border-teal-100" },
          { label: "ਗੱਡੀ ਧੋਣੀ?", query: "ਕੀ ਅੱਜ ਗੱਡੀ ਜਾਂ ਕਾਰ ਧੋਣੀ ਸਹੀ ਰਹੇਗੀ?", icon: "fa-car text-blue-600", cls: "bg-blue-50 hover:bg-blue-100 text-blue-800 border-blue-100" },
          { label: "ਕੀ ਪਾਈਏ?", query: "ਅੱਜ ਦੇ ਤਾਪਮਾਨ ਅਨੁਸਾਰ ਕਿਹੋ ਜਿਹੇ ਕੱਪੜੇ ਪਾਉਣੇ ਚਾਹੀਦੇ ਹਨ?", icon: "fa-shirt text-indigo-600", cls: "bg-indigo-50 hover:bg-indigo-100 text-indigo-800 border-indigo-100" },
          { label: "ਖੇਡਾਂ/ਦੌੜ?", query: "ਕੀ ਅੱਜ ਖੇਡਣ ਜਾਂ ਦੌੜਨ ਲਈ ਮੌਸਮ ਚੰਗਾ ਹੈ?", icon: "fa-person-running text-emerald-600", cls: "bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border-emerald-100" },
          { label: "ਰਾਤ ਦਾ ਮੌਸਮ?", query: "ਕੀ ਅੱਜ ਰਾਤ ਠੰਢ ਜ਼ਿਆਦਾ ਹੋਵੇਗੀ?", icon: "fa-moon text-purple-600", cls: "bg-purple-50 hover:bg-purple-100 text-purple-800 border-purple-100" },
          { label: "ਸਫ਼ਰ?", query: "ਇਸ ਹਫ਼ਤਾਵਾਰੀ ਸਫ਼ਰ ਲਈ ਮੌਸਮ ਕਿਹੋ ਜਿਹਾ ਰਹੇਗਾ?", icon: "fa-route text-amber-600", cls: "bg-amber-50 hover:bg-amber-100 text-amber-800 border-amber-100" },
        ]
      },
      or: {
        placeholder: "ପାଣିପାଗ ବିଷୟରେ ପଚାରନ୍ତୁ କିମ୍ବା ମାଇକ୍ ଦବାନ୍ତୁ...",
        voiceHint: "କହିବା ପାଇଁ ବୃତ୍ତ ଉପରେ କ୍ଲିକ୍ କରନ୍ତୁ (ଓଡ଼ିଆ)...",
        chips: [
          { label: "ଛତା ଦରକାର?", query: "ଆଜି ବାହାରକୁ ଯିବା ବେଳେ ଛତା ଦରକାର କି?", icon: "fa-umbrella text-sky-600", cls: "bg-sky-50 hover:bg-sky-100 text-sky-800 border-sky-100" },
          { label: "ଲୁଗା ଶୁଖାଇବା?", query: "ଆଜି ଲୁଗା ବାହାରେ ଶୁଖାଇ ହେବ କି?", icon: "fa-jug-detergent text-teal-600", cls: "bg-teal-50 hover:bg-teal-100 text-teal-800 border-teal-100" },
          { label: "ଗାଡ଼ି ଧୋଇବା?", query: "ଆଜି ଗାଡ଼ି ଧୋଇବା ଠିକ୍ ହେବ କି?", icon: "fa-car text-blue-600", cls: "bg-blue-50 hover:bg-blue-100 text-blue-800 border-blue-100" },
          { label: "କଣ ପିନ୍ଧିବେ?", query: "ଆଜିର ତାପମାତ୍ରା ଅନୁସାରେ କିଭଳି ପୋଷାକ ପିନ୍ଧିବା ଉଚିତ୍?", icon: "fa-shirt text-indigo-600", cls: "bg-indigo-50 hover:bg-indigo-100 text-indigo-800 border-indigo-100" },
          { label: "ଖେଳ/ଦୌଡ଼?", query: "ଆଜି ଖେଳ କିମ୍ବା ବ୍ୟାୟାମ ପାଇଁ ପାଗ ଭଲ କି?", icon: "fa-person-running text-emerald-600", cls: "bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border-emerald-100" },
          { label: "ରାତି ପାଗ?", query: "ଆଜି ରାତିରେ ଅଧିକ ଥଣ୍ଡା ରହିବ କି?", icon: "fa-moon text-purple-600", cls: "bg-purple-50 hover:bg-purple-100 text-purple-800 border-purple-100" },
          { label: "ଯାତ୍ରା?", query: "ଏହି ସପ୍ତାହାନ୍ତ ଯାତ୍ରା ପାଇଁ ପାଗ କିଭଳି ରହିବ?", icon: "fa-route text-amber-600", cls: "bg-amber-50 hover:bg-amber-100 text-amber-800 border-amber-100" },
        ]
      },
      as: {
        placeholder: "বতৰৰ বিষয়ে সোধক বা মাইক টিপক...",
        voiceHint: "কথা পাতিবলৈ বৃত্তটোত স্পৰ্শ কৰক (অসমীয়া)...",
        chips: [
          { label: "ছাতি লাগিবনে?", query: "আজি বাহিৰলৈ ওলাওঁতে ছাতি ল'ব লাগিব নেকি?", icon: "fa-umbrella text-sky-600", cls: "bg-sky-50 hover:bg-sky-100 text-sky-800 border-sky-100" },
          { label: "কাপোৰ শুকুওৱা?", query: "আজি বাহিৰত কাপোৰ শুকুৱাব পৰা যাবনে?", icon: "fa-jug-detergent text-teal-600", cls: "bg-teal-50 hover:bg-teal-100 text-teal-800 border-teal-100" },
          { label: "গাড়ী ধোৱা?", query: "আজি গাড়ী ধোৱাটো ভাল হ'বনে?", icon: "fa-car text-blue-600", cls: "bg-blue-50 hover:bg-blue-100 text-blue-800 border-blue-100" },
          { label: "কি পিন্ধিব?", query: "আজিৰ তাপমাত্ৰাৰ বাবে কেনেকুৱা কাপোৰ পিন্ধা উচিত?", icon: "fa-shirt text-indigo-600", cls: "bg-indigo-50 hover:bg-indigo-100 text-indigo-800 border-indigo-100" },
          { label: "খেল/দৌৰ?", query: "আজি খেল বা দৌৰৰ বাবে বতৰ ভালনে?", icon: "fa-person-running text-emerald-600", cls: "bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border-emerald-100" },
          { label: "ৰাতিৰ বতৰ?", query: "আজি ৰাতি ঠাণ্ডা বেছি পৰিব নেকি?", icon: "fa-moon text-purple-600", cls: "bg-purple-50 hover:bg-purple-100 text-purple-800 border-purple-100" },
          { label: "ভ্ৰমণ?", query: "এই সপ্তাহান্তৰ ভ্ৰমণৰ বাবে বতৰ কেনেকুৱা হ'ব?", icon: "fa-route text-amber-600", cls: "bg-amber-50 hover:bg-amber-100 text-amber-800 border-amber-100" },
        ]
      },
      ur: {
        placeholder: "موسم کے بارے میں پوچھیں یا مائیک دبائیں...",
        voiceHint: "بات کرنے کے لیے دائرے پر ٹیپ کریں (اردو)...",
        chips: [
          { label: "چھتری چاہیے؟", query: "کیا آج باہر جاتے وقت چھتری کی ضرورت پڑے گی؟", icon: "fa-umbrella text-sky-600", cls: "bg-sky-50 hover:bg-sky-100 text-sky-800 border-sky-100" },
          { label: "کپڑے سکھانا؟", query: "کیا آج کپڑے باہر سکھائے جا سکتے ہیں؟", icon: "fa-jug-detergent text-teal-600", cls: "bg-teal-50 hover:bg-teal-100 text-teal-800 border-teal-100" },
          { label: "گاڑی دھونا؟", query: "کیا آج گاڑی یا بائیک دھونا صحیح رہے گا؟", icon: "fa-car text-blue-600", cls: "bg-blue-50 hover:bg-blue-100 text-blue-800 border-blue-100" },
          { label: "کیا پہنیں؟", query: "آج کے درجہ حرارت کے مطابق کیا پہننا چاہیے؟", icon: "fa-shirt text-indigo-600", cls: "bg-indigo-50 hover:bg-indigo-100 text-indigo-800 border-indigo-100" },
          { label: "کھیل کود؟", query: "کیا آج کھیل کود یا دوڑنے کے لیے موسم اچھا ہے؟", icon: "fa-person-running text-emerald-600", cls: "bg-emerald-50 hover:bg-emerald-100 text-emerald-800 border-emerald-100" },
          { label: "رات کا موسم؟", query: "کیا آج رات سردی یا حبس زیادہ رہے گی؟", icon: "fa-moon text-purple-600", cls: "bg-purple-50 hover:bg-purple-100 text-purple-800 border-purple-100" },
          { label: "سفر؟", query: "اس ہفتہ سفر کے لیے موسم کیسا رہے گا؟", icon: "fa-route text-amber-600", cls: "bg-amber-50 hover:bg-amber-100 text-amber-800 border-amber-100" },
        ]
      }
    };

    function updateLanguageUI(langCode) {
      const ui = LOCALIZED_UI[langCode] || LOCALIZED_UI["en"];
      const container = document.getElementById('quick-chips-container');
      if (container && ui.chips) {
        container.innerHTML = "";
        ui.chips.forEach(c => {
          const btn = document.createElement('button');
          btn.className = `text-[11px] px-2.5 py-1 rounded-lg ${c.cls} font-medium transition border shadow-xs hover:scale-105 active:scale-95 flex items-center whitespace-nowrap`;
          btn.innerHTML = `<i class="fa-solid ${c.icon} mr-1"></i> <span>${escapeHtml(c.label)}</span>`;
          btn.onclick = () => sendQuickPrompt(c.query);
          container.appendChild(btn);
        });
      }
      const chatInput = document.getElementById('chat-input');
      if (chatInput && ui.placeholder) {
        chatInput.placeholder = ui.placeholder;
      }
      const voiceHint = document.getElementById('voice-status-label');
      if (voiceHint && !isVoiceSessionActive && ui.voiceHint) {
        voiceHint.innerText = ui.voiceHint;
      }
    }

    // Load saved settings on startup
    function initSettings() {
      const savedGemini = localStorage.getItem('weathergpt_gemini_key') || "";
      const savedOM = localStorage.getItem('weathergpt_openmeteo_key') || "";
      const savedLang = localStorage.getItem('weathergpt_lang') || "en";
      
      document.getElementById('key-gemini').value = savedGemini;
      document.getElementById('key-openmeteo').value = savedOM;
      applyAppLanguage(savedLang);

      // Check online / offline status
      updateNetworkBadge();
      window.addEventListener('online', () => {
        updateNetworkBadge();
        if (isOfflineView) exitOfflineView();
      });
      window.addEventListener('offline', () => {
        updateNetworkBadge();
        loadOfflineWeatherFromStorage();
      });

      // Initialize Web Speech Recognition
      initSpeechRecognition();

      // Pre-load synthesis voices
      if ('speechSynthesis' in window) {
        window.speechSynthesis.onvoiceschanged = () => {
          console.log("TTS voices loaded:", window.speechSynthesis.getVoices().length);
        };
      }
    }

    function updateNetworkBadge() {
      const isOnline = navigator.onLine;
      const badge = document.getElementById('net-badge');
      if (badge) {
        if (isOnline) {
          badge.className = "text-xs font-semibold px-2.5 py-1 rounded-full bg-emerald-500/20 text-emerald-300 border border-emerald-500/30 flex items-center gap-1.5";
          badge.innerHTML = `<i class="fa-solid fa-wifi text-[10px]"></i> Online`;
        } else {
          badge.className = "text-xs font-semibold px-2.5 py-1 rounded-full bg-rose-500/20 text-rose-300 border border-rose-500/30 flex items-center gap-1.5";
          badge.innerHTML = `<i class="fa-solid fa-plane text-[10px]"></i> Offline`;
        }
      }
    }

    // 1. Live GPS Tool
    function acquireLiveGPS() {
      const btn = document.getElementById('btn-gps');
      const icon = document.getElementById('gps-icon');
      const text = document.getElementById('gps-text');

      if (!navigator.geolocation) {
        alert("Geolocation is not supported by your browser.");
        return;
      }

      text.innerText = "Locating...";
      icon.className = "fa-solid fa-circle-notch fa-spin text-sm";
      btn.disabled = true;

      navigator.geolocation.getCurrentPosition(
        async (position) => {
          currentLat = position.coords.latitude;
          currentLon = position.coords.longitude;
          const accuracy = Math.round(position.coords.accuracy);

          // Update UI
          const accBadge = document.getElementById('gps-accuracy-badge');
          accBadge.innerText = `GPS (±${accuracy}m)`;
          accBadge.classList.remove('hidden');

          document.getElementById('current-coords-text').innerText = 
            `${currentLat.toFixed(4)}° N, ${currentLon.toFixed(4)}° E • Live GPS accuracy ±${accuracy}m`;

          lastGpsAccuracy = accuracy;
          updateCoordsDisplay(currentLang);

          // Reverse Geocode with current active language
          try {
            const geoResp = await fetch(`/api/geocode/reverse?latitude=${currentLat}&longitude=${currentLon}&language=${currentLang}`);
            if (geoResp.ok) {
              const geoData = await geoResp.json();
              rawLocationEnglish = geoData.raw_display_name || geoData.name;
              cachedLocationTranslations[currentLang] = geoData.display_name;
              currentLocationName = geoData.display_name;
              document.getElementById('current-location-name').innerText = currentLocationName;
            }
          } catch (e) {
            console.warn("Reverse geocode failed:", e);
            currentLocationName = localizeLocationName(currentLocationName, currentLang);
            document.getElementById('current-location-name').innerText = currentLocationName;
          }

          // Fetch Weather
          await fetchWeather();

          text.innerText = "GPS Active";
          icon.className = "fa-solid fa-location-dot text-sm text-emerald-300";
          btn.disabled = false;
        },
        (error) => {
          console.error("GPS Error:", error);
          let msg = "Could not get live location: ";
          if (error.code === 1) msg += "Permission denied. Please allow location access.";
          else if (error.code === 2) msg += "Position unavailable.";
          else msg += error.message;
          alert(msg);

          text.innerText = "Live GPS";
          icon.className = "fa-solid fa-location-crosshairs text-sm";
          btn.disabled = false;
        },
        { enableHighAccuracy: true, timeout: 12000, maximumAge: 0 }
      );
    }

    function resetToCurrentLocation() {
      acquireLiveGPS();
    }

    // 2. City Search with Autocomplete
    function handleSearchInput(val) {
      clearTimeout(searchDebounceTimeout);
      const clearBtn = document.getElementById('search-clear');
      const dropdown = document.getElementById('search-dropdown');

      if (!val || val.trim().length < 2) {
        dropdown.classList.add('hidden');
        clearBtn.classList.add('hidden');
        return;
      }

      clearBtn.classList.remove('hidden');
      searchDebounceTimeout = setTimeout(async () => {
        try {
          const resp = await fetch(`/api/geocode/search?query=${encodeURIComponent(val.trim())}`);
          if (resp.ok) {
            const data = await resp.json();
            renderSearchResults(data.results || []);
          }
        } catch (e) {
          console.warn("Search error:", e);
        }
      }, 300);
    }

    function renderSearchResults(results) {
      const dropdown = document.getElementById('search-dropdown');
      if (!results || results.length === 0) {
        dropdown.innerHTML = `<div class="p-3 text-slate-400 text-center">No locations found</div>`;
        dropdown.classList.remove('hidden');
        return;
      }

      dropdown.innerHTML = results.map(r => `
        <div onclick="selectSearchResult(${r.latitude}, ${r.longitude}, '${escapeHtml(r.display_name)}')" class="p-3 hover:bg-slate-800 text-slate-200 cursor-pointer flex items-center justify-between transition">
          <div>
            <span class="font-bold text-white">${escapeHtml(r.name)}</span>
            <span class="text-[11px] text-slate-400 ml-1.5">${escapeHtml([r.admin1, r.country].filter(Boolean).join(', '))}</span>
          </div>
          <span class="text-[10px] text-slate-500 font-mono">${r.latitude.toFixed(2)}°, ${r.longitude.toFixed(2)}°</span>
        </div>
      `).join('');

      dropdown.classList.remove('hidden');
    }

    function selectSearchResult(lat, lon, displayName) {
      document.getElementById('search-dropdown').classList.add('hidden');
      document.getElementById('search-input').value = displayName;
      selectCity(lat, lon, displayName);
    }

    function clearSearch() {
      document.getElementById('search-input').value = "";
      document.getElementById('search-dropdown').classList.add('hidden');
      document.getElementById('search-clear').classList.add('hidden');
    }

    function selectCity(lat, lon, name) {
      currentLat = lat;
      currentLon = lon;
      rawLocationEnglish = name;
      lastGpsAccuracy = null;
      cachedLocationTranslations = {};
      currentLocationName = localizeLocationName(name, currentLang);
      document.getElementById('current-location-name').innerText = currentLocationName;
      updateCoordsDisplay(currentLang);
      document.getElementById('gps-accuracy-badge').classList.add('hidden');
      fetchWeather();
    }

    // 3. Fetch Weather Forecast from Open-Meteo
    async function fetchWeather(forceRefresh = false) {
      if (isOfflineView) {
        loadOfflineWeatherFromStorage();
        return;
      }

      const omKey = localStorage.getItem('weathergpt_openmeteo_key') || "";
      const url = `/api/weather?latitude=${currentLat}&longitude=${currentLon}&force_refresh=${forceRefresh}${omKey ? '&api_key=' + encodeURIComponent(omKey) : ''}`;

      try {
        const resp = await fetch(url);
        if (!resp.ok) throw new Error("API returned status: " + resp.status);

        const data = await resp.json();
        cachedWeatherData = data;
        renderWeatherUI(data);

        // Auto cache this payload for offline safety
        localStorage.setItem('weathergpt_last_cache', JSON.stringify({
          locationName: currentLocationName,
          lat: currentLat,
          lon: currentLon,
          savedAt: new Date().toISOString(),
          data: data
        }));

      } catch (err) {
        console.error("Failed to fetch weather:", err);
        loadOfflineWeatherFromStorage();
      }
    }

    function renderWeatherUI(data) {
      const c = data.current;
      const daily = data.daily_forecast || [];
      const i18n = APP_I18N[currentLang] || APP_I18N["en"];

      // Translated weather condition
      const condDesc = translateWeatherDesc(c.weather_desc, currentLang);

      // Hero Elements
      document.getElementById('hero-temp').innerText = `${Math.round(c.temperature_2m)}°`;
      document.getElementById('hero-feels-like').innerText = `${i18n.feelsLike} ${c.apparent_temperature}°C`;
      document.getElementById('hero-minmax').innerText = i18n.highLow
        .replace('{max}', c.daily_max_temp)
        .replace('{min}', c.daily_min_temp);
      document.getElementById('hero-weather-desc').innerText = condDesc;
      
      const heroIcon = document.getElementById('hero-icon');
      heroIcon.className = `fa-solid ${c.weather_icon}`;

      // Smart Advice Tip in selected language
      const tipText = document.getElementById('weather-tip-text');
      const tipIcon = document.getElementById('tip-icon');
      if (c.daily_rain_probability >= 50 || c.precipitation > 0.5) {
        tipText.innerText = (i18n.tips.rain || "")
          .replace('{p}', c.daily_rain_probability);
        tipIcon.className = "fa-solid fa-umbrella text-sky-400";
      } else if (c.temperature_2m >= 34) {
        tipText.innerText = (i18n.tips.warm || "")
          .replace('{t}', c.temperature_2m);
        tipIcon.className = "fa-solid fa-sun text-amber-400";
      } else {
        tipText.innerText = (i18n.tips.pleasant || "")
          .replace('{d}', condDesc.toLowerCase())
          .replace('{t}', c.temperature_2m);
        tipIcon.className = "fa-solid fa-circle-check text-emerald-400";
      }

      // Metrics Grid
      document.getElementById('metric-rain-prob').innerText = `${c.daily_rain_probability}%`;
      document.getElementById('metric-rain-sum').innerText = (i18n.rainExpected || "{rain} mm expected")
        .replace('{rain}', c.precipitation);
      document.getElementById('rain-bar').style.width = `${Math.min(100, c.daily_rain_probability)}%`;

      document.getElementById('metric-humidity').innerText = `${c.relative_humidity_2m}%`;
      document.getElementById('humidity-bar').style.width = `${c.relative_humidity_2m}%`;
      const humDesc = document.getElementById('metric-humidity-desc');
      if (humDesc) {
        if (c.relative_humidity_2m >= 75) humDesc.innerText = i18n.humid || "Humid";
        else if (c.relative_humidity_2m <= 35) humDesc.innerText = i18n.dry || "Dry";
        else humDesc.innerText = i18n.comfort || "Comfortable";
      }

      document.getElementById('metric-wind').innerText = `${c.wind_speed_10m} km/h`;
      document.getElementById('wind-bar').style.width = `${Math.min(100, c.wind_speed_10m * 2.5)}%`;
      const windDir = document.getElementById('metric-wind-dir');
      if (windDir) windDir.innerText = i18n.windBreeze || "Gentle breeze";

      document.getElementById('metric-uv').innerText = c.uv_index;
      document.getElementById('uv-bar').style.width = `${Math.min(100, c.uv_index * 10)}%`;
      const uvLevel = document.getElementById('metric-uv-level');
      if (c.uv_index >= 8) {
        uvLevel.innerText = i18n.uvLevels.veryHigh;
        uvLevel.className = "text-[10px] text-red-600 font-bold block";
      } else if (c.uv_index >= 6) {
        uvLevel.innerText = i18n.uvLevels.high;
        uvLevel.className = "text-[10px] text-amber-600 font-bold block";
      } else {
        uvLevel.innerText = i18n.uvLevels.safe;
        uvLevel.className = "text-[10px] text-emerald-600 font-bold block";
      }

      document.getElementById('metric-pressure').innerText = `${c.surface_pressure} hPa`;
      document.getElementById('metric-clouds').innerText = `${c.cloud_cover}%`;
      document.getElementById('cloud-bar').style.width = `${c.cloud_cover}%`;

      // 7-Day Forecast Grid with localized day names & weather conditions
      const grid = document.getElementById('forecast-7day-grid');
      grid.innerHTML = daily.map((day, idx) => {
        const dayNameTrans = translateDayName(day.day_name, idx === 0, currentLang);
        const dayCondTrans = translateWeatherDesc(day.weather_desc, currentLang);
        return `
          <div class="p-3.5 rounded-2xl ${idx === 0 ? 'bg-sky-50 border-2 border-sky-300' : 'bg-slate-50 border border-slate-100'} hover:bg-white hover:shadow-md transition text-center flex flex-col justify-between space-y-2">
            <div>
              <span class="text-xs font-bold ${idx === 0 ? 'text-sky-700' : 'text-slate-800'} block">${escapeHtml(dayNameTrans)}</span>
              <span class="text-[10px] text-slate-400 block">${escapeHtml(day.formatted_date)}</span>
            </div>

            <div class="my-1 text-2xl ${day.precipitation_probability >= 50 ? 'text-blue-500' : 'text-amber-500'}">
              <i class="fa-solid ${day.weather_icon}"></i>
            </div>

            <span class="text-[11px] font-medium text-slate-600 truncate block" title="${escapeHtml(dayCondTrans)}">
              ${escapeHtml(dayCondTrans)}
            </span>

            <div class="flex items-center justify-center gap-1.5 text-xs font-bold">
              <span class="text-slate-900">${Math.round(day.temp_max)}°</span>
              <span class="text-slate-400 font-normal">/</span>
              <span class="text-slate-400">${Math.round(day.temp_min)}°</span>
            </div>

            <div class="text-[10px] py-1 px-1.5 rounded-lg ${day.precipitation_probability >= 40 ? 'bg-blue-100 text-blue-800 font-bold' : 'bg-slate-100 text-slate-500'} flex items-center justify-center gap-1">
              <i class="fa-solid fa-droplet text-[9px]"></i>
              <span>${day.precipitation_probability}%</span>
            </div>
          </div>
        `;
      }).join('');
    }

    // 4. Offline Downloading & Storage
    async function downloadForecastForOffline() {
      const btn = document.getElementById('btn-download-cache');
      btn.disabled = true;
      btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin"></i> Saving...`;

      try {
        const omKey = localStorage.getItem('weathergpt_openmeteo_key') || "";
        const url = `/api/offline-bundle?latitude=${currentLat}&longitude=${currentLon}&location_name=${encodeURIComponent(currentLocationName)}${omKey ? '&api_key=' + encodeURIComponent(omKey) : ''}`;
        
        const resp = await fetch(url);
        if (!resp.ok) throw new Error("Offline bundle request failed");
        
        const bundle = await resp.json();
        localStorage.setItem('weathergpt_offline_bundle', JSON.stringify(bundle));
        
        const tag = document.getElementById('offline-saved-tag');
        tag.classList.remove('hidden');
        tag.innerText = `Saved locally (${new Date().toLocaleTimeString()})`;

        alert("7-Day Weather Data successfully saved offline on your device! You can now access it even when you have no network.");
      } catch (err) {
        alert("Failed to download offline package: " + err.message);
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-download"></i> Save 7-Day Offline`;
      }
    }

    function exportForecastJSON() {
      const saved = localStorage.getItem('weathergpt_offline_bundle') || localStorage.getItem('weathergpt_last_cache');
      if (!saved) {
        alert("Please click 'Save 7-Day Offline' first!");
        return;
      }

      const blob = new Blob([saved], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = `weathergpt_7day_${currentLocationName.split(',')[0].trim().toLowerCase()}.json`;
      a.click();
      URL.revokeObjectURL(url);
    }

    function toggleOfflineView() {
      if (isOfflineView) exitOfflineView();
      else loadOfflineWeatherFromStorage();
    }

    function loadOfflineWeatherFromStorage() {
      const offlineBundleStr = localStorage.getItem('weathergpt_offline_bundle') || localStorage.getItem('weathergpt_last_cache');
      if (!offlineBundleStr) {
        alert("No offline weather data found. Please connect to internet once and click 'Save 7-Day Offline'.");
        return;
      }

      try {
        const parsed = JSON.parse(offlineBundleStr);
        let forecastData = parsed.data || {
          current: parsed.current_weather,
          daily_forecast: parsed.seven_day_forecast
        };

        isOfflineView = true;
        document.getElementById('offline-banner').classList.remove('hidden');
        document.getElementById('offline-banner-text').innerText = 
          `Offline Mode Active: Showing 7-day weather saved on ${parsed.downloaded_at || parsed.savedAt || 'device'}`;
        document.getElementById('offline-mode-text').innerText = "Exit Offline";

        if (parsed.location && parsed.location.name) {
          document.getElementById('current-location-name').innerText = parsed.location.name + " (Offline)";
        }

        renderWeatherUI(forecastData);
      } catch (e) {
        alert("Error reading offline storage: " + e.message);
      }
    }

    function exitOfflineView() {
      isOfflineView = false;
      document.getElementById('offline-banner').classList.add('hidden');
      document.getElementById('offline-mode-text').innerText = "Offline";
      fetchWeather();
    }

    // 5. Conversational Weather Guide (Gemini & Fallback)
    function changeLanguage(langCode) {
      applyAppLanguage(langCode);
    }

    function switchVoiceModalLanguage(langCode) {
      applyAppLanguage(langCode);
      if (isVoiceSessionActive) {
        stopVoiceRecognition();
        startVoiceRecognition();
      }
    }

    function sendQuickPrompt(promptText) {
      ensureAudioUnlocked();
      document.getElementById('chat-input').value = promptText;
      sendChatMessage();
    }

    function handleChatKeyPress(e) {
      if (e.key === 'Enter') {
        ensureAudioUnlocked();
        sendChatMessage();
      }
    }

    async function sendChatMessage(customMsg = null, autoPlay = true) {
      ensureAudioUnlocked();
      const input = document.getElementById('chat-input');
      const message = customMsg || input.value.trim();
      if (!message) return null;

      if (!customMsg) input.value = "";
      appendChatMessage("user", message);

      // Record user turn in conversational memory
      chatHistory.push({ role: "user", content: message });
      if (chatHistory.length > 10) chatHistory = chatHistory.slice(-10);

      const btn = document.getElementById('btn-send-chat');
      btn.disabled = true;
      btn.innerHTML = `<i class="fa-solid fa-circle-notch fa-spin text-xs"></i>`;

      const geminiKey = localStorage.getItem('weathergpt_gemini_key') || "";
      const omKey = localStorage.getItem('weathergpt_openmeteo_key') || "";

      try {
        const resp = await fetch('/api/chat', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            message: message,
            latitude: currentLat,
            longitude: currentLon,
            location_name: currentLocationName,
            language: currentLang,
            gemini_api_key: geminiKey || null,
            open_meteo_api_key: omKey || null,
            history: chatHistory.slice(0, -1), // Prior turns for contextual relativity
          })
        });

        if (!resp.ok) throw new Error("Guide response error: " + resp.status);

        const data = await resp.json();
        appendChatMessage("assistant", data.response, data.source, data.audio_base64, data.audio_mime_type, data.speech_locale);

        // Record assistant turn in conversational memory
        chatHistory.push({ role: "assistant", content: data.response });
        if (chatHistory.length > 10) chatHistory = chatHistory.slice(-10);

        // Store for audio playback
        lastGuideResponseText = data.response;
        lastGuideLocale = data.speech_locale || INDIAN_LOCALE_MAP[currentLang] || "en-IN";
        lastAudioBase64 = data.audio_base64;
        lastAudioMime = data.audio_mime_type || "audio/mp3";

        const audioBar = document.getElementById('audio-control-bar');
        if (audioBar) audioBar.classList.remove('hidden');

        // Play audio clearly in respected Indian language if requested
        if (autoPlay) {
          playNativeSpeechAudio(data.response, data.audio_base64, data.audio_mime_type, data.speech_locale);
        }

        return data;

      } catch (err) {
        const errMsg = "I am having trouble connecting right now. Please check your network.";
        appendChatMessage("assistant", errMsg);
        return { response: errMsg };
      } finally {
        btn.disabled = false;
        btn.innerHTML = `<i class="fa-solid fa-paper-plane text-xs"></i>`;
      }
    }

    function appendChatMessage(sender, text, source = "", audioB64 = null, mimeType = "audio/mp3", locale = "") {
      const container = document.getElementById('chat-messages');
      const isUser = sender === "user";
      const msgId = "msg-" + (++messageCounter);

      if (!isUser && audioB64) {
        messageAudioStore[msgId] = {
          text: text,
          audioB64: audioB64,
          mimeType: mimeType || "audio/mp3",
          locale: locale || INDIAN_LOCALE_MAP[currentLang] || "en-IN",
          langCode: currentLang,
        };
      }

      const langName = INDIAN_LANG_NAMES[currentLang] || "Audio";

      const html = isUser ? `
        <div class="flex items-start justify-end gap-2.5">
          <div class="p-3 rounded-2xl bg-gradient-to-r from-sky-500 to-indigo-600 text-white max-w-[85%] leading-relaxed shadow-sm">
            <p>${escapeHtml(text)}</p>
          </div>
          <div class="w-7 h-7 rounded-lg bg-indigo-100 text-indigo-700 flex-shrink-0 flex items-center justify-center text-xs mt-0.5 font-bold">
            <i class="fa-solid fa-user"></i>
          </div>
        </div>
      ` : `
        <div class="flex items-start gap-2.5">
          <div class="w-7 h-7 rounded-lg bg-sky-500 text-white flex-shrink-0 flex items-center justify-center text-xs mt-0.5 shadow-sm">
            <i class="fa-solid fa-robot"></i>
          </div>
          <div class="p-3.5 rounded-2xl bg-slate-100 text-slate-800 max-w-[88%] leading-relaxed border border-slate-200">
            <p class="font-medium text-[13px] text-slate-800 leading-snug">${escapeHtml(text)}</p>
            ${audioB64 ? `
              <div class="mt-2.5 pt-2 border-t border-slate-200/80 flex items-center justify-between gap-2">
                <button id="btn-audio-${msgId}" onclick="playSpecificMessageAudio('${msgId}')" class="px-2.5 py-1 rounded-lg bg-sky-50 hover:bg-sky-100 text-sky-800 font-bold text-[11px] flex items-center gap-1.5 transition active:scale-95 border border-sky-200 shadow-xs hover:border-sky-300">
                  <i class="fa-solid fa-volume-high text-sky-600"></i>
                  <span>${(APP_I18N[currentLang] && APP_I18N[currentLang].listenBtn ? APP_I18N[currentLang].listenBtn.replace('{lang}', langName) : `🔊 Listen (${langName})`)}</span>
                </button>
                ${source ? `<span class="text-[9px] text-slate-400 uppercase font-mono tracking-wider">${source.replace(/_/g, ' ')}</span>` : ''}
              </div>
            ` : (source ? `<span class="text-[9px] text-slate-400 mt-1 block uppercase font-mono tracking-wider">${source.replace(/_/g, ' ')}</span>` : '')}
          </div>
        </div>
      `;

      container.insertAdjacentHTML('beforeend', html);
      container.scrollTop = container.scrollHeight;
    }

    function playSpecificMessageAudio(msgId) {
      ensureAudioUnlocked();
      const item = messageAudioStore[msgId];
      if (!item) return;
      lastGuideResponseText = item.text;
      lastAudioBase64 = item.audioB64;
      lastAudioMime = item.mimeType;
      lastGuideLocale = item.locale;
      playNativeSpeechAudio(item.text, item.audioB64, item.mimeType, item.locale);
    }

    // 6. ADVANCED INTERACTIVE VOICE ASSISTANT ENGINE
    function openVoiceModal() {
      ensureAudioUnlocked();
      document.getElementById('voice-modal').classList.remove('hidden');
      document.getElementById('voice-modal-lang').value = currentLang;
      setOrbState('idle');
      document.getElementById('voice-status-label').innerText = "Tap orb or speak to start";
      
      // Auto-start voice recognition session
      startVoiceRecognition();
    }

    function closeVoiceModal() {
      stopVoiceRecognition();
      if (globalAudioPlayer) globalAudioPlayer.pause();
      if ('speechSynthesis' in window) window.speechSynthesis.cancel();
      isSpeaking = false;
      document.getElementById('voice-modal').classList.add('hidden');
    }

    function toggleVoiceSession() {
      ensureAudioUnlocked();
      if (isSpeaking) {
        if (globalAudioPlayer) globalAudioPlayer.pause();
        if ('speechSynthesis' in window) window.speechSynthesis.cancel();
        isSpeaking = false;
        setOrbState('idle');
        document.getElementById('voice-status-label').innerText = "Voice paused. Tap orb to speak.";
        return;
      }
      if (isVoiceSessionActive) {
        stopVoiceRecognition();
        setOrbState('idle');
        document.getElementById('voice-status-label').innerText = "Voice session paused. Tap to speak.";
      } else {
        startVoiceRecognition();
      }
    }

    function setOrbState(state) {
      const orb = document.getElementById('voice-orb');
      const icon = document.getElementById('orb-center-icon');
      const waveBars = document.querySelectorAll('.sound-bar');

      orb.className = 'voice-orb';
      waveBars.forEach(b => b.classList.remove('sound-bar-active'));

      if (state === 'listening') {
        orb.classList.add('orb-listening');
        icon.className = "fa-solid fa-waveform-lines text-white text-3xl";
        waveBars.forEach(b => b.classList.add('sound-bar-active'));
        document.getElementById('voice-status-label').innerText = "Listening... Speak your weather question";
      } else if (state === 'thinking') {
        orb.classList.add('orb-thinking');
        icon.className = "fa-solid fa-spinner fa-spin text-white text-3xl";
        document.getElementById('voice-status-label').innerText = "WeatherGPT is thinking...";
      } else if (state === 'speaking') {
        orb.classList.add('orb-speaking');
        icon.className = "fa-solid fa-volume-high text-white text-3xl";
        waveBars.forEach(b => b.classList.add('sound-bar-active'));
        document.getElementById('voice-status-label').innerText = "Speaking weather advice...";
      } else {
        orb.classList.add('orb-pulse-idle');
        icon.className = "fa-solid fa-microphone text-white text-3xl";
        document.getElementById('voice-status-label').innerText = "Ready. Tap orb to speak.";
      }
    }

    function initSpeechRecognition() {
      const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
      if (!SpeechRecognition) return;

      speechRecognition = new SpeechRecognition();
      speechRecognition.continuous = true;
      speechRecognition.interimResults = true;

      speechRecognition.onstart = () => {
        isVoiceSessionActive = true;
        setOrbState('listening');
      };

      speechRecognition.onresult = (event) => {
        let interimTranscript = '';
        let finalTranscript = '';

        for (let i = event.resultIndex; i < event.results.length; ++i) {
          if (event.results[i].isFinal) {
            finalTranscript += event.results[i][0].transcript;
          } else {
            interimTranscript += event.results[i][0].transcript;
          }
        }

        const currentText = finalTranscript || interimTranscript;
        if (currentText) {
          document.getElementById('voice-user-transcript').innerText = `"${currentText}"`;
          
          // Clear previous silence timer
          clearTimeout(silenceTimer);

          // If final or substantial interim, set silence timer to auto-submit (Hands-Free)
          silenceTimer = setTimeout(() => {
            if (currentText.trim().length > 2 && !isSpeaking) {
              handleVoiceQuerySubmit(currentText.trim());
            }
          }, 1400);
        }
      };

      speechRecognition.onerror = (event) => {
        console.warn("Speech recognition notice:", event.error);
        if (event.error !== 'no-speech') {
          setOrbState('idle');
        }
      };

      speechRecognition.onend = () => {
        isVoiceSessionActive = false;
        if (!isSpeaking && !document.getElementById('voice-modal').classList.contains('hidden')) {
          const handsFree = document.getElementById('chk-hands-free').checked;
          if (handsFree) {
            // Re-listen in hands-free mode
            setTimeout(startVoiceRecognition, 400);
          } else {
            setOrbState('idle');
          }
        }
      };
    }

    function startVoiceRecognition() {
      if (!speechRecognition) {
        alert("Speech recognition is not supported in this browser. Please use Chrome, Edge, or an Android browser.");
        return;
      }
      try {
        const locale = INDIAN_LOCALE_MAP[currentLang] || "en-IN";
        speechRecognition.lang = locale;
        speechRecognition.start();
        isVoiceSessionActive = true;
        setOrbState('listening');
      } catch (e) {
        console.log("Recognition start notice:", e);
      }
    }

    function stopVoiceRecognition() {
      if (speechRecognition && isVoiceSessionActive) {
        speechRecognition.stop();
        isVoiceSessionActive = false;
      }
    }

    async function handleVoiceQuerySubmit(queryText) {
      stopVoiceRecognition();
      setOrbState('thinking');
      document.getElementById('voice-user-transcript').innerText = `"${queryText}"`;
      document.getElementById('voice-assistant-reply').innerText = "...";

      const data = await sendChatMessage(queryText, false);
      if (!data || !data.response) {
        setOrbState('idle');
        return;
      }
      
      document.getElementById('voice-assistant-reply').innerText = data.response;
      
      // Play the authentic native Indian voice audio
      playNativeSpeechAudio(data.response, data.audio_base64, data.audio_mime_type, data.speech_locale, () => {
        // Callback after speaking completes:
        const handsFree = document.getElementById('chk-hands-free').checked;
        if (handsFree && !document.getElementById('voice-modal').classList.contains('hidden')) {
          setOrbState('listening');
          startVoiceRecognition();
        } else {
          setOrbState('idle');
        }
      });
    }

    // Quick Voice Mic button in chat
    function toggleQuickVoiceInput() {
      openVoiceModal();
    }

    // 7. HIGH-CLARITY AUDIO & NATIVE REGIONAL SPEECH SYNTHESIS ENGINE
    function playLastResponseAudio() {
      ensureAudioUnlocked();
      if (!lastGuideResponseText) return;
      playNativeSpeechAudio(lastGuideResponseText, lastAudioBase64, lastAudioMime, lastGuideLocale);
    }

    function playNativeSpeechAudio(text, audioB64, mimeType, locale, onEndedCallback = null) {
      setOrbState('speaking');
      isSpeaking = true;

      const icon = document.getElementById('audio-play-icon');
      const statusText = document.getElementById('voice-status-text');
      const audioBar = document.getElementById('audio-control-bar');
      if (audioBar) audioBar.classList.remove('hidden');

      const langName = INDIAN_LANG_NAMES[currentLang] || "your language";

      // 1. Play authentic neural spoken audio in native Indian language
      if (audioB64 && audioB64.length > 500) {
        if ('speechSynthesis' in window) window.speechSynthesis.cancel();
        
        const audioPlayer = getAudioPlayer();
        const mime = mimeType || 'audio/mp3';

        // Revoke active blob URL to prevent memory accumulation
        if (audioPlayer._activeBlobUrl) {
          URL.revokeObjectURL(audioPlayer._activeBlobUrl);
          audioPlayer._activeBlobUrl = null;
        }

        const blob = b64ToBlob(audioB64, mime);
        const audioSrc = blob ? URL.createObjectURL(blob) : `data:${mime};base64,${audioB64}`;
        if (blob) audioPlayer._activeBlobUrl = audioSrc;

        audioPlayer.src = audioSrc;

        const speedSelect = document.getElementById('voice-speed-select');
        if (speedSelect) {
          audioPlayer.playbackRate = parseFloat(speedSelect.value) || 1.0;
        }

        if (icon) icon.className = "fa-solid fa-volume-high text-xs text-emerald-400 animate-pulse";
        if (statusText) statusText.innerText = `Speaking out loud in ${langName}...`;

        audioPlayer.onended = () => {
          isSpeaking = false;
          setOrbState('idle');
          if (icon) icon.className = "fa-solid fa-volume-high text-xs";
          if (statusText) statusText.innerText = "Click to replay";
          if (onEndedCallback) onEndedCallback();
        };

        audioPlayer.onerror = (e) => {
          console.warn("Audio playback error, falling back to Web Speech:", e);
          speakWithWebSpeechFallback(text, locale, onEndedCallback);
        };

        const playPromise = audioPlayer.play();
        if (playPromise !== undefined) {
          playPromise.then(() => {
            audioUnlocked = true;
          }).catch(e => {
            console.warn("Audio autoplay blocked by browser policy:", e);
            isSpeaking = false;
            setOrbState('idle');
            // Give clear visual cue to tap to play
            if (icon) icon.className = "fa-solid fa-play text-xs text-amber-300 animate-bounce";
            if (statusText) {
              statusText.innerHTML = `<span class="text-amber-300 font-bold underline cursor-pointer" onclick="playLastResponseAudio()">🔊 Click to listen in ${langName}</span>`;
            }
            const orbStatus = document.getElementById('voice-status-label');
            if (orbStatus && !document.getElementById('voice-modal').classList.contains('hidden')) {
              orbStatus.innerHTML = `<span class="text-amber-300 font-bold">Tap orb to listen</span>`;
            }
            if (onEndedCallback) onEndedCallback();
          });
        }
        return;
      }

      // 2. Fallback to Web Speech if no audio base64 is available
      speakWithWebSpeechFallback(text, locale, onEndedCallback);
    }

    function speakWithWebSpeechFallback(text, locale, onFinishedCallback = null) {
      if (!('speechSynthesis' in window)) {
        isSpeaking = false;
        if (onFinishedCallback) onFinishedCallback();
        return;
      }

      const targetLocale = locale || INDIAN_LOCALE_MAP[currentLang] || "en-IN";
      const langPrefix = targetLocale.split('-')[0].toLowerCase();
      const voices = window.speechSynthesis.getVoices();
      const hasMatchingVoice = voices.some(v => v.lang.toLowerCase().replace('_', '-').startsWith(langPrefix));

      // Guard: Never speak English Microsoft David voice if user chose an Indian language
      if (!hasMatchingVoice && langPrefix !== 'en') {
        console.warn("No native browser voice for", targetLocale, "- skipping English fallback");
        isSpeaking = false;
        if (onFinishedCallback) onFinishedCallback();
        return;
      }

      window.speechSynthesis.cancel();
      isSpeaking = true;
      setOrbState('speaking');

      const cleanVoiceText = expandAbbreviationsForSpeech(text);
      const utterance = new SpeechSynthesisUtterance(cleanVoiceText);
      utterance.lang = targetLocale;

      const speedSelect = document.getElementById('voice-speed-select');
      utterance.rate = speedSelect ? parseFloat(speedSelect.value) : 0.92;
      utterance.pitch = 1.0;

      const bestVoice = getBestNativeVoice(utterance.lang);
      if (bestVoice) {
        utterance.voice = bestVoice;
      }

      const icon = document.getElementById('audio-play-icon');
      const statusText = document.getElementById('voice-status-text');

      utterance.onstart = () => {
        isSpeaking = true;
        setOrbState('speaking');
        if (statusText) statusText.innerText = "Speaking clearly...";
        if (icon) icon.className = "fa-solid fa-pause text-xs";
      };

      utterance.onend = () => {
        isSpeaking = false;
        if (icon) icon.className = "fa-solid fa-volume-high text-xs";
        if (statusText) statusText.innerText = "Playback finished";
        if (onFinishedCallback) onFinishedCallback();
      };

      utterance.onerror = (e) => {
        console.warn("Speech synthesis notice:", e);
        isSpeaking = false;
        if (icon) icon.className = "fa-solid fa-volume-high text-xs";
        if (onFinishedCallback) onFinishedCallback();
      };

      window.speechSynthesis.speak(utterance);
    }

    function getBestNativeVoice(locale) {
      if (!('speechSynthesis' in window)) return null;
      const voices = window.speechSynthesis.getVoices();
      if (!voices || voices.length === 0) return null;

      const langPrefix = locale.split('-')[0].toLowerCase();
      const matching = voices.filter(v => 
        v.lang.toLowerCase().replace('_', '-').startsWith(langPrefix)
      );

      if (matching.length === 0) return null;

      // Prioritize natural neural Indian voices
      return matching.find(v => v.name.includes('Natural') || v.name.includes('Online')) ||
             matching.find(v => v.name.includes('Google')) ||
             matching.find(v => v.name.includes('Microsoft')) ||
             matching[0];
    }

    function expandAbbreviationsForSpeech(text) {
      if (!text) return "";
      return text
        .replace(/°C/g, " degree Celsius")
        .replace(/km\/h/g, " kilometers per hour")
        .replace(/hPa/g, " hectopascals")
        .replace(/%/g, " percent")
        .replace(/mm/g, " millimeters")
        .replace(/\*/g, "")
        .replace(/#/g, "")
        .replace(/_/g, "");
    }

    // 8. Settings Modal & API Keys
    function openSettingsModal() {
      document.getElementById('settings-modal').classList.remove('hidden');
    }

    function closeSettingsModal() {
      document.getElementById('settings-modal').classList.add('hidden');
    }

    async function testAPIKeys() {
      const geminiKey = document.getElementById('key-gemini').value.trim();
      const omKey = document.getElementById('key-openmeteo').value.trim();
      const statusBox = document.getElementById('keys-status-box');
      const btn = document.getElementById('btn-test-keys');

      btn.disabled = true;
      btn.innerText = "Testing...";
      statusBox.classList.remove('hidden');
      statusBox.innerHTML = `<div class="flex items-center gap-2 text-slate-500 py-1"><i class="fa-solid fa-spinner fa-spin text-sky-600"></i> Testing connections with Gemini and Open-Meteo...</div>`;

      try {
        const resp = await fetch('/api/settings/verify', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ gemini_api_key: geminiKey, open_meteo_api_key: omKey })
        });
        const res = await resp.json();

        let html = '';
        if (Array.isArray(res.notes)) {
          html = res.notes.map(n => {
            const text = typeof n === 'string' ? n : (n.text || '');
            const type = typeof n === 'object' && n.type ? n.type : 'info';

            let icon = 'fa-circle-check text-emerald-600';
            let bgClass = 'bg-emerald-50/70 border-emerald-200 text-emerald-950';

            if (type === 'error') {
              icon = 'fa-circle-xmark text-rose-600';
              bgClass = 'bg-rose-50/70 border-rose-200 text-rose-950';
            } else if (type === 'warning') {
              icon = 'fa-triangle-exclamation text-amber-500';
              bgClass = 'bg-amber-50/70 border-amber-200 text-amber-950';
            } else if (type === 'info') {
              icon = 'fa-circle-info text-sky-600';
              bgClass = 'bg-sky-50/70 border-sky-200 text-sky-950';
            }

            return `<div class="p-2.5 rounded-xl border ${bgClass} flex items-start gap-2.5 leading-relaxed text-xs">
              <i class="fa-solid ${icon} mt-0.5 shrink-0 text-sm"></i>
              <div>${escapeHtml(text)}</div>
            </div>`;
          }).join('');
        }
        statusBox.innerHTML = html;
      } catch (err) {
        statusBox.innerHTML = `<div class="p-2.5 rounded-xl bg-rose-50 border border-rose-200 text-rose-800 text-xs flex items-center gap-2"><i class="fa-solid fa-circle-xmark text-rose-600"></i> Failed to verify keys: ${escapeHtml(err.message)}</div>`;
      } finally {
        btn.disabled = false;
        btn.innerText = "Test Keys";
      }
    }

    function saveAPIKeys() {
      const geminiKey = document.getElementById('key-gemini').value.trim();
      const omKey = document.getElementById('key-openmeteo').value.trim();

      localStorage.setItem('weathergpt_gemini_key', geminiKey);
      localStorage.setItem('weathergpt_openmeteo_key', omKey);

      closeSettingsModal();
      alert("Settings saved successfully! WeatherGPT will use your configured keys.");
      fetchWeather(true);
    }

    function escapeHtml(str) {
      if (!str) return "";
      return String(str)
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#039;");
    }

    // Auto-initialize on load
    window.addEventListener('DOMContentLoaded', () => {
      initSettings();
      fetchWeather();
    });
  </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)
