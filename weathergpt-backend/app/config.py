import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env file from project root or backend root
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent
env_file = project_root / ".env"
if env_file.exists():
    load_dotenv(dotenv_path=env_file)
else:
    load_dotenv()

class Settings:
    PROJECT_NAME: str = "WeatherGPT Backend"
    VERSION: str = "1.0.0"
    
    BASE_DIR: Path = project_root
    APP_DIR: Path = current_dir
    MODELS_DIR: Path = current_dir / "models"
    DATA_DIR: Path = current_dir / "data"
    
    DB_PATH: Path = BASE_DIR / os.getenv("DB_PATH", "weather_cache.db")
    CACHE_EXPIRY_HOURS: int = int(os.getenv("CACHE_EXPIRY_HOURS", "3"))
    
    BUILTIN_GEMINI_KEY: str = "AIzaSyCx80ru6-RXeTi3GvqkFsMVyMf-vpgIoVw"
    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip() or BUILTIN_GEMINI_KEY
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    OPEN_METEO_API_KEY: str = os.getenv("OPEN_METEO_API_KEY", "")
    OPEN_METEO_URL: str = "https://api.open-meteo.com/v1/forecast"
    OPEN_METEO_CUSTOMER_URL: str = "https://customer-api.open-meteo.com/v1/forecast"
    OPEN_METEO_GEOCODING_URL: str = "https://geocoding-api.open-meteo.com/v1/search"
    
    DEFAULT_LAT: float = float(os.getenv("DEFAULT_LAT", "13.0827"))
    DEFAULT_LON: float = float(os.getenv("DEFAULT_LON", "80.2707"))
    
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))

    def get_open_meteo_endpoint(self, key_override: str = "") -> str:
        key = key_override or self.OPEN_METEO_API_KEY
        if key:
            return self.OPEN_METEO_CUSTOMER_URL
        return self.OPEN_METEO_URL

settings = Settings()
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
settings.DATA_DIR.mkdir(parents=True, exist_ok=True)
