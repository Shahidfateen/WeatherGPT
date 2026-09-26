"""Offline Caching Engine using SQLite for WeatherGPT."""

import sqlite3
import json
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List
from pathlib import Path
from app.config import settings

class CacheService:
    def __init__(self, db_path: Path = settings.DB_PATH, expiry_hours: int = settings.CACHE_EXPIRY_HOURS):
        self.db_path = db_path
        self.expiry_seconds = expiry_hours * 3600
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Create required tables if they don't exist."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            # Weather Forecast Cache Table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS weather_forecast_cache (
                    coord_key TEXT PRIMARY KEY,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    fetched_at INTEGER NOT NULL,
                    expires_at INTEGER NOT NULL,
                    payload_json TEXT NOT NULL
                )
            """)
            # Advisory Log Table for auditing and offline review
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS advisory_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    query TEXT NOT NULL,
                    language TEXT NOT NULL,
                    advisory_text TEXT NOT NULL,
                    risk_level TEXT,
                    pesticide_spray_safe INTEGER
                )
            """)
            conn.commit()

    @staticmethod
    def _coord_key(lat: float, lon: float) -> str:
        """Round coordinates to 2 decimals (~1.1 km grid) for efficient caching."""
        return f"{lat:.2f}:{lon:.2f}"

    def get_cached_forecast(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Retrieve valid cached weather forecast if not expired."""
        key = self._coord_key(lat, lon)
        current_time = int(time.time())
        
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT payload_json, expires_at FROM weather_forecast_cache
                WHERE coord_key = ?
            """, (key,))
            row = cursor.fetchone()
            if row:
                expires_at = row["expires_at"]
                if current_time <= expires_at:
                    try:
                        return json.loads(row["payload_json"])
                    except json.JSONDecodeError:
                        return None
        return None

    def get_any_cached_forecast(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        """Retrieve cached forecast even if expired (used for offline fallback)."""
        key = self._coord_key(lat, lon)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT payload_json FROM weather_forecast_cache
                WHERE coord_key = ?
            """, (key,))
            row = cursor.fetchone()
            if row:
                try:
                    return json.loads(row["payload_json"])
                except json.JSONDecodeError:
                    pass
            # If exact key not found, return the most recent forecast cached anywhere
            cursor.execute("""
                SELECT payload_json FROM weather_forecast_cache
                ORDER BY fetched_at DESC LIMIT 1
            """)
            fallback_row = cursor.fetchone()
            if fallback_row:
                try:
                    return json.loads(fallback_row["payload_json"])
                except json.JSONDecodeError:
                    return None
        return None

    def save_forecast(self, lat: float, lon: float, payload: Dict[str, Any]):
        """Save or update weather forecast in SQLite cache."""
        key = self._coord_key(lat, lon)
        current_time = int(time.time())
        expires_at = current_time + self.expiry_seconds
        payload_str = json.dumps(payload)

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR REPLACE INTO weather_forecast_cache 
                (coord_key, latitude, longitude, fetched_at, expires_at, payload_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (key, round(lat, 4), round(lon, 4), current_time, expires_at, payload_str))
            conn.commit()

    def log_advisory(self, lat: float, lon: float, query: str, language: str, 
                     advisory_text: str, risk_level: str, pesticide_spray_safe: bool):
        """Log farmer advisory requests and responses."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO advisory_logs 
                (latitude, longitude, query, language, advisory_text, risk_level, pesticide_spray_safe)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (round(lat, 4), round(lon, 4), query, language, advisory_text, risk_level, int(pesticide_spray_safe)))
            conn.commit()

    def get_recent_advisories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Retrieve recent advisory logs."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT id, created_at, latitude, longitude, query, language, advisory_text, risk_level, pesticide_spray_safe
                FROM advisory_logs
                ORDER BY id DESC LIMIT ?
            """, (limit,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

# Singleton instance
cache_service = CacheService()
