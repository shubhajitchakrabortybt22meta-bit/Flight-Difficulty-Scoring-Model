"""Weather enrichment utilities.

Attempts to read a weather data CSV (resources/Weather_Data.csv) with columns:
  station_code, observation_time, weather_severity_index

Joins severity index to flights on (scheduled_departure_station_code, scheduled_departure_date_local) using nearest hour.
If file absent or columns missing, returns original DataFrame with 'weather_severity_index' filled with 0.
"""
from __future__ import annotations
import pandas as pd
from pathlib import Path

WEATHER_FILE = Path('resources/Weather_Data.csv')

def enrich_with_weather(flights: pd.DataFrame) -> pd.DataFrame:
    if 'scheduled_departure_datetime_local' not in flights.columns:
        flights['weather_severity_index'] = 0
        return flights
    if not WEATHER_FILE.exists():
        flights['weather_severity_index'] = 0
        return flights
    try:
        weather = pd.read_csv(WEATHER_FILE)
    except Exception:
        flights['weather_severity_index'] = 0
        return flights
    needed = {'station_code','observation_time','weather_severity_index'}
    if not needed.issubset(weather.columns):
        flights['weather_severity_index'] = 0
        return flights
    weather['observation_time'] = pd.to_datetime(weather['observation_time'], errors='coerce')
    flights = flights.copy()
    flights['scheduled_departure_datetime_local'] = pd.to_datetime(flights['scheduled_departure_datetime_local'], errors='coerce')
    # Use lowercase 'h' to avoid FutureWarning (floor hourly)
    flights['dep_hour'] = flights['scheduled_departure_datetime_local'].dt.floor('h').dt.tz_localize(None)
    weather['obs_hour'] = weather['observation_time'].dt.floor('h').dt.tz_localize(None)
    # Aggregate multiple reports per hour per station
    w_hour = weather.groupby(['station_code','obs_hour']).agg(weather_severity_index=('weather_severity_index','mean')).reset_index()
    merged = flights.merge(w_hour, left_on=['scheduled_departure_station_code','dep_hour'], right_on=['station_code','obs_hour'], how='left')
    merged['weather_severity_index'] = merged['weather_severity_index'].fillna(0)
    merged.drop(columns=['station_code','obs_hour'], errors='ignore', inplace=True)
    return merged
