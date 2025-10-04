"""Quick demo to validate weather enrichment logic with synthetic dataset.
Run:
  python weather_enrichment_demo.py
"""
import pandas as pd
import weather_enrichment as wx

flights = pd.read_csv('resources/Flight_Level_Data_weather_demo.csv')
enriched = wx.enrich_with_weather(flights)
print(enriched[['flight_number','scheduled_departure_station_code','scheduled_departure_datetime_local','weather_severity_index']])
