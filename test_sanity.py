"""Sanity and regression tests for flight difficulty feature engineering.

Supports direct execution or pytest discovery.
"""
from pathlib import Path
import pandas as pd
import difficulty_scoring as ds

CONFIG_PATH = 'config.yaml'


def assert_column(df, col):
    assert col in df.columns, f"Missing expected column: {col}"


def load_master():
    config = ds.load_config(CONFIG_PATH)
    flights = pd.read_csv('resources/Flight_Level_Data.csv')
    bags = pd.read_csv('resources/Bag_Level_Data.csv')
    pnr_f = pd.read_csv('resources/PNR_Flight_Level_Data.csv')
    pnr_r = pd.read_csv('resources/PNR_Remark_Level_Data.csv')
    airports = pd.read_csv('resources/Airports_Data.csv')
    master = ds.engineer_features(flights, bags, pnr_f, pnr_r, airports, config=config)
    return master, config


def test_priority2_features():
    master, _ = load_master()
    for col in ['ssr_mobility_count','ssr_special_handling_count','bag_issue_lead_mean','bag_late_issue_ratio','bag_per_seat_ratio']:
        assert_column(master, col)


def test_priority3_features_present():
    master, _ = load_master()
    for col in ['load_factor_volatility','ground_time_pressure_volatility','slot_congestion_count','transfer_ground_interaction','load_pressure_interaction','weather_severity_index']:
        assert_column(master, col)


def test_scoring_integrity():
    master, config = load_master()
    weights = config.get('weights', ds.DEFAULT_WEIGHTS)
    feature_cols = [f for f in weights.keys() if f in master.columns]
    scored = ds.per_day_scale_and_score(master, weights=weights, feature_cols=feature_cols,
                                        difficult_threshold=config.get('thresholds',{}).get('difficult',0.25),
                                        medium_threshold=config.get('thresholds',{}).get('medium',0.75))
    assert scored['difficulty_score'].between(0,1).any(), 'Scores not within expected scaled combination range'
    assert scored['daily_rank'].min() == 1, 'Daily rank should start at 1'
    assert set(scored['difficulty_class'].unique()).issubset({'Difficult','Medium','Easy'})


if __name__ == '__main__':
    # Execute a subset of tests manually
    test_priority2_features()
    test_priority3_features_present()
    test_scoring_integrity()
    print('All sanity tests passed (manual run).')
