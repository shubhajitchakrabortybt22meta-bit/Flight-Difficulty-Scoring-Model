"""Flight Difficulty Scoring Module

Provides a daily-reset scoring pipeline that:
 1. Engineers operational + customer service features.
 2. Performs per-day min-max scaling of feature set.
 3. Applies configurable weights to derive a difficulty score.
 4. Ranks flights within each day (dense rank, 1 = most difficult).
 5. Classifies flights into Difficult / Medium / Easy based on rank distribution.

Assumptions / Notes:
 - Scaling is executed per day so each calendar day's difficulty distribution is independent.
 - If a feature has zero variance within a day (min=max), its scaled contribution is 0 for all flights that day.
 - Bag types expected: Checked, Transfer, Hot Transfer ("Hot Transfer" treated as subset if present; if not, counts = 0).
 - SSR counts derived from PNR remarks joined via record_locator.
 - Passenger composition aggregated from PNR flight-level data (children, stroller users, lap children, total pax).
 - International flag derived by joining arrival station to airports file (iso_country_code != 'US').

CLI Usage (after repo restructuring):
    python difficulty_scoring.py \
            --flights resources/Flight_Level_Data.csv \
            --bags resources/Bag_Level_Data.csv \
            --pnr_f resources/PNR_Flight_Level_Data.csv \
            --pnr_r resources/PNR_Remark_Level_Data.csv \
            --airports resources/Airports_Data.csv \
            --output final_result_data/flight_difficulty_scores.csv

"""
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from typing import Dict, List, Sequence

FLIGHT_KEY = ["company_id", "flight_number", "scheduled_departure_date_local"]

# Default feature weights (must sum to 1; if not, will be normalized)
DEFAULT_WEIGHTS = {
    'ground_time_pressure': 0.15,
    'actual_turn_deficit': 0.07,
    'turn_deficit_ratio': 0.05,
    'total_bags': 0.06,
    'transfer_ratio': 0.06,
    'hot_transfer_count': 0.04,
    'total_pax': 0.08,
    'load_factor': 0.06,
    'basic_economy_ratio': 0.05,
    'booking_lead_days_mean': 0.04,
    'late_booking_ratio': 0.04,
    'children': 0.04,
    'lap_children': 0.03,
    'stroller_users': 0.03,
    'ssr_count': 0.10,
    'is_international': 0.04,
    'widebody_flag': 0.06,
}

FEATURE_ORDER = list(DEFAULT_WEIGHTS.keys())


def _read_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    return df


def _parse_dates(df: pd.DataFrame, date_cols: Sequence[str]):
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors='coerce')


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(weights.values())
    if not np.isclose(total, 1.0):
        return {k: v / total for k, v in weights.items()}
    return weights


def engineer_features(flights: pd.DataFrame,
                      bags: pd.DataFrame,
                      pnr_flight: pd.DataFrame,
                      pnr_remark: pd.DataFrame,
                      airports: pd.DataFrame) -> pd.DataFrame:
    """Return master DataFrame with engineered features ready for scaling/scoring."""
    # Base copy
    master = flights.copy()

    # --- Passenger Features ---
    # Booking lead time features
    if 'pnr_creation_date' in pnr_flight.columns:
        pnr_flight = pnr_flight.copy()
        pnr_flight['pnr_creation_date'] = pd.to_datetime(pnr_flight['pnr_creation_date'], errors='coerce')
        # derive scheduled date if possible (string vs date objects handled upstream)
        if 'scheduled_departure_date_local' in pnr_flight.columns:
            sched_series = pd.to_datetime(pnr_flight['scheduled_departure_date_local'], errors='coerce')
            pnr_flight['booking_lead_days'] = (sched_series - pnr_flight['pnr_creation_date']).dt.days
        else:
            pnr_flight['booking_lead_days'] = np.nan
    else:
        pnr_flight['booking_lead_days'] = np.nan

    # Late booking indicator (within X days threshold)
    LATE_THRESHOLD_DAYS = 3
    pnr_flight['late_booking_flag'] = (pnr_flight['booking_lead_days'] <= LATE_THRESHOLD_DAYS).astype(int)

    pax_agg = pnr_flight.groupby(FLIGHT_KEY).agg(
        total_pax=('total_pax', 'sum'),
        basic_economy_pax=('basic_economy_pax', 'sum') if 'basic_economy_pax' in pnr_flight.columns else ('total_pax', 'sum'),
        children=('is_child', lambda s: (s == 'Y').sum()),
        stroller_users=('is_stroller_user', lambda s: (s == 'Y').sum()),
        lap_children=('lap_child_count', 'sum'),
        booking_lead_days_mean=('booking_lead_days', 'mean'),
        booking_lead_days_median=('booking_lead_days', 'median'),
        late_booking_ratio=('late_booking_flag', 'mean'),
    ).reset_index()
    master = master.merge(pax_agg, on=FLIGHT_KEY, how='left')

    # --- Baggage Features ---
    if 'bag_type' in bags.columns:
        bags = bags.copy()
        bags['bag_type'] = bags['bag_type'].replace({'Origin': 'Checked'})
        bag_counts = bags.groupby(FLIGHT_KEY + ['bag_type'], dropna=False).size().unstack(fill_value=0)
        for c in ['Checked', 'Transfer', 'Hot Transfer']:
            if c not in bag_counts.columns:
                bag_counts[c] = 0
        bag_counts['hot_transfer_count'] = bag_counts['Hot Transfer']
        bag_counts['total_transfer_bags'] = bag_counts['Transfer'] + bag_counts['Hot Transfer']
        bag_counts['total_bags'] = bag_counts['Checked'] + bag_counts['total_transfer_bags']
        bag_counts['transfer_ratio'] = bag_counts['total_transfer_bags'] / bag_counts['total_bags'].replace(0, np.nan)
        master = master.merge(bag_counts.reset_index(), on=FLIGHT_KEY, how='left')
    else:
        master['hot_transfer_count'] = 0
        master['total_transfer_bags'] = 0
        master['total_bags'] = 0
        master['transfer_ratio'] = np.nan

    # --- SSR Features ---
    pnr_keys = pnr_flight[FLIGHT_KEY + ['record_locator']].drop_duplicates()
    ssr = pnr_remark.merge(pnr_keys, on='record_locator', how='inner')
    # If flight_number missing after merge (e.g., remarks lacked it), fall back to distributing by record_locator occurrences
    if not set(FLIGHT_KEY).issubset(ssr.columns):
        # Count remarks per record_locator then merge back and aggregate to flight
        remark_counts = ssr.groupby('record_locator').size().reset_index(name='ssr_count')
        ssr_temp = pnr_keys.merge(remark_counts, on='record_locator', how='left').fillna({'ssr_count': 0})
        ssr_counts = ssr_temp.groupby(FLIGHT_KEY)['ssr_count'].sum().reset_index()
    else:
        ssr_counts = ssr.groupby(FLIGHT_KEY).size().reset_index(name='ssr_count')
    master = master.merge(ssr_counts, on=FLIGHT_KEY, how='left')

    # --- Ground Time Pressure ---
    if {'minimum_turn_minutes', 'scheduled_ground_time_minutes'}.issubset(master.columns):
        master['ground_time_pressure'] = (master['minimum_turn_minutes'] - master['scheduled_ground_time_minutes']).clip(lower=0)
    else:
        master['ground_time_pressure'] = 0

    # Actual turn deficit and ratio
    if {'minimum_turn_minutes', 'actual_ground_time_minutes'}.issubset(master.columns):
        master['actual_turn_deficit'] = (master['minimum_turn_minutes'] - master['actual_ground_time_minutes']).clip(lower=0)
        master['turn_deficit_ratio'] = np.where(master['minimum_turn_minutes'] > 0,
                                                master['actual_ground_time_minutes'] / master['minimum_turn_minutes'], np.nan)
    else:
        master['actual_turn_deficit'] = 0
        master['turn_deficit_ratio'] = np.nan

    # --- International Flag ---
    airports_small = airports[['airport_iata_code', 'iso_country_code']].drop_duplicates()
    master = master.merge(airports_small, left_on='scheduled_arrival_station_code', right_on='airport_iata_code', how='left')
    master['is_international'] = (master['iso_country_code'] != 'US').astype(int)

    # --- Load Factor ---
    master['total_pax'] = master['total_pax'].fillna(0)
    master['load_factor'] = master['total_pax'] / master['total_seats'].replace(0, np.nan)
    # basic economy ratio
    if 'basic_economy_pax' in master.columns:
        master['basic_economy_ratio'] = np.where(master['total_pax'] > 0, master['basic_economy_pax'] / master['total_pax'], np.nan)
    else:
        master['basic_economy_ratio'] = np.nan

    # widebody flag (threshold or fleet_type pattern)
    WIDEBODY_SEAT_THRESHOLD = 250
    master['widebody_flag'] = ((master.get('total_seats', 0) >= WIDEBODY_SEAT_THRESHOLD) |
                               master.get('fleet_type','').astype(str).str.contains('767|777|787|330|350', case=False, na=False)).astype(int)

    # Fill numeric NaNs with 0 (except ratios keep NaN for now; will handle after scaling)
    for col in ['children', 'stroller_users', 'lap_children', 'ssr_count', 'hot_transfer_count',
                'total_transfer_bags', 'total_bags', 'ground_time_pressure', 'actual_turn_deficit']:
        if col in master.columns:
            master[col] = master[col].fillna(0)

    # Fill ratios that might remain NaN if division by 0
    ratio_cols = ['transfer_ratio','basic_economy_ratio','turn_deficit_ratio']
    for rc in ratio_cols:
        if rc in master.columns:
            master[rc] = master[rc].fillna(0)

    return master


def per_day_scale_and_score(master: pd.DataFrame,
                            weights: Dict[str, float] = None,
                            date_col: str = 'scheduled_departure_date_local',
                            feature_cols: Sequence[str] = FEATURE_ORDER,
                            difficult_threshold: float = 0.25,
                            medium_threshold: float = 0.75) -> pd.DataFrame:
    """Scale features per day and compute scores, ranks, and classification.

    Parameters:
      master: DataFrame with engineered features
      weights: dict of feature weights; normalized if not summing to 1
      date_col: name of the date column (Date or string convertible)
      feature_cols: ordered list of feature names to include in scoring
      difficult_threshold: top proportion (by difficulty) to classify as 'Difficult'
      medium_threshold: cumulative proportion (after Difficult) to classify as 'Medium'
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS
    weights = _normalize_weights(weights)

    # Ensure date column is date
    master = master.copy()
    master[date_col] = pd.to_datetime(master[date_col], errors='coerce').dt.date

    missing = [c for c in feature_cols if c not in master.columns]
    if missing:
        raise KeyError(f"Missing feature columns for scoring: {missing}")

    # Placeholder for scaled values
    scaled_cols = {c: [] for c in feature_cols}
    scores = []
    ranks = []
    classifications = []

    grouped = master.groupby(date_col, dropna=False)

    for day, group in grouped:
        g = group.copy()
        # Per-day min-max scaling
        for col in feature_cols:
            col_values = g[col].astype(float)
            min_v = col_values.min()
            max_v = col_values.max()
            if pd.isna(min_v) or pd.isna(max_v) or max_v - min_v == 0:
                scaled = np.zeros(len(g))  # no variation or all NaN
            else:
                scaled = (col_values - min_v) / (max_v - min_v)
            scaled_cols[col].extend(scaled)
        # Weighted score
        feature_matrix = np.column_stack([scaled_cols[col][-len(g):] for col in feature_cols])
        weight_vector = np.array([weights[c] for c in feature_cols])
        day_scores = feature_matrix @ weight_vector
        # Rank (1 is most difficult => highest score)
        # Use 'dense' ranking; ties share rank, next rank increments by 1.
        order = (-day_scores).argsort(kind='stable')
        dense_rank = np.empty(len(g), dtype=int)
        dense_rank[order] = pd.Series(day_scores[order]).rank(method='dense', ascending=False).astype(int).values
        # Classification via rank percentile
        n = len(g)
        # Convert rank to zero-based position percentile
        # Note: lower rank number = more difficult
        percentile = (dense_rank - 1) / max(n - 1, 1)
        day_classes = []
        for p in percentile:
            if p < difficult_threshold:
                day_classes.append('Difficult')
            elif p < medium_threshold:
                day_classes.append('Medium')
            else:
                day_classes.append('Easy')
        scores.extend(day_scores)
        ranks.extend(dense_rank)
        classifications.extend(day_classes)

    # Attach results
    master['difficulty_score'] = scores
    master['daily_rank'] = ranks
    master['difficulty_class'] = classifications

    # Store scaled columns (optional: suffix)
    for col in feature_cols:
        master[f'{col}_scaled'] = scaled_cols[col]

    return master


def compute_difficulty_scores(flights_path: str,
                              bags_path: str,
                              pnr_flight_path: str,
                              pnr_remark_path: str,
                              airports_path: str,
                              weights: Dict[str, float] | None = None,
                              output_csv: str | None = None) -> pd.DataFrame:
    flights = _read_csv(flights_path)
    bags = _read_csv(bags_path)
    pnr_f = _read_csv(pnr_flight_path)
    pnr_r = _read_csv(pnr_remark_path)
    airports = _read_csv(airports_path)

    # Parse relevant datetimes
    _parse_dates(flights, [
        'scheduled_departure_datetime_local', 'scheduled_arrival_datetime_local',
        'actual_departure_datetime_local', 'actual_arrival_datetime_local'
    ])

    # Harmonize date column types across input frames for key
    for df in (flights, bags, pnr_f):
        if 'scheduled_departure_date_local' in df.columns:
            df['scheduled_departure_date_local'] = pd.to_datetime(df['scheduled_departure_date_local'], errors='coerce').dt.date

    master = engineer_features(flights, bags, pnr_f, pnr_r, airports)
    scored = per_day_scale_and_score(master, weights=weights)

    if output_csv:
        scored.to_csv(output_csv, index=False)

    return scored


def _build_arg_parser():
    p = argparse.ArgumentParser(description="Compute per-day flight difficulty scores")
    p.add_argument('--flights', required=True)
    p.add_argument('--bags', required=True)
    p.add_argument('--pnr_f', required=True, help='PNR flight level file')
    p.add_argument('--pnr_r', required=True, help='PNR remark level file')
    p.add_argument('--airports', required=True)
    p.add_argument('--output', required=False, default='final_result_data/flight_difficulty_scores.csv')
    p.add_argument('--difficult_threshold', type=float, default=0.25, help='Top proportion classified as Difficult')
    p.add_argument('--medium_threshold', type=float, default=0.75, help='Proportion boundary for Medium (rest Easy)')
    return p


def main():
    args = _build_arg_parser().parse_args()
    scores = compute_difficulty_scores(
        flights_path=args.flights,
        bags_path=args.bags,
        pnr_flight_path=args.pnr_f,
        pnr_remark_path=args.pnr_r,
        airports_path=args.airports,
        weights=DEFAULT_WEIGHTS,
        output_csv=args.output,
    )
    # Simple summary output
    print("Computed difficulty scores for {} flights.".format(len(scores)))
    print(scores[['company_id','flight_number','scheduled_departure_date_local','difficulty_score','daily_rank','difficulty_class']].head())
    print(f"Output written to: {args.output}")


if __name__ == '__main__':
    main()
