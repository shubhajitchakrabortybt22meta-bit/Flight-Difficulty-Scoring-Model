import pandas as pd
import numpy as np
from pathlib import Path

# EDA Script answering specified questions
# Dataset files are expected in the current working directory.

FLIGHT_FILE = 'Flight_Level_Data.csv'
BAG_FILE = 'Bag_Level_Data.csv'
PNR_FLIGHT_FILE = 'PNR_Flight_Level_Data.csv'
PNR_REMARK_FILE = 'PNR_Remark_Level_Data.csv'

# Flight key for joins
FLIGHT_KEY = ['company_id', 'flight_number', 'scheduled_departure_date_local']


def load_data():
    flights = pd.read_csv(FLIGHT_FILE)
    bags = pd.read_csv(BAG_FILE)
    pnr_f = pd.read_csv(PNR_FLIGHT_FILE)
    remarks = pd.read_csv(PNR_REMARK_FILE)

    # Strip whitespace from columns
    for df in [flights, bags, pnr_f, remarks]:
        df.columns = df.columns.str.strip()

    # Parse dates / datetimes
    datetime_cols_flights = [
        'scheduled_departure_datetime_local', 'scheduled_arrival_datetime_local',
        'actual_departure_datetime_local', 'actual_arrival_datetime_local'
    ]
    for c in datetime_cols_flights:
        if c in flights.columns:
            flights[c] = pd.to_datetime(flights[c], errors='coerce')

    # Convert scheduled_departure_date_local into date (ensure consistent type across files)
    for df in [flights, bags, pnr_f]:
        if 'scheduled_departure_date_local' in df.columns:
            # Handle both YYYY-MM-DD and M/D/YYYY
            df['scheduled_departure_date_local'] = pd.to_datetime(
                df['scheduled_departure_date_local'], errors='coerce'
            ).dt.date

    # PNR creation date & bag tag issue date (optional parsing)
    for date_col in ['pnr_creation_date', 'bag_tag_issue_date']:
        if date_col in pnr_f.columns:
            pnr_f[date_col] = pd.to_datetime(pnr_f[date_col], errors='coerce')
        if date_col in bags.columns:
            bags[date_col] = pd.to_datetime(bags[date_col], errors='coerce')
        if date_col in remarks.columns:
            remarks[date_col] = pd.to_datetime(remarks[date_col], errors='coerce')

    return flights, bags, pnr_f, remarks


def compute_delays(flights: pd.DataFrame) -> pd.DataFrame:
    df = flights.copy()
    if 'scheduled_departure_datetime_local' in df.columns and 'actual_departure_datetime_local' in df.columns:
        df['dep_delay_minutes'] = (df['actual_departure_datetime_local'] - df['scheduled_departure_datetime_local']).dt.total_seconds() / 60.0
    else:
        df['dep_delay_minutes'] = np.nan
    return df


def ground_time_analysis(flights: pd.DataFrame) -> pd.DataFrame:
    df = flights.copy()
    if {'scheduled_ground_time_minutes', 'minimum_turn_minutes'}.issubset(df.columns):
        df['ground_time_delta'] = df['scheduled_ground_time_minutes'] - df['minimum_turn_minutes']
        df['at_or_below_min_turn'] = (df['ground_time_delta'] <= 0).astype(int)
        df['within_5_min_above_min'] = ((df['ground_time_delta'] > 0) & (df['ground_time_delta'] <= 5)).astype(int)
    return df


def bag_ratio(bags: pd.DataFrame) -> pd.DataFrame:
    df = bags.copy()
    if 'bag_type' not in df.columns:
        df['bag_type'] = 'Unknown'
    # Normalize origin -> Checked if present
    df['bag_type'] = df['bag_type'].replace({'Origin': 'Checked'})
    grp = df.groupby(FLIGHT_KEY + ['bag_type']).size().unstack(fill_value=0)
    # Ensure expected columns
    for c in ['Checked', 'Transfer']:
        if c not in grp.columns:
            grp[c] = 0
    grp['total_bags'] = grp['Checked'] + grp['Transfer']
    grp['transfer_ratio'] = np.where(grp['total_bags'] > 0, grp['Transfer'] / grp['total_bags'], np.nan)
    return grp.reset_index()


def passenger_loads(flights: pd.DataFrame, pnr_f: pd.DataFrame) -> pd.DataFrame:
    # Aggregate PNR passenger counts
    pax_agg = pnr_f.groupby(FLIGHT_KEY).agg(
        total_pax=('total_pax', 'sum')
    ).reset_index()
    merged = flights.merge(pax_agg, on=FLIGHT_KEY, how='left')
    merged['total_pax'] = merged['total_pax'].fillna(0)
    if 'total_seats' in merged.columns:
        merged['load_factor'] = np.where(merged['total_seats'] > 0, merged['total_pax'] / merged['total_seats'], np.nan)
    return merged


def ssr_counts(pnr_f: pd.DataFrame, remarks: pd.DataFrame) -> pd.DataFrame:
    # Ensure flight_number types align (PNR flight_level has integers; remarks may have string)
    if 'flight_number' in remarks.columns and remarks['flight_number'].dtype != pnr_f['flight_number'].dtype:
        # Attempt safe casting
        try:
            remarks = remarks.copy()
            remarks['flight_number'] = pd.to_numeric(remarks['flight_number'], errors='coerce').astype(pnr_f['flight_number'].dtype)
        except Exception:
            pass

    base = pnr_f[FLIGHT_KEY + ['record_locator']].drop_duplicates()
    # First merge on record locator only (flight number from remarks may be partial or missing in some rows)
    ssr = remarks.merge(base, on='record_locator', how='inner')
    # If any of the flight key columns are missing after merge, drop those rows
    missing_key_cols = [col for col in FLIGHT_KEY if col not in ssr.columns]
    if missing_key_cols:
        # Fall back: merge pnr_f subset with remarks counts by record_locator
        counts = ssr.groupby('record_locator').size().reset_index(name='ssr_count')
        # Re-merge back to base to distribute counts (each flight occurrence of record_locator gets the count)
        ssr_cnt = base.merge(counts, on='record_locator', how='left').fillna({'ssr_count': 0})
        ssr_cnt = ssr_cnt.groupby(FLIGHT_KEY)['ssr_count'].sum().reset_index()
        return ssr_cnt
    # Standard path: group by full flight key
    ssr_cnt = ssr.groupby(FLIGHT_KEY).size().reset_index(name='ssr_count')
    return ssr_cnt


def regression_delay_on_ssr(load_df: pd.DataFrame, ssr_df: pd.DataFrame) -> dict:
    # Merge SSR counts
    df = load_df.merge(ssr_df, on=FLIGHT_KEY, how='left')
    df['ssr_count'] = df['ssr_count'].fillna(0)
    # Keep needed cols
    model_df = df[['dep_delay_minutes', 'ssr_count', 'load_factor']].dropna()
    model_df = model_df[(model_df['dep_delay_minutes'].abs() < 24 * 60)]  # filter extreme outliers (>1 day)
    if len(model_df) < 10:
        return {'note': 'Not enough data for regression', 'rows': len(model_df)}
    y = model_df['dep_delay_minutes'].values
    X = np.column_stack([
        np.ones(len(model_df)),
        model_df['ssr_count'].values,
        model_df['load_factor'].values
    ])
    # OLS coefficients
    beta = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    y_hat = X @ beta
    resid = y - y_hat
    sse = (resid ** 2).sum()
    sst = ((y - y.mean()) ** 2).sum()
    r2 = 1 - sse / sst if sst else np.nan
    return {
        'intercept': beta[0],
        'coef_ssr_count': beta[1],
        'coef_load_factor': beta[2],
        'r2': r2,
        'n_obs': len(model_df)
    }


def main():
    flights, bags, pnr_f, remarks = load_data()

    # Q1: Average delay & % late departures
    flights_delay = compute_delays(flights)
    avg_delay = flights_delay['dep_delay_minutes'].mean()
    pct_late = (flights_delay['dep_delay_minutes'] > 0).mean() * 100

    # Q2: Ground time vs minimum turn
    ground_df = ground_time_analysis(flights_delay)
    pct_at_or_below = ground_df.get('at_or_below_min_turn', pd.Series(dtype=int)).mean() * 100 if 'at_or_below_min_turn' in ground_df else np.nan
    pct_within5 = ground_df.get('within_5_min_above_min', pd.Series(dtype=int)).mean() * 100 if 'within_5_min_above_min' in ground_df else np.nan

    # Q3: Avg ratio transfer vs checked bags
    bag_stats = bag_ratio(bags)
    avg_transfer_ratio = bag_stats['transfer_ratio'].mean()

    # Q4: Passenger loads & correlation with operational difficulty proxy (departure delay & ground time delta)
    load_df = passenger_loads(ground_df, pnr_f)
    corr_delay = load_df[['dep_delay_minutes', 'load_factor']].corr().iloc[0, 1]
    if 'ground_time_delta' in load_df.columns:
        corr_ground_pressure = load_df[['ground_time_delta', 'load_factor']].corr().iloc[0, 1]
    else:
        corr_ground_pressure = np.nan

    # Q5: SSR vs delay controlling for load
    ssr_df = ssr_counts(pnr_f, remarks)
    reg_results = regression_delay_on_ssr(load_df, ssr_df)

    # High vs low SSR comparison
    load_ssr = load_df.merge(ssr_df, on=FLIGHT_KEY, how='left').fillna({'ssr_count': 0})
    if not load_ssr.empty:
        threshold = load_ssr['ssr_count'].quantile(0.75)
        high = load_ssr[load_ssr['ssr_count'] >= threshold]
        low = load_ssr[load_ssr['ssr_count'] == 0]
        mean_delay_high = high['dep_delay_minutes'].mean() if not high.empty else np.nan
        mean_delay_low = low['dep_delay_minutes'].mean() if not low.empty else np.nan
    else:
        mean_delay_high = mean_delay_low = np.nan

    # Print Report
    print("=== Exploratory Data Analysis Report ===")
    print("Q1. Average departure delay (minutes): {:.2f}".format(avg_delay))
    print("    % of flights departing late: {:.2f}%".format(pct_late))
    print()
    print("Q2. % flights with scheduled ground time <= minimum turn: {:.2f}%".format(pct_at_or_below))
    print("    % flights within +5 minutes above minimum turn: {:.2f}%".format(pct_within5))
    print()
    print("Q3. Average transfer bag ratio (Transfer / (Checked + Transfer)): {:.4f}".format(avg_transfer_ratio))
    print()
    print("Q4. Correlation load factor vs departure delay: {:.4f}".format(corr_delay))
    print("    Correlation load factor vs ground time delta: {:.4f}".format(corr_ground_pressure))
    print()
    print("Q5. Regression delay ~ SSR_count + load_factor:")
    if 'coef_ssr_count' in reg_results:
        print("    Intercept: {:.3f}".format(reg_results['intercept']))
        print("    Coef SSR Count (minutes per SSR controlling for load): {:.3f}".format(reg_results['coef_ssr_count']))
        print("    Coef Load Factor (minutes per unit LF): {:.3f}".format(reg_results['coef_load_factor']))
        print("    R^2: {:.3f} (n={})".format(reg_results['r2'], reg_results['n_obs']))
    else:
        print("    Not enough data for reliable regression (n={}).".format(reg_results.get('rows')))    
    print("    Mean delay (high SSR, top quartile): {:.2f}".format(mean_delay_high))
    print("    Mean delay (no SSR): {:.2f}".format(mean_delay_low))
    print()
    print("Notes:")
    print(" - Ground time delta = scheduled_ground_time_minutes - minimum_turn_minutes (negative/zero indicates pressure).")
    print(" - Transfer ratio excludes flights with zero total bags (NaN).")
    print(" - SSR counts joined via record_locator; flights without remarks treated as 0.")
    print(" - Regression is simple OLS without robust errors; interpret direction & magnitude cautiously.")
    print(" - 'Hot transfer' differentiation not computed (missing connection time data).")


if __name__ == '__main__':
    main()
