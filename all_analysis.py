"""Unified Flight Difficulty & Operational Insights Analysis

This script orchestrates:
 1. Data loading & feature engineering
 2. Exploratory Data Analysis (EDA) metrics & visuals
 3. Daily flight difficulty scoring (ranking + classification)
 4. Additional feature exploration per guidance
 5. Operational insights: destination consistency, drivers, recommendations
 6. Graph artifact generation

Outputs:
  - flight_difficulty_scores.csv
  - destination_consistency.csv
  - drivers.csv
  - Multiple PNG charts (see GRAPH_FILES dict)

Run:
  python all_analysis.py

Optional arguments:
  --show  (if supplied, will display plots interactively)

Dependencies: pandas, numpy, seaborn, matplotlib (and existing local modules)
"""
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import textwrap

# Import local modules (already created earlier)
import difficulty_scoring as ds
import eda
import operational_insights as oi

INPUT_DIR = Path('resources')
RESULT_DATA_DIR = Path('final_result_data')
RESULT_IMG_DIR = Path('result_overview')
RESULT_DATA_DIR.mkdir(exist_ok=True)
RESULT_IMG_DIR.mkdir(exist_ok=True)

DATA_FILES = {
    'flights': INPUT_DIR / 'Flight_Level_Data.csv',
    'bags': INPUT_DIR / 'Bag_Level_Data.csv',
    'pnr_flight': INPUT_DIR / 'PNR_Flight_Level_Data.csv',
    'pnr_remark': INPUT_DIR / 'PNR_Remark_Level_Data.csv',
    'airports': INPUT_DIR / 'Airports_Data.csv'
}

OUTPUT_FILES = {
    'scores': RESULT_DATA_DIR / 'flight_difficulty_scores.csv',
    'dest_consistency': RESULT_DATA_DIR / 'destination_consistency.csv',
    'drivers': RESULT_DATA_DIR / 'drivers.csv',
    'eda_metrics': RESULT_DATA_DIR / 'eda_metrics.csv'
}

GRAPH_FILES = {
    'delay_distribution': RESULT_IMG_DIR / 'delay_distribution.png',
    'ground_vs_delay': RESULT_IMG_DIR / 'ground_pressure_vs_delay.png',
    'transfer_ratio_hist': RESULT_IMG_DIR / 'transfer_ratio_hist.png',
    'ssr_vs_delay': RESULT_IMG_DIR / 'ssr_vs_delay.png',
    'top_destination_consistency': RESULT_IMG_DIR / 'top_destination_consistency.png',
    'driver_feature_lift': RESULT_IMG_DIR / 'driver_feature_lift.png'
}

TOP_N_DEST = 10

# ------------------------------
# Helper / Feature Engineering Extensions
# ------------------------------

def add_additional_features(master: pd.DataFrame) -> pd.DataFrame:
    df = master.copy()
    # Turn buffer ratio
    if {'scheduled_ground_time_minutes', 'minimum_turn_minutes'}.issubset(df.columns):
        df['turn_buffer_ratio'] = np.where(df['minimum_turn_minutes'] > 0,
                                           df['scheduled_ground_time_minutes'] / df['minimum_turn_minutes'],
                                           np.nan)
    # Scheduled block time in minutes (scheduled arrival - scheduled departure)
    if {'scheduled_departure_datetime_local', 'scheduled_arrival_datetime_local'}.issubset(df.columns):
        dep = pd.to_datetime(df['scheduled_departure_datetime_local'], errors='coerce')
        arr = pd.to_datetime(df['scheduled_arrival_datetime_local'], errors='coerce')
        df['scheduled_block_minutes'] = (arr - dep).dt.total_seconds() / 60.0
        df['long_haul_indicator'] = (df['scheduled_block_minutes'] >= 6 * 60).astype(int)
    else:
        df['scheduled_block_minutes'] = np.nan
        df['long_haul_indicator'] = 0
    # Express vs Mainline
    if 'carrier' in df.columns:
        df['is_express'] = (df['carrier'].str.lower() == 'express').astype(int)
    else:
        df['is_express'] = 0
    # Aircraft size category
    if 'total_seats' in df.columns:
        bins = [0, 70, 150, 250, np.inf]
        labels = ['Regional', 'Narrowbody', 'Large Narrowbody', 'Widebody']
        df['aircraft_size_category'] = pd.cut(df['total_seats'], bins=bins, labels=labels, include_lowest=True)
    # Ratios per passenger
    if 'total_pax' in df.columns:
        df['bag_intensity'] = np.where(df['total_pax'] > 0, df.get('total_bags', 0) / df['total_pax'], np.nan)
        df['ssr_per_pax'] = np.where(df['total_pax'] > 0, df.get('ssr_count', 0) / df['total_pax'], np.nan)
    # Pressure index (composite exploratory metric)
    df['pressure_index'] = (
        df.get('ground_time_pressure', 0).fillna(0).astype(float) * 0.6 +
        df.get('transfer_ratio', 0).fillna(0).astype(float) * 0.2 +
        df.get('ssr_count', 0).fillna(0).astype(float) * 0.2
    )
    return df

# ------------------------------
# Visualization Helpers
# ------------------------------

def save_or_show(fig, path, show: bool):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_delay_distribution(flights_delay: pd.DataFrame, show: bool):
    fig, ax = plt.subplots(figsize=(8,5))
    sns.histplot(flights_delay['dep_delay_minutes'].dropna(), bins=50, kde=True, ax=ax, color='steelblue')
    ax.set_title('Departure Delay Distribution')
    ax.set_xlabel('Departure Delay (minutes)')
    ax.set_ylabel('Flight Count')
    save_or_show(fig, GRAPH_FILES['delay_distribution'], show)


def plot_ground_vs_delay(df: pd.DataFrame, show: bool):
    if 'ground_time_delta' not in df or 'dep_delay_minutes' not in df:
        return
    fig, ax = plt.subplots(figsize=(7,5))
    sns.scatterplot(data=df.sample(min(len(df), 3000), random_state=42),
                    x='ground_time_delta', y='dep_delay_minutes', alpha=0.4, ax=ax)
    ax.set_title('Ground Time Delta vs Departure Delay')
    ax.set_xlabel('Ground Time Delta (scheduled - minimum)')
    ax.set_ylabel('Departure Delay (min)')
    save_or_show(fig, GRAPH_FILES['ground_vs_delay'], show)


def plot_transfer_ratio(bag_stats: pd.DataFrame, show: bool):
    if 'transfer_ratio' not in bag_stats:
        return
    fig, ax = plt.subplots(figsize=(7,5))
    sns.histplot(bag_stats['transfer_ratio'].dropna(), bins=40, ax=ax, color='darkorange')
    ax.set_title('Transfer Bag Ratio Distribution')
    ax.set_xlabel('Transfer Ratio')
    save_or_show(fig, GRAPH_FILES['transfer_ratio_hist'], show)


def plot_ssr_vs_delay(load_df: pd.DataFrame, ssr_df: pd.DataFrame, show: bool):
    merged = load_df.merge(ssr_df, on=eda.FLIGHT_KEY, how='left').fillna({'ssr_count':0})
    if 'dep_delay_minutes' not in merged:
        return
    fig, ax = plt.subplots(figsize=(7,5))
    sample = merged.sample(min(len(merged), 3000), random_state=42)
    sns.scatterplot(data=sample, x='ssr_count', y='dep_delay_minutes', alpha=0.4, ax=ax)
    # Simple trend line
    if sample['ssr_count'].nunique() > 1:
        coef = np.polyfit(sample['ssr_count'], sample['dep_delay_minutes'], 1)
        x_line = np.linspace(sample['ssr_count'].min(), sample['ssr_count'].max(), 100)
        y_line = coef[0]*x_line + coef[1]
        ax.plot(x_line, y_line, color='red', linewidth=2, label=f'Trend: y={coef[0]:.2f}x+{coef[1]:.1f}')
        ax.legend()
    ax.set_title('SSR Count vs Departure Delay')
    ax.set_xlabel('SSR Count')
    ax.set_ylabel('Departure Delay (min)')
    save_or_show(fig, GRAPH_FILES['ssr_vs_delay'], show)


def plot_destination_consistency(dest_df: pd.DataFrame, show: bool):
    fig, ax = plt.subplots(figsize=(10,5))
    display_df = dest_df.reset_index().copy()
    sns.barplot(data=display_df, x='scheduled_arrival_station_code', y='consistency_index', ax=ax, palette='viridis')
    ax.set_title('Top Destination Consistency Index')
    ax.set_xlabel('Destination')
    ax.set_ylabel('Consistency Index')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    save_or_show(fig, GRAPH_FILES['top_destination_consistency'], show)


def plot_driver_feature_lift(drivers: pd.DataFrame, show: bool):
    avg_lift = drivers.groupby('feature')['lift_vs_overall'].mean().sort_values(ascending=False).reset_index()
    fig, ax = plt.subplots(figsize=(9,5))
    sns.barplot(data=avg_lift.head(12), x='feature', y='lift_vs_overall', ax=ax, palette='magma')
    ax.set_title('Average Feature Lift (Difficult vs Overall)')
    ax.set_xlabel('Feature')
    ax.set_ylabel('Lift')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    save_or_show(fig, GRAPH_FILES['driver_feature_lift'], show)

# ------------------------------
# Main Orchestration
# ------------------------------

def _resolve_input_path(p: Path) -> Path:
    """Return existing path; if not found in resources, fall back to root file name."""
    if p.exists():
        return p
    fallback = Path(p.name)
    if fallback.exists():
        print(f"[INFO] Fallback to root-level file for {p.name} (resources/ missing)" )
        return fallback
    raise FileNotFoundError(f"Input file not found in resources or root: {p}")


def run_pipeline(show_plots: bool = False):
    print("=== 1. Loading Data ===")
    flights = pd.read_csv(_resolve_input_path(DATA_FILES['flights']))
    bags = pd.read_csv(_resolve_input_path(DATA_FILES['bags']))
    pnr_f = pd.read_csv(_resolve_input_path(DATA_FILES['pnr_flight']))
    pnr_r = pd.read_csv(_resolve_input_path(DATA_FILES['pnr_remark']))
    airports = pd.read_csv(_resolve_input_path(DATA_FILES['airports']))

    # Normalize headers
    for df in [flights, bags, pnr_f, pnr_r, airports]:
        df.columns = df.columns.str.strip()

    # Parse datetime columns for flights
    for c in ['scheduled_departure_datetime_local','scheduled_arrival_datetime_local',
              'actual_departure_datetime_local','actual_arrival_datetime_local']:
        if c in flights.columns:
            flights[c] = pd.to_datetime(flights[c], errors='coerce')

    # Harmonize date columns
    for df in [flights, bags, pnr_f]:
        if 'scheduled_departure_date_local' in df.columns:
            df['scheduled_departure_date_local'] = pd.to_datetime(df['scheduled_departure_date_local'], errors='coerce').dt.date

    print("Flights rows:", len(flights), "Bags:", len(bags), "PNR flights:", len(pnr_f), "Remarks:", len(pnr_r))

    print("\n=== 2. Feature Engineering (Core) ===")
    master = ds.engineer_features(flights, bags, pnr_f, pnr_r, airports)
    master = add_additional_features(master)
    print("Master engineered columns (sample):", master.columns[:25].tolist())
    print(master.head())

    print("\n=== 3. Exploratory Data Analysis (Metrics) ===")
    flights_delay = eda.compute_delays(flights)
    ground_df = eda.ground_time_analysis(flights_delay)
    # Optimize bag ratio computation: sample if dataset extremely large to avoid slow groupby in overview
    BAG_SAMPLE_CAP = 250000
    bags_for_ratio = bags
    if len(bags_for_ratio) > BAG_SAMPLE_CAP:
        bags_for_ratio = bags_for_ratio.sample(BAG_SAMPLE_CAP, random_state=42)
        print(f"Bag dataset sampled to {BAG_SAMPLE_CAP} rows for transfer ratio overview (original {len(bags)}).")
    bag_stats = eda.bag_ratio(bags_for_ratio)
    load_df = eda.passenger_loads(ground_df, pnr_f)
    ssr_df = eda.ssr_counts(pnr_f, pnr_r)
    reg_results = eda.regression_delay_on_ssr(load_df, ssr_df)

    avg_delay = flights_delay['dep_delay_minutes'].mean()
    pct_late = (flights_delay['dep_delay_minutes'] > 0).mean() * 100
    pct_at_or_below = ground_df.get('at_or_below_min_turn', pd.Series(dtype=int)).mean() * 100
    pct_within5 = ground_df.get('within_5_min_above_min', pd.Series(dtype=int)).mean() * 100
    avg_transfer_ratio = bag_stats['transfer_ratio'].mean()
    corr_delay = load_df[['dep_delay_minutes','load_factor']].corr().iloc[0,1]
    corr_ground = load_df[['ground_time_delta','load_factor']].corr().iloc[0,1] if 'ground_time_delta' in load_df else np.nan

    eda_summary = pd.DataFrame([
        {"metric":"avg_departure_delay_min","value":avg_delay},
        {"metric":"pct_flights_late","value":pct_late},
        {"metric":"pct_ground_time_at_or_below_min","value":pct_at_or_below},
        {"metric":"pct_ground_time_within_5_min","value":pct_within5},
        {"metric":"avg_transfer_ratio","value":avg_transfer_ratio},
        {"metric":"corr_loadfactor_delay","value":corr_delay},
        {"metric":"corr_loadfactor_ground_delta","value":corr_ground},
    ])
    print(eda_summary)
    print("Regression SSR + Load Factor -> Delay:", reg_results)

    print("\n=== 4. Difficulty Scoring (Daily Reset) ===")
    scored = ds.per_day_scale_and_score(master)
    scored[['company_id','flight_number','scheduled_departure_date_local','difficulty_score','daily_rank','difficulty_class']].head().to_string()
    scored.to_csv(OUTPUT_FILES['scores'], index=False)
    print("Scoring completed. Rows:", len(scored))
    print(scored[['difficulty_class']].value_counts().rename('count'))

    # Additional correlation: load_factor vs difficulty_score (post-score)
    if 'difficulty_score' in scored and 'load_factor' in scored:
        lf_diff_corr = scored[['difficulty_score','load_factor']].corr().iloc[0,1]
        print(f"Load Factor vs Difficulty Score correlation: {lf_diff_corr:.4f}")
    else:
        lf_diff_corr = np.nan

    # Feature correlation matrix (scaled features only among scoring set)
    scoring_feats = [c for c in ds.FEATURE_ORDER if f"{c}_scaled" in scored.columns]
    if scoring_feats:
        corr_matrix = scored[[f"{c}_scaled" for c in scoring_feats]].corr()
        corr_path = RESULT_DATA_DIR / 'difficulty_feature_corr.csv'
        corr_matrix.to_csv(corr_path)
        print(f"Saved difficulty feature correlation matrix to {corr_path}")

    print("\n=== 5. Operational Insights ===")
    dest_consistency = oi.destination_consistency(scored, top_n=TOP_N_DEST)
    dest_consistency.to_csv(OUTPUT_FILES['dest_consistency'])
    print(dest_consistency)

    difficult_only = scored[scored['difficulty_class']=='Difficult']
    drivers = oi.driver_analysis(scored[scored['scheduled_arrival_station_code'].isin(dest_consistency.index)],
                                 difficult_only[difficult_only['scheduled_arrival_station_code'].isin(dest_consistency.index)],
                                 oi.BASE_FEATURES)
    drivers.to_csv(OUTPUT_FILES['drivers'], index=False)
    print("Top driver lifts (sample):")
    print(drivers.head(20))

    recs = oi.aggregate_recommendations(drivers)
    print("\nRecommendations:")
    for i,r in enumerate(recs,1):
        print(f" {i}. {r}")

    print("\n=== 6. Visualization Artifacts ===")
    plot_delay_distribution(flights_delay, show_plots)
    plot_ground_vs_delay(ground_df, show_plots)
    plot_transfer_ratio(bag_stats, show_plots)
    plot_ssr_vs_delay(load_df, ssr_df, show_plots)
    plot_destination_consistency(dest_consistency, show_plots)
    plot_driver_feature_lift(drivers, show_plots)
    print("Saved charts:")
    for k,v in GRAPH_FILES.items():
        print(f" - {k}: {v}")

    print("\n=== 7. Feature Exploration Guidance Coverage ===")
    coverage = {
        'ground_time_constraints': bool({'ground_time_pressure','turn_buffer_ratio'}.issubset(scored.columns)),
        'flight_specific_characteristics': bool({'total_bags','load_factor','aircraft_size_category','is_express'}.intersection(scored.columns)),
        'passenger_service_needs': bool({'ssr_count','children','stroller_users','lap_children'}.issubset(scored.columns)),
        'additional_exploration': bool({'bag_intensity','ssr_per_pax','pressure_index','scheduled_block_minutes'}.issubset(scored.columns)),
    }
    coverage_df = pd.DataFrame([
        {'category':k,'implemented':v} for k,v in coverage.items()
    ])
    print(coverage_df)
    eda_summary.to_csv(OUTPUT_FILES['eda_metrics'], index=False)

    missing = [k for k,v in coverage.items() if not v]
    if missing:
        print("\nNOTE: Some guidance categories incomplete:", missing)
    else:
        print("All guidance categories covered with engineered features.")

    print("\n=== 8. Completed Unified Analysis ===")
    print("Artifacts generated:")
    for name, path in {**OUTPUT_FILES, **GRAPH_FILES}.items():
        print(f" * {name}: {path}")

    return {
        'master': master,
        'scored': scored,
        'dest_consistency': dest_consistency,
        'drivers': drivers,
        'eda_summary': eda_summary,
        'recommendations': recs,
        'coverage': coverage_df
    }

# ------------------------------
# CLI Entry
# ------------------------------

def parse_args():
    p = argparse.ArgumentParser(description='Run full unified analysis pipeline.')
    p.add_argument('--show', action='store_true', help='Display plots interactively')
    return p.parse_args()


def main():
    args = parse_args()
    run_pipeline(show_plots=args.show)

if __name__ == '__main__':
    main()
