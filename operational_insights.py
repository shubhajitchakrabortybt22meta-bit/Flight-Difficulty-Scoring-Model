"""Operational Insights on Flight Difficulty

This script consumes the output of difficulty_scoring.py (flight_difficulty_scores.csv)
and produces:
 1. Destination difficulty consistency metrics.
 2. Driver analysis: which engineered features most differentiate Difficult flights for each destination.
 3. Actionable recommendations based on aggregated drivers across top difficult destinations.

Run:
    python operational_insights.py --scores final_result_data/flight_difficulty_scores.csv --top_n 10 \
            --drivers_csv final_result_data/drivers.csv --dest_csv final_result_data/destination_consistency.csv

Outputs printed to console; optionally writes CSV summaries.
"""
from __future__ import annotations
import argparse
import pandas as pd
import numpy as np
from typing import List, Dict

DIFFICULT_CLASS = 'Difficult'

# Features (scaled versions will be used for comparability)
BASE_FEATURES = [
    'ground_time_pressure', 'total_bags', 'transfer_ratio', 'total_pax',
    'load_factor', 'children', 'lap_children', 'stroller_users',
    'ssr_count', 'is_international', 'hot_transfer_count'
]


def load_scores(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Ensure date parsing if present
    if 'scheduled_departure_date_local' in df.columns:
        df['scheduled_departure_date_local'] = pd.to_datetime(df['scheduled_departure_date_local'], errors='coerce')
    return df


def destination_consistency(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if 'scheduled_arrival_station_code' not in df.columns:
        raise KeyError("Column 'scheduled_arrival_station_code' missing from scores file.")

    grp = df.groupby('scheduled_arrival_station_code')
    summary = grp.agg(
        flights=('flight_number', 'count'),
        difficult_flights=('difficulty_class', lambda s: (s == DIFFICULT_CLASS).sum()),
        mean_score=('difficulty_score', 'mean'),
        mean_rank=('daily_rank', 'mean')
    )
    summary['difficult_rate'] = summary['difficult_flights'] / summary['flights']

    # Consistency index: combine proportion difficult and relative mean rank (rank shrunk via inverse)
    # Lower mean_rank is more difficult; transform to difficulty_intensity = 1 / (1 + mean_rank)
    summary['difficulty_intensity'] = 1 / (1 + summary['mean_rank'])
    # Composite (simple average of normalized components)
    summary['consistency_index'] = (summary['difficult_rate'].rank(pct=True) + summary['difficulty_intensity'].rank(pct=True)) / 2

    summary = summary.sort_values(['consistency_index', 'difficult_rate', 'mean_score'], ascending=False)
    return summary.head(top_n)


def driver_analysis(df: pd.DataFrame, difficult_only: pd.DataFrame, feature_list: List[str]) -> pd.DataFrame:
    # Use scaled columns if present, else raw
    scaled_cols = [f"{c}_scaled" for c in feature_list]
    use_scaled = all(c in df.columns for c in scaled_cols)
    cols = scaled_cols if use_scaled else feature_list

    # Aggregate mean per destination for difficult vs overall
    overall_means = df.groupby('scheduled_arrival_station_code')[cols].mean()
    difficult_means = difficult_only.groupby('scheduled_arrival_station_code')[cols].mean()

    # Difference (difficult - overall) highlights which features are elevated when flights are difficult at that destination
    diff = difficult_means.subtract(overall_means, fill_value=0)
    # Flatten for top feature extraction
    records = []
    for dest, row in diff.iterrows():
        top_feats = row.sort_values(ascending=False).head(5)
        for feat, val in top_feats.items():
            records.append({
                'scheduled_arrival_station_code': dest,
                'feature': feat.replace('_scaled', ''),
                'lift_vs_overall': val
            })
    drivers = pd.DataFrame(records).sort_values(['scheduled_arrival_station_code', 'lift_vs_overall'], ascending=[True, False])
    return drivers


def aggregate_recommendations(drivers: pd.DataFrame) -> List[str]:
    # Rank features across all destinations by average lift
    feat_rank = drivers.groupby('feature')['lift_vs_overall'].mean().sort_values(ascending=False)
    recs = []
    for feat, lift in feat_rank.items():
        if lift <= 0:  # Only positive contributors
            continue
        if feat == 'ground_time_pressure':
            recs.append('High ground_time_pressure: consider padding schedule or accelerating turnaround processes at top destinations.')
        elif feat == 'transfer_ratio':
            recs.append('Elevated transfer_ratio: deploy additional transfer bag routing staff or automate transfer bag tracking.')
        elif feat == 'total_bags':
            recs.append('High total_bags volume: pre-stage baggage carts and allocate extra belt/loader resources.')
        elif feat == 'ssr_count':
            recs.append('High ssr_count: coordinate early with customer service for mobility/wheelchair & special assistance staging.')
        elif feat == 'load_factor':
            recs.append('High load_factor: prioritize boarding gate staffing and proactive seating resolution during high-density departures.')
        elif feat == 'children':
            recs.append('More children onboard: ensure family boarding support and stroller tagging efficiency.')
        elif feat == 'lap_children':
            recs.append('Lap children elevated: verify safety briefings and pre-boarding assistance for families.')
        elif feat == 'stroller_users':
            recs.append('Stroller user increase: stage gate-check space and coordinate return logistics on arrival.')
        elif feat == 'is_international':
            recs.append('International flights challenging: align with customs/immigration staffing and multilingual support resources.')
        elif feat == 'hot_transfer_count':
            recs.append('Hot transfer presence: ensure tight connection escorting and prioritize quick bag transfers.')
    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for r in recs:
        if r not in seen:
            deduped.append(r)
            seen.add(r)
    return deduped[:12]


def build_report(scores_path: str, top_n: int, drivers_csv: str | None, dest_csv: str | None):
    df = load_scores(scores_path)
    if df.empty:
        raise ValueError('Scores file is empty.')

    difficult = df[df['difficulty_class'] == DIFFICULT_CLASS]

    # Destination consistency
    top_dest = destination_consistency(df, top_n=top_n)

    # Driver analysis limited to those destinations
    focus_df = df[df['scheduled_arrival_station_code'].isin(top_dest.index)]
    focus_difficult = difficult[difficult['scheduled_arrival_station_code'].isin(top_dest.index)]
    drivers = driver_analysis(focus_df, focus_difficult, BASE_FEATURES)

    # Aggregate recommendations
    recs = aggregate_recommendations(drivers)

    # Optional exports
    if dest_csv:
        top_dest.to_csv(dest_csv, index=True)
    if drivers_csv:
        drivers.to_csv(drivers_csv, index=False)

    print('=== Destination Difficulty Consistency (Top {}) ==='.format(top_n))
    print(top_dest[['flights','difficult_flights','difficult_rate','mean_score','mean_rank','consistency_index']].round(4))
    print('\n=== Top Feature Drivers (Per Destination; top 5 lifts each) ===')
    print(drivers.head(50))
    print('\n=== Consolidated Recommendations ===')
    for i, r in enumerate(recs, 1):
        print(f'{i}. {r}')


def parse_args():
    p = argparse.ArgumentParser(description='Generate operational insights from flight difficulty scores.')
    p.add_argument('--scores', default='final_result_data/flight_difficulty_scores.csv', help='Path to scored flights CSV')
    p.add_argument('--top_n', type=int, default=10, help='Top N destinations for deep dive')
    p.add_argument('--drivers_csv', default=None, help='Optional path to write feature driver table')
    p.add_argument('--dest_csv', default=None, help='Optional path to write destination consistency table')
    return p.parse_args()


def main():
    args = parse_args()
    build_report(args.scores, args.top_n, args.drivers_csv, args.dest_csv)


if __name__ == '__main__':
    main()
