# Flight Difficulty Scoring & Operational Insights Framework

## Overview
This project implements a reproducible, data-driven framework to:
1. Engineer multi-source operational + customer service features at the flight level.
2. Compute a daily-reset Flight Difficulty Score, ranking and classifying flights (Difficult / Medium / Easy).
3. Perform Exploratory Data Analysis (EDA) to understand delays, ground time pressure, baggage transfer patterns, and passenger composition impacts.
4. Identify consistently difficult destinations and their top differentiating operational drivers.
5. Produce actionable recommendations to guide staffing, turnaround optimization, passenger service preparation, and baggage handling prioritization.

The framework is modular—each stage can run independently—but `all_analysis.py` orchestrates the entire workflow end-to-end and writes structured outputs.

## Directory Structure
```
resources/                 # (You move raw CSVs here) Input source datasets
  Airports_Data.csv
  Bag_Level_Data.csv
  Flight_Level_Data.csv
  PNR_Flight_Level_Data.csv
  PNR_Remark_Level_Data.csv

final_result_data/         # Generated structured analytical datasets
  flight_difficulty_scores.csv
  destination_consistency.csv
  drivers.csv
  eda_metrics.csv
  difficulty_feature_corr.csv        # Correlation matrix of scaled difficulty features (for weight tuning)

result_overview/           # Visualization artifacts (PNG charts)
  delay_distribution.png
  ground_pressure_vs_delay.png
  transfer_ratio_hist.png
  ssr_vs_delay.png
  top_destination_consistency.png
  driver_feature_lift.png

*.py                       # Source scripts (scoring, EDA, unified pipeline, insights)
```
If `resources/` is empty, the unified pipeline falls back to root-level CSVs (allows gradual migration).

## Key Scripts
| Script | Purpose |
|--------|---------|
| `difficulty_scoring.py` | Feature engineering + daily difficulty scoring + ranking + classification |
| `eda.py` | Computes EDA metrics & regression (delay ~ SSR + load factor) |
| `operational_insights.py` | Destination consistency, driver lifts, recommendations |
| `all_analysis.py` | Orchestrates: data load → feature engineering → EDA → scoring → insights → visuals |

## Feature Engineering Highlights
Category | Features / Logic
---------|-----------------
Ground Turn Constraints | `ground_time_pressure = max(min_turn - scheduled_ground, 0)`, `turn_buffer_ratio`, `pressure_index`, `actual_turn_deficit`, `turn_deficit_ratio`
Flight Characteristics | `total_bags`, `transfer_ratio`, `hot_transfer_count`, `aircraft_size_category`, `long_haul_indicator`, `is_express`, `widebody_flag`
Passenger Service Needs | `ssr_count`, `children`, `lap_children`, `stroller_users`, `ssr_per_pax`
Operational Load & Density | `total_pax`, `load_factor`, `bag_intensity`
Booking & Demand Timing | `booking_lead_days_mean`, `booking_lead_days_median`, `late_booking_ratio`
Fare Product Mix | `basic_economy_ratio`
International Complexity | `is_international` joined via airports country code
Composite / Exploratory | `pressure_index` (weighted mix of ground_time_pressure, transfer_ratio, ssr_count)

All four guidance categories are covered (verified in pipeline coverage output).

## Difficulty Scoring Method
1. Engineer features (see above) across flights.
2. For each departure date, min–max scale each feature independently (prevents cross-day scale drift).
3. Apply weights (default sums to 1; auto-normalized if edited) to scaled feature vector via linear combination.
4. Dense rank descending by score within each day (rank 1 = most difficult).
5. Classify by percentile of rank within day: top 25% → Difficult, next 50% → Medium, rest → Easy (configurable thresholds).

During Priority 1 expansion additional engineered features (booking timing, fare mix, actual turn performance, widebody flag) were incorporated into the weighted vector; weights auto-normalize if you edit `DEFAULT_WEIGHTS`.

## Destination Consistency Logic
For each arrival station:
- Difficult Rate = Difficult Flights / Total Flights
- Difficulty Intensity = 1 / (1 + mean(daily_rank))  (lower mean rank → higher intensity)
- Consistency Index = average of percentile ranks of (Difficult Rate, Difficulty Intensity)
- Top N destinations (default 10) selected for driver analysis.

## Driver Lift Analysis
For each top destination:
- Compute mean of each (scaled) feature for all flights (baseline) vs Difficult flights.
- Lift = Difficult mean − Overall mean (positive indicates over-indexing when flights become difficult).
- Top 5 lifts per destination extracted; aggregated mean lifts across destinations drive recommendations.

## Recommendations (Dynamic)
Generated from dominant lifted features. Examples:
- High `ground_time_pressure`: adjust scheduling / accelerate turn processes.
- Elevated `transfer_ratio` or `hot_transfer_count`: increase transfer routing staff, expedite connection handling.
- High `ssr_count`: pre-stage mobility & special service resources.
- High `bag_intensity`: pre-stage carts / loaders.
- High `load_factor`: reinforce gate staffing & bin space strategy.
- Increased children / lap children / stroller users: family boarding support & stroller logistics.
- International complexity: customs / multilingual coordination.

## Priority 1 Feature Expansion (New)
Rationale for newly added high-impact operational predictors:
- Booking Lead Metrics (`booking_lead_days_mean`, `booking_lead_days_median`): Shorter lead times indicate demand materializing late, compressing staffing & planning windows.
- Late Booking Ratio (`late_booking_ratio`): Proportion of passengers booking very close to departure; correlates with volatile final loads and potential gate crowding surges.
- Basic Economy Ratio (`basic_economy_ratio`): Higher proportion can drive earlier baggage volume at counters and tighter bin competition (indirect operational friction).
- Actual Turn Deficit & Ratio (`actual_turn_deficit`, `turn_deficit_ratio`): Quantifies realized ground time shortfall relative to the required minimum—direct signal of execution stress.
- Widebody Flag (`widebody_flag`): Larger aircraft amplify coordination complexity (cabin zones, catering, cleaning, baggage volume) even after per-pax normalization.

These features are daily scaled alongside existing drivers and their inter-correlations are exported to `final_result_data/difficulty_feature_corr.csv` to support transparent weight refinement and multicollinearity monitoring.

Interpretation Tips:
- High correlation clusters suggest potential for weight consolidation (e.g., multiple booking lead metrics); keep only the most stable signal in future iterations to reduce noise.
- Unique (low-correlation) features often provide incremental explanatory power—avoid prematurely dropping them.
- Revisit feature lifts post-expansion to ensure recommendations remain anchored to materially differentiating factors rather than redundant proxies.

## Running the Unified Pipeline
Ensure virtual environment with dependencies (pandas, numpy, seaborn, matplotlib, scikit-learn if needed for extensions). Place source CSVs into `resources/`.

Run:
```bash
python all_analysis.py
```
Add plots interactively:
```bash
python all_analysis.py --show
```
Outputs land in `final_result_data/` and `result_overview/`.

## Running Individual Components
Difficulty scoring only:
```bash
python difficulty_scoring.py \
  --flights resources/Flight_Level_Data.csv \
  --bags resources/Bag_Level_Data.csv \
  --pnr_f resources/PNR_Flight_Level_Data.csv \
  --pnr_r resources/PNR_Remark_Level_Data.csv \
  --airports resources/Airports_Data.csv \
  --output final_result_data/flight_difficulty_scores.csv
```
Operational insights on existing score file:
```bash
python operational_insights.py \
  --scores final_result_data/flight_difficulty_scores.csv \
  --drivers_csv final_result_data/drivers.csv \
  --dest_csv final_result_data/destination_consistency.csv
```

## Logical Reasoning Path to Insights
1. Baseline Operations: Quantify delay distribution and ground-time pressure to capture latent operational stress.
2. Load & Composition: Evaluate passenger load factor and service complexity (SSR, children, stroller users) as multipliers of turnaround difficulty.
3. Baggage Dynamics: Distinguish between checked vs transfer vs hot transfer to capture connection pressure.
4. International Complexity: Incorporate regulatory/processing overhead via country code.
5. Normalize per Day: Eliminate cross-day comparability issues (operational scale may vary by day) using daily scaling.
6. Score & Rank: Produce a single difficulty metric to prioritize proactive attention.
7. Consistency & Drivers: Identify where difficulty is systemic (destination consistency index) vs transient.
8. Prescriptive Layer: Translate statistically elevated drivers into targeted operational recommendations.

## Extensibility Ideas
- Introduce weather integration (e.g., METAR-based disruption risk) as additional features.
- Add rolling 7-day or seasonal smoothing for destination difficulty trend monitoring.
- Incorporate anomaly detection for sudden spikes in pressure_index.
- Model calibration using historical disruption labels (if/when available) to optimize weights.

## Data Quality & Edge Handling
- Missing or zero-variance features within a day scale to 0 (neutral contribution).
- Division safeguards: replaces zero denominators with NaN then neutralizes after scaling.
- SSR fallback: if remarks lack flight identifiers, counts allocated via record_locator merges.
- Large bag dataset sampling used only for *EDA histogram* performance; scoring uses full dataset.

## Troubleshooting
Issue | Resolution
------|-----------
No output files | Ensure CSVs exist in `resources/` or root (fallback). Check path exactness.
All scores 0 | Likely all feature variance zero for a given day—confirm date parsing and feature engineering.
KeyError on columns | Confirm dataset headers not shifted (avoid manual header row skipping).
Slow bag ratio | Expected on large dataset; adjust BAG_SAMPLE_CAP in `all_analysis.py` if needed.

## License / Usage
Internal analytical framework example. Add appropriate license if distributing externally.

---
Generated README provides a holistic operational + analytical storyline enabling exploratory reasoning through to action.
