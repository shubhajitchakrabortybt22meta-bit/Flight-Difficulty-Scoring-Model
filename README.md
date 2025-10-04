# Flight Difficulty Scoring & Operational Insights Framework

Repository: https://github.com/tusaryan/Flight-Difficulty-Scoring-Model  
Clone:
```bash
git clone https://github.com/tusaryan/Flight-Difficulty-Scoring-Model.git
cd Flight-Difficulty-Scoring-Model
```

> Repository Name: **Flight-Difficulty-Scoring-Model**

## Quick Start & Local Setup
For step-by-step environment creation, required data files, and run commands see: **[SETUP_GUIDE.md](SETUP_GUIDE.md)**.

Fast path (after cloning and changing into the repo directory):  
> **Note:** Ensure Python 3 and pip are installed and available on your PATH.  
Verify with:
### For MacOS/Linux
```bash
python3 --version
pip --version
```
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p resources  # place source CSVs here
python3 all_analysis.py --config config.yaml
```
### For Windows (PowerShell)
```powershell
python --version
pip --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
mkdir resources  # place source CSVs here
python all_analysis.py --config config.yaml
```

If anything fails, consult the troubleshooting table in `SETUP_GUIDE.md`.

## Overview
This project implements a reproducible, data-driven framework to:
1. Engineer multi-source operational + customer service features at the flight level.
2. Compute a daily-reset Flight Difficulty Score, ranking and classifying flights (Difficult / Medium / Easy).
3. Perform Exploratory Data Analysis (EDA) to understand delays, ground time pressure, baggage transfer patterns, and passenger composition impacts.
4. Identify consistently difficult destinations and their top differentiating operational drivers.
5. Produce actionable recommendations to guide staffing, turnaround optimization, passenger service preparation, and baggage handling prioritization.

The framework is modular—each stage can run independently—but `all_analysis.py` orchestrates the entire workflow end-to-end and writes structured outputs.

---
## Problem Statement
Frontline teams must prepare every flight for an on‑time departure, yet complexity varies widely: constrained ground time, heavy transfer baggage, volatile late bookings, service-intensive passenger mixes, and structural factors (aircraft size, international processing) compound operational load. Today, difficulty identification relies on tacit, localized knowledge—non-repeatable, inconsistent, and reactive. A systematic, data-driven lens is required so resource planning (people, equipment, prioritization) can shift from anecdotal to anticipatory.

## Objective
Design a framework (Flight-Difficulty-Scoring-Model) that:
1. Quantifies relative per‑flight operational complexity (daily comparable within each day).
2. Surfaces primary drivers of difficulty (explainability & actionability built in).
3. Highlights destinations exhibiting persistent structural challenge vs transient spikes.
4. Enables iterative extensibility (new features, external data, optimizer loops) without rewiring core pipeline.

## Deliverables (Original → Implementation Mapping)
| Required Deliverable | Implemented Component |
|----------------------|------------------------|
| Average delay & % late | `eda.compute_delays` + metrics in `eda_metrics.csv` |
| Flights near/below min turn | `eda.ground_time_analysis` (at_or_below / within_5) |
| Transfer vs checked bag ratio | `eda.bag_ratio` + histogram chart |
| Passenger load correlation | Load factor computed; correlation & regression outputs |
| SSR vs delay controlling load | Regression (`eda.regression_delay_on_ssr`) |
| Daily difficulty ranking | `difficulty_scoring.per_day_scale_and_score` → `daily_rank` |
| Difficulty classification | Same function (percentile thresholds configurable) |
| Consistently difficult destinations | `operational_insights.destination_consistency` |
| Common drivers | `operational_insights.driver_analysis` & driver lift chart |
| Actionable recommendations | `operational_insights.aggregate_recommendations` |

## High-Level Approach Summary
1. Ingest heterogeneous CSV sources (flight, baggage, passenger, SSR remarks, airports, optional weather).
2. Engineer multi-layer features (structural, temporal, behavioral, contextual, interaction, volatility).
3. Scale features *per day* to avoid cross-day operational scale bias.
4. Apply weighted linear composite → difficulty score (transparent, auditable).
5. Rank & classify flights for daily triage.
6. Compute station consistency & feature lifts to uncover systemic pressure points.
7. Produce human-readable recommendations & correlation diagnostics for governance.
8. Optional optimizer tunes weights via constrained CV while retaining interpretability.

## Why Daily Reset Scaling?
Operational load, flight mix, and schedule density can vary by day (e.g., weekday vs weekend). Scaling per day ensures the top-ranked flights always reflect *relative* intra-day challenge—supporting actionable prioritization for that shift without historical drift dominating distribution.

---

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

## Input Dataset Schemas
Below are the required (R) and optional (O) columns for each input dataset. Optional columns enable additional features if present.

### Flight_Level_Data.csv
| Column | R/O | Purpose |
|--------|-----|---------|
| company_id | R | Part of flight key |
| flight_number | R | Part of flight key |
| scheduled_departure_date_local | R | Part of flight key; daily scaling unit |
| scheduled_departure_station_code | R | Weather & slot congestion context |
| scheduled_arrival_station_code | R | Destination consistency & international flag |
| scheduled_departure_datetime_local | R | Time-of-day slot congestion; block time; weather join |
| scheduled_arrival_datetime_local | R | Block time computation |
| actual_departure_datetime_local | O | (Could extend real-time metrics) |
| actual_arrival_datetime_local | O | (Could extend) |
| total_seats | R | Load factor, bag_per_seat_ratio, size category, widebody flag |
| minimum_turn_minutes | R | Ground pressure & deficits |
| scheduled_ground_time_minutes | R | Ground pressure calculation |
| actual_ground_time_minutes | O | Actual turn deficit, ratio |
| fleet_type | O | Widebody detection (regex), size category |
| carrier | O | Express vs mainline |
| weather_severity_index | O (derived) | Populated by enrichment step if Weather_Data present |

### Bag_Level_Data.csv
| Column | R/O | Purpose |
|--------|-----|---------|
| company_id | R | Join key |
| flight_number | R | Join key |
| scheduled_departure_date_local | R | Join key |
| bag_type | R | Checked vs Transfer vs Hot Transfer counts |
| bag_tag_issue_datetime / bag_issue_datetime / bag_tag_issue_date | O | Bag timing features (lead & late ratios) |

### PNR_Flight_Level_Data.csv
| Column | R/O | Purpose |
|--------|-----|---------|
| company_id | R | Join key |
| flight_number | R | Join key |
| scheduled_departure_date_local | R | Join key |
| record_locator | R | Bridge to remarks for SSR counts |
| pnr_creation_date | O | Booking lead metrics |
| total_pax | R | Load factor, per-pax ratios |
| basic_economy_pax | O | Basic economy ratio |
| is_child | O | Children count |
| is_stroller_user | O | Stroller users |
| lap_child_count | O | Lap children |

### PNR_Remark_Level_Data.csv
| Column | R/O | Purpose |
|--------|-----|---------|
| record_locator | R | Link to PNR flight |
| special_service_request | O | SSR categorization (mobility / special handling) |

### Airports_Data.csv
| Column | R/O | Purpose |
|--------|-----|---------|
| airport_iata_code | R | Join for international flag |
| iso_country_code | R | Domestic vs international determination |

### Weather_Data.csv (Optional)
| Column | R/O | Purpose |
|--------|-----|---------|
| station_code | R | Matched to `scheduled_departure_station_code` |
| observation_time | R | Floored to hour for temporal join |
| weather_severity_index | R | Operational environment stress proxy |

If `Weather_Data.csv` is absent, `weather_severity_index` defaults to 0 and scoring proceeds unchanged.

---
## Weather Enrichment Feature
When `resources/Weather_Data.csv` exists:
1. Each flight’s `scheduled_departure_datetime_local` is floored to the hour.
2. Weather observations are floored similarly; multiple observations per hour are averaged.
3. A left join assigns `weather_severity_index` (0 if no match).
4. The feature can be weighted via `config.yaml` (see `weights.weather_severity_index`).

Recommended value range: 0 (benign) → 1 (severe). If using raw METAR-derived indices, normalize before ingest or extend enrichment to compute severity.

Fallback Behavior: Missing file or missing required columns triggers automatic zero fill (no failure). This preserves reproducibility without external data.

Testing: Run the included demo
```bash
python3 weather_enrichment_demo.py
```
to view sample severity assignment (see `resources/Flight_Level_Data_weather_demo.csv`).

Timezone Note: Supply weather and flight times in consistent local timezone or UTC; the enrichment coerces both to naive and floors to hour (`.dt.floor('h')`).

---


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

Priority 2 Additions | `ssr_mobility_count`, `ssr_special_handling_count`, `bag_issue_lead_mean`, `bag_late_issue_ratio`, `bag_per_seat_ratio`, `destination_base_difficulty`
Priority 3 Additions | `load_factor_volatility`, `ground_time_pressure_volatility`, `slot_congestion_count`, `transfer_ground_interaction`, `load_pressure_interaction`, `weather_severity_index`

All four guidance categories are covered (verified in pipeline coverage output).

## Difficulty Scoring Method
1. Engineer features (see above) across flights.
2. For each departure date, min–max scale each feature independently (prevents cross-day scale drift).
3. Apply weights (default sums to 1; auto-normalized if edited) to scaled feature vector via linear combination.
4. Dense rank descending by score within each day (rank 1 = most difficult).
5. Classify by percentile of rank within day: top 25% → Difficult, next 50% → Medium, rest → Easy (configurable thresholds).

During Priority 1 expansion additional engineered features (booking timing, fare mix, actual turn performance, widebody flag) were incorporated into the weighted vector; weights auto-normalize if you edit `DEFAULT_WEIGHTS`.

---
## Detailed Pipeline Walkthrough
| Step | Module / Function | Purpose | Key Safeguards |
|------|-------------------|---------|---------------|
| 1 | `all_analysis.run_pipeline` | Orchestrates end-to-end flow | Path fallback, directory creation |
| 2 | `weather_enrichment.enrich_with_weather` | Optional weather join | Defaults severity to 0 if absent |
| 3 | `difficulty_scoring.engineer_features` | Core + Priority 1–3 feature creation | Null handling, fallback joins, flexible column detection |
| 4 | `add_additional_features` (in `all_analysis`) | Supplemental exploratory metrics | Protective NaN logic |
| 5 | `eda.*` functions | Descriptive metrics + regression | Sampling only for histogram performance |
| 6 | `difficulty_scoring.per_day_scale_and_score` | Scaling, scoring, ranking, classification | Zero-variance → neutral (0) contribution |
| 7 | `operational_insights.destination_consistency` | Station-level stability indices | Avoids leakage by daily grouping |
| 8 | `operational_insights.driver_analysis` | Feature lift extraction | Uses scaled columns when available |
| 9 | `operational_insights.aggregate_recommendations` | Textual action layer | Maps strongest lifts to playbook actions |
| 10 | Correlation export | Feature multicollinearity audit | Guides pruning / weight consolidation |
| 11 | `optimize_weights` (optional) | Constrained CV weight refinement | Simplex projection (non-negative, sums=1) |

### Core Score Formula
Goal: Combine heterogeneous operational drivers (different scales & units) into a single transparent daily prioritization score.

Formula (flight f on day d):
```
difficulty_score_{f,d} = Σ_{i=1..k} ( w_i * s_{f,d,i} )
```

Daily per‑day scaling of feature i (informal piecewise):
```
if max_i(d) == min_i(d):
  s_{f,d,i} = 0   # zero variance safeguard
else:
  s_{f,d,i} = ( x_{f,d,i} - min_i(d) ) / ( max_i(d) - min_i(d) )
```

Definitions
- k: number of (available ∩ weighted) features used that day
- x_{f,d,i}: raw engineered value of feature i for flight f on day d
- min_i(d), max_i(d): per-day minimum and maximum of feature i across all flights on day d
- s_{f,d,i}: per-day min–max scaled feature value (neutral 0 if zero variance)
- w_i: non‑negative weight from config (auto-normalized so Σ_i w_i = 1 each run)
- difficulty_score_{f,d} in [0,1] because each s_{f,d,i} in [0,1] and weights sum to 1

Edge & handling rules
- Zero variance feature (max_i(d)=min_i(d)) ⇒ contributes 0 for all flights that day
- Missing feature or absent weight key ⇒ excluded from k
- Residual NaNs after engineering ⇒ filled before scaling; any remaining become 0 contribution
- Single-flight day ⇒ all s = 0 ⇒ score = 0 (rank still 1)

Interpretation
Linear, transparent, additive: each feature’s weighted daily relative position contributes independently to the final difficulty score.

Interpretation:
• Linear & additive → each feature contributes independently; no hidden transformations.  
• Weights act as proportional importance multipliers over the *within-day dispersion* of each feature.  
• A feature with no variation that day has zero influence (neutral) regardless of weight (prevents noise inflation).  
• To invert a feature whose higher value actually indicates *less* difficulty, either (a) pre-transform it (e.g., use its negative) before scaling, or (b) assign it a zero weight. Current feature set is oriented so “larger == more difficult pressure”.

Step-by-step Computation (per day):
1. Collect all flights for that calendar day.
2. For each weighted feature i, compute daily min_i and max_i.
3. Scale each flight’s value via min–max (guard zero variance → 0).
4. Multiply each scaled value by its weight and sum.
5. Store the raw linear sum (already in [0,1] because weights sum to 1 and each s in [0,1]).
6. Dense-rank descending by score (ties share rank).

### Mini Example (Illustrative Daily Scoring & Ranking)

Assume 1 day, 5 flights (F1–F5), 3 weighted features A,B,C with weights: w_A=0.5, w_B=0.3, w_C=0.2 (already summing to 1 so no re‑normalization needed).

Raw feature values
| Flight | A | B | C |
|--------|---|---|---|
| F1 | 10 | 5  | 0 |
| F2 | 20 | 5  | 5 |
| F3 | 25 | 10 | 8 |
| F4 | 10 | 10 | 2 |
| F5 | 25 | 5  | 8 |

Per‑day mins: A=10, B=5, C=0  
Per‑day maxes: A=25, B=10, C=8

Min–max scaled features
| Flight | A_s | B_s | C_s |
|--------|-----|-----|-----|
| F1 | (10-10)/15 = 0.00 | (5-5)/5 = 0.00 | (0-0)/8 = 0.00 |
| F2 | (20-10)/15 = 0.67 | 0.00 | (5-0)/8 = 0.625 |
| F3 | (25-10)/15 = 1.00 | (10-5)/5 = 1.00 | (8-0)/8 = 1.00 |
| F4 | 0.00 | 1.00 | (2-0)/8 = 0.25 |
| F5 | 1.00 | 0.00 | 1.00 |

Composite difficulty scores
| Flight | Calculation | Score |
|--------|-------------|-------|
| F1 | 0.5*0.00 + 0.3*0.00 + 0.2*0.00 | 0.000 |
| F2 | 0.5*0.67 + 0.3*0.00 + 0.2*0.625 | 0.460 |
| F3 | 0.5*1.00 + 0.3*1.00 + 0.2*1.00 | 1.000 |
| F4 | 0.5*0.00 + 0.3*1.00 + 0.2*0.25 | 0.350 |
| F5 | 0.5*1.00 + 0.3*0.00 + 0.2*1.00 | 0.700 |

Dense rank (descending score)
| Flight | Score | Dense Rank |
|--------|-------|------------|
| F3 | 1.000 | 1 |
| F5 | 0.700 | 2 |
| F2 | 0.460 | 3 |
| F4 | 0.350 | 4 |
| F1 | 0.000 | 5 |

Percentile rank (p = (rank−1)/(N−1), N=5)
| Flight | Rank | Percentile p |
|--------|------|--------------|
| F3 | 1 | 0.00 |
| F5 | 2 | 0.25 |
| F2 | 3 | 0.50 |
| F4 | 4 | 0.75 |
| F1 | 5 | 1.00 |

Category assignment (thresholds: difficult <0.25, medium <0.75)
| Flight | Percentile | Category |
|--------|------------|----------|
| F3 | 0.00 | Difficult |
| F5 | 0.25 | Medium |
| F2 | 0.50 | Medium |
| F4 | 0.75 | Easy |
| F1 | 1.00 | Easy |

Notes
- Ties would share a dense rank; percentile uses the shared rank.
- Boundary equality (e.g., p == 0.25) falls into the next band because comparison is p < threshold.
- Scores remain within [0,1] given min–max scaling and weights summing to 1.
- Features with zero intra-day variance would contribute 0 uniformly (not present in this example).

This compact example demonstrates the exact daily sequence: raw → scaled → weighted sum → rank → percentile → category.

Edge Cases & Safeguards
• Single Flight Day: All scaled features → 0 (variance=0); score=0; rank=1; percentile rank defined as 0 (hardest).  
• All Flights Identical on a Feature: That feature contributes 0 for all flights that day (neutral).  
• Missing Feature in Data or Weight Dict: Ignored (only intersection of available features & weight keys used).  
• NaNs After Engineering: Imputed/filled prior to scaling; residual NaNs treated as zero contribution.

Why Daily Min–Max (not z-score)?
• Respects bounded [0,1] interpretability & ensures comparability across features linearly.  
• Prevents extreme-value leverage that a standard deviation (z) could introduce on skewed distributions.  
• Centers decision-making on *relative* intra-day spread (operational capacity & composition fluctuate daily).

Pseudocode (simplified):
```
for each day d:
  flights_d = flights[day==d]
  for each feature i in weighted_features:
    mn, mx = min(flights_d[i]), max(flights_d[i])
    if mx == mn: scaled[i] = 0 for all flights_d
    else: scaled[i] = (flights_d[i]-mn)/(mx-mn)
  # Normalize weights (safety)
  W = weights / sum(weights)
  score = Σ_i (W[i] * scaled[i])
  rank = dense_rank(desc(score))
```

### Classification Logic
Objective: Convert continuous difficulty scores into stable categorical bands (Difficult / Medium / Easy) based on *relative* daily ordering.

Definitions (per day with N flights):
- Dense Rank r_f: 1 = most difficult (highest score). Ties share the same r.
- Percentile Rank p_f (0 hardest, 1 easiest):
  - if N == 1: `p_f = 0`
  - else: `p_f = (r_f - 1) / (N - 1)`
  (Ensures hardest → 0, easiest → 1 regardless of gaps from ties.)

Threshold Mapping (from `config.yaml`):
• difficult_threshold (e.g., 0.25)
• medium_threshold (e.g., 0.75)

Category Assignment:
  if p_f < difficult_threshold:  'Difficult'
  elif p_f < medium_threshold:   'Medium'
  else:                          'Easy'

Rationale:
• Maintains approximate proportions even if absolute score distribution narrows or widens.  
• Percentile-based approach is robust to feature scale shifts after weight adjustments.  
• Dense ranks avoid inflation of percentile steps caused by tied scores (ties occupy a single rank value).

Example (N = 8, thresholds 0.25 / 0.75):
Ranks 1..8 → Percentiles: 0.00, 0.14, 0.29, 0.43, 0.57, 0.71, 0.86, 1.00
⇒ Difficult: ranks 1–2 (p < 0.25)
   Medium: ranks 3–6 (0.25 ≤ p < 0.75)
   Easy: ranks 7–8 (p ≥ 0.75)

Edge Considerations:
• Boundary Tie: If multiple flights tie at the cutoff rank, all acquire the same percentile and thus same category (may slightly shift proportion—acceptable tradeoff for fairness).  
• Extreme Threshold Tuning: Setting difficult_threshold too low (<0.05) can create unstable daily counts; recommended floor ~0.15 unless the day has large flight volume.  
• Small N Days: With very few flights (e.g., N=3), distributions become coarse—documented in methodology to contextualize category proportions.

Customizing Thresholds:
1. Adjust `difficult` & `medium` in `config.yaml` (must satisfy 0 < difficult < medium < 1).  
2. Re-run pipeline; categories recompute automatically—no need to modify code.  
3. Validate resulting proportions using `eda_metrics.csv` (optional) to ensure operational alignment.

Sanity Metrics to Monitor:
• Daily Difficult Flight Count variance (should reflect intended proportion, not drift systematically).  
• Stability of top driver lifts when thresholds change (indicates robustness of underlying feature importance).  
• Correlation between score and observed disruption proxies (if/when available) should not degrade markedly after threshold tuning.

### Leakage Avoidance
| Feature | Protection |
|---------|-----------|
| `destination_base_difficulty` | Rolling past window shifted (excludes current day) |
| Volatility metrics | Uses shifted historical occurrences only |
| Weather | Hour-aligned join; no future forecast leakage |

---
## Data Sources & Field Mapping (Condensed)
| Dataset | Critical Columns Leveraged | Uses |
|---------|---------------------------|------|
| Flight Level | dates, seats, min turn, actual/scheduled times, station codes, fleet_type | Turn pressure, load factor, scheduling, size, international join |
| Bag Level | bag_type, issue timestamp | Bag totals, transfer ratios, timing lead metrics |
| PNR Flight | pax counts, booking creation, basic economy, child/stroller flags | Load, booking dynamics, composition |
| PNR Remarks | record_locator, special_service_request text | SSR categorization & counts |
| Airports | airport_iata_code, iso_country_code | International flag |
| Weather (optional) | station_code, observation_time, severity index | Environmental stress proxy |

---
## Feature Engineering Layers (Taxonomy)
1. Structural Capacity: seats, widebody flag, aircraft size category.  
2. Temporal Compression: ground_time_pressure, actual_turn_deficit, turn_deficit_ratio.  
3. Flow Complexity: total_bags, transfer_ratio, hot_transfer_count, bag_issue_lead_mean, bag_late_issue_ratio.  
4. Passenger Composition & Service Load: ssr_count, ssr_mobility_count, ssr_special_handling_count, children, lap_children, stroller_users.  
5. Demand Dynamics: booking_lead_days_mean/median, late_booking_ratio, basic_economy_ratio.  
6. Volatility & Stability: load_factor_volatility, ground_time_pressure_volatility.  
7. Contextual Baseline: destination_base_difficulty.  
8. Interaction Surface: transfer_ground_interaction, load_pressure_interaction.  
9. Environmental: weather_severity_index (optional).  
10. Composite / Exploratory: pressure_index, bag_per_seat_ratio, bag_intensity, ssr_per_pax.

---
## Operational Insights Methodology
| Insight | Metric | Rationale |
|---------|--------|-----------|
| Destination Consistency | Blended percentile of (difficult_rate, difficulty_intensity) | Distinguishes persistent structural issues from episodic spikes |
| Driver Lifts | Δ (Difficult mean scaled feature − Overall mean) | Surfaces over-indexing operational stressors |
| Recommendations | Rule-based mapping from strongest lifts (e.g., ground pressure → turn process acceleration) | Action-oriented translation layer |

---
## Weight Optimization & Governance
1. **Input Matrix**: Use already-scaled feature columns when available to maintain comparability.  
2. **Constraint**: Simplex (non-negative, sum=1) preserves interpretability (each weight = proportional contribution).  
3. **Objective**: Maximize correlation (continuous) or AUC (binary difficult vs not).  
4. **Cross-Validation**: Temporal randomness mitigated via K-fold; future enhancement: time-block CV to respect chronology.  
5. **Review Hooks**: Correlation matrix + driver lifts used to flag redundancy before adopting new weights.  

Adoption path: generate → inspect top weights vs multicollinearity → update `config.yaml` → rerun pipeline → validate driver stability.

---
## End-to-End Workflow (Schematic)
```
     +------------------+
     |   CSV Inputs     |
     | Flights / Bags   |
     | PNR / Remarks    |
     | Airports / Wx    |
     +---------+--------+
          |
          v
     +------------------+
     | Feature Eng.     |  (engineer_features + add_additional_features)
     +---------+--------+
          |
          v
     +------------------+
     | Daily Scaling &  |
     | Difficulty Score |
     +---------+--------+
          |
   +-----------+-----------+
   |                       |
   v                       v
 +-------------+        +---------------+
 | Dest. Cons. |        | Driver Lifts  |
 +------+------+        +-------+-------+
   \                       /
    \                     /
     v                   v
     +------------------+
     | Recommendations  |
     +---------+--------+
          |
          v
     +------------------+
     | Reports & Charts |
     +------------------+
```

---
## Code Module Reference (Key Functions)
| Module | Function | Brief |
|--------|----------|-------|
| `difficulty_scoring` | `engineer_features` | Assemble all raw & derived features with robust fallback logic |
| `difficulty_scoring` | `per_day_scale_and_score` | Day-wise min–max scaling, composite scoring, ranking, classification |
| `all_analysis` | `run_pipeline` | Unified orchestration & artifact generation |
| `operational_insights` | `destination_consistency` | Station difficulty stability index |
| `operational_insights` | `driver_analysis` | Feature lifts comparing Difficult subset |
| `optimize_weights` | `optimize_weights / cross_validate` | Heuristic constrained weight tuning |
| `weather_enrichment` | `enrich_with_weather` | Optional severity index augmentation |
| `test_sanity` | `test_*` | Feature presence & scoring integrity guards |

---
## Validation & Testing Strategy
| Layer | Check | Tool |
|-------|-------|------|
| Data Integrity | Column presence, type coercion | `engineer_features` guards, tests |
| Feature Presence | Priority 2 & 3 columns exist | `test_priority2_features`, `test_priority3_features_present` |
| Scoring Stability | Rank starts at 1, classes valid | `test_scoring_integrity` |
| Multicollinearity | Correlation matrix persisted | `difficulty_feature_corr.csv` |
| Weight Governance | Optimized vs config diff | Manual review + JSON artifact |

---
## Performance & Scaling Considerations
| Concern | Mitigation |
|---------|-----------|
| Large bag dataset | Sampling only for visualization (not for scoring) |
| Memory usage | Per-day scaling eliminates need for full normalized history in memory |
| High feature count | Config weights allow pruning; correlation aids consolidation |
| Future external data | Weather enrichment modular; more can follow same pre-merge pattern |

---
## Extensibility Roadmap
1. **Predictive Layer**: Train supervised disruption risk model (logit / gradient boosting) using difficulty features as inputs.  
2. **Anomaly Detection**: Flag unusual single-day driver surges via robust z-scores on scaled features.  
3. **Temporal Trend Module**: Rolling 7/14/28-day trend deltas for each station’s baseline difficulty.  
4. **Resource Recommendation API**: Map feature intensity to estimated incremental staffing units (calibrated from historical staffing outcome data).  
5. **Explainability Dashboard**: Interactive SHAP/permutation importance complementing current lifts.

---
## How the Solution Addresses the Problem (Justification)
| Problem Aspect | Challenge | Implemented Resolution | Justification |
|----------------|-----------|------------------------|---------------|
| Subjective flight prioritization | Reliance on tribal knowledge | Quantitative per-day scoring & ranking | Provides consistent, comparable triage each operational day |
| Hidden structural complexity | Persistent station issues unnoticed | Destination consistency index | Differentiates chronic vs transient, enabling structural fixes |
| Lack of actionable insight | Raw data ≠ decisions | Driver lifts → recommendation mapping | Direct translation of data to staffing/turn process levers |
| Feature proliferation risk | Hard to maintain & tune | Config-driven weights + correlation export | Governs complexity; allows safe iteration |
| Data leakage / bias | Using future info or same-day aggregated outcomes | Shifted rolling windows & per-day scaling | Ensures fairness and real-time plausibility |
| Non-repeatable weight tuning | Manual intuition only | Constrained optimization script | Objective calibration while retaining interpretability |
| Scalability for new data | Hard-coded pipelines | Modular enrichment (weather, volatility layers) | Pluggable architecture for future feeds |

### Strategic Value
- **Transparency**: Linear weighted model + stored scaled features = full auditability.
- **Actionability**: Recommendations tie directly to over-indexing operational domains.
- **Adaptability**: Config & modular design reduce change friction (weights, thresholds, new features).
- **Preventive Posture**: Early identification of flights likely to strain limited gate / ramp / service resources.

---
## Appendix: Key Formulas & Definitions
| Name | Formula / Definition |
|------|----------------------|
| Ground Time Pressure | `max(minimum_turn_minutes - scheduled_ground_time_minutes, 0)` |
| Actual Turn Deficit | `max(minimum_turn_minutes - actual_ground_time_minutes, 0)` |
| Transfer Ratio | `(Transfer + Hot Transfer Bags) / Total Bags` |
| Load Factor | `total_pax / total_seats` |
| Bag Per Seat Ratio | `total_bags / total_seats` |
| Late Booking Ratio | `bookings with lead_days ≤ threshold / total bookings` |
| Destination Base Difficulty | Rolling past difficult flights / past total flights (window, shifted) |
| Volatility (generic) | Rolling std of shifted metric over last *N* occurrences |
| Interaction (example) | `transfer_ratio * ground_time_pressure` |
| Difficulty Score | `Σ w_i * scaled_feature_i` (daily scaled) |

---

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

## Priority 2 Feature Expansion
Operational resilience & contextual planning features:
- SSR Categorization: `ssr_mobility_count` (wheelchair/mobility), `ssr_special_handling_count` (pets, UMNR, medical) to separate distinct support resource profiles.
- Bag Tag Timing: `bag_issue_lead_mean`, `bag_late_issue_ratio` capture how late baggage loads materialize (late surge risk).
- Structural Ratios: `bag_per_seat_ratio` decouples luggage volume from passenger load factor nuances.
- Contextual Baseline: `destination_base_difficulty` (rolling past-day difficult rate) informs whether a station is inherently complex vs transient anomaly (built without future leakage).

## Priority 3 Feature Expansion
Advanced temporal & interaction modeling:
- Volatility Metrics: `load_factor_volatility`, `ground_time_pressure_volatility` highlight unstable operational patterns (rolling trailing window).
- Slot Congestion: `slot_congestion_count` proxies simultaneous departure pressure (gate/aircraft resource contention).
- Interaction Terms: `transfer_ground_interaction`, `load_pressure_interaction` expose compounding effects between passenger/transfer complexity and ground time scarcity.
- Weather: `weather_severity_index` placeholder (set to 0 unless external weather data integrated).

## Weight Optimization (Planned Toolkit)
Current approach: manually curated weights (config-driven) with correlation matrix review. Future enhancement ideas:
1. Data-Driven Calibration: Optimize weights via constrained regression against downstream disruption labels (when available) or proxy (e.g., departure delay percentile).
2. Regularization: L1/L2 penalties to prevent overweighting collinear clusters (e.g., booking leads, volatility variants).
3. Stability Checks: K-fold temporal block validation ensuring weight robustness across operational periods.
4. Auto-Prune: Iteratively drop features with negligible marginal gain in cross-validated explanatory power.

Config Editing: Adjust `config.yaml` -> `weights:`; all features present in the dataset and weight dict are auto-included and normalized. Thresholds (`difficult`, `medium`) also configurable.

## Running the Unified Pipeline
Ensure virtual environment with dependencies (pandas, numpy, seaborn, matplotlib, scikit-learn if needed for extensions). Place source CSVs into `resources/`.

Run:
```bash
python3 all_analysis.py --config config.yaml
```
Add plots interactively:
```bash
python3 all_analysis.py --config config.yaml --show
```
Outputs land in `final_result_data/` and `result_overview/`.

### Environment Activation Reminder
If you encounter `ModuleNotFoundError: yaml` or missing libs, you likely ran the system interpreter (`python3`) instead of the project venv. Always activate the environment first or invoke with `.venv/bin/python3`:
```bash
source .venv/bin/activate
python3 all_analysis.py --config config.yaml
# OR
.venv/bin/python3 all_analysis.py --config config.yaml
```

### Timezone Handling
Datetime fields from baggage tag issue and scheduled departure are normalized to tz-naive internally. If you later introduce timezone-aware columns, ensure they have a consistent timezone (preferably UTC) or they will be coerced.

## Running Individual Components
Difficulty scoring only:
```bash
python3 difficulty_scoring.py \
  --flights resources/Flight_Level_Data.csv \
  --bags resources/Bag_Level_Data.csv \
  --pnr_f resources/PNR_Flight_Level_Data.csv \
  --pnr_r resources/PNR_Remark_Level_Data.csv \
  --airports resources/Airports_Data.csv \
  --output final_result_data/flight_difficulty_scores.csv
```
Operational insights on existing score file:
```bash
python3 operational_insights.py \
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
