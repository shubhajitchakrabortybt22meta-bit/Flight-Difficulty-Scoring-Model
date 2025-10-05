# Local Setup & Run Guide (Flight-Difficulty-Scoring-Model)

This guide provides a clean, step‑by‑step path to install, configure, and run the Flight Difficulty Scoring & Operational Insights framework on a fresh machine. It complements the main `README.md` by focusing strictly on execution workflow and common operational tasks.

---
## 1. Quick Start (TL;DR)
```bash
git clone https://github.com/shubhajitchakrabortybt22meta-bit/Flight-Difficulty-Scoring-Model.git
cd Flight-Difficulty-Scoring-Model
python3 -m venv .venv
source .venv/bin/activate  # macOS / Linux (zsh)
pip install --upgrade pip
pip install -r requirements.txt
mkdir -p resources
# Copy required CSV files into resources/ (see Section 3)
python3 all_analysis.py --config config.yaml
```
Outputs land in `final_result_data/` (CSVs) and `result_overview/` (charts).


---
## 2. Prerequisites
| Need | Notes |
|------|-------|
| Python 3.10+ | Project tested with 3.11 |
| pip | For dependency installation |
| (Optional) GitHub account | To fork & run CI |
| (Optional) Weather data CSV | Enables weather severity feature |

---
## 3. Required Input Data
Place the following source CSVs in the `resources/` folder (exact filenames expected):

| File | Purpose |
|------|---------|
| `Flight_Level_Data.csv` | Core flight schedule & operational metrics |
| `Bag_Level_Data.csv` | Individual bag records (origin / transfer / hot transfer) |
| `PNR_Flight_Level_Data.csv` | Passenger booking & composition (lead time, basic economy, etc.) |
| `PNR_Remark_Level_Data.csv` | SSR & special request remarks |
| `Airports_Data.csv` | Airport metadata (country code for international flag) |
| (Optional) `Weather_Data.csv` | Hourly station weather severity index |

### 3.1 Detailed Schemas (Quick Reference)
See full schema table in the main `README.md` (Input Dataset Schemas). Minimal required columns per file:
- Flights: company_id, flight_number, scheduled_departure_date_local, scheduled_departure_station_code, scheduled_arrival_station_code, scheduled_departure_datetime_local, scheduled_arrival_datetime_local, total_seats, minimum_turn_minutes, scheduled_ground_time_minutes
- Bags: company_id, flight_number, scheduled_departure_date_local, bag_type
- PNR Flight: company_id, flight_number, scheduled_departure_date_local, record_locator, total_pax
- PNR Remarks: record_locator (plus special_service_request for categorization)
- Airports: airport_iata_code, iso_country_code
- Weather (optional): station_code, observation_time, weather_severity_index

Optional columns enhance features (see README for list).

If `resources/` is empty, the pipeline will attempt to fall back to root-level copies (not recommended for new setups).

---
## 4. Create & Activate Virtual Environment
```bash
python3 -m venv .venv
source .venv/bin/activate  # zsh / bash
pip install --upgrade pip
pip install -r requirements.txt
```
To exit later: `deactivate`

---
## 5. Configuration (`config.yaml`)
Key sections:
- `weights:` Feature weight map (any feature name present & weighted >0 is used and normalized).
- `thresholds:` Classification percentiles (`difficult`, `medium`).
- `rolling:` Windows for destination base difficulty & volatility.
- Keyword lists for SSR categorization and widebody seat threshold.

To disable a feature: set its weight to `0` or remove it (removal means it won’t be included).

---
## 6. Run the Unified Pipeline
```bash
python3 all_analysis.py --config config.yaml
```
Optional (show plots interactively):
```bash
python3 all_analysis.py --config config.yaml --show
```
Artifacts:
- `final_result_data/flight_difficulty_scores.csv`
- `final_result_data/destination_consistency.csv`
- `final_result_data/drivers.csv`
- `final_result_data/eda_metrics.csv`
- `final_result_data/difficulty_feature_corr.csv`
- PNG charts in `result_overview/`

---
## 7. Individual Module Runs
Difficulty scoring only:
```bash
python3 difficulty_scoring.py \
  --flights resources/Flight_Level_Data.csv \
  --bags resources/Bag_Level_Data.csv \
  --pnr_f resources/PNR_Flight_Level_Data.csv \
  --pnr_r resources/PNR_Remark_Level_Data.csv \
  --airports resources/Airports_Data.csv \
  --config config.yaml \
  --output final_result_data/flight_difficulty_scores.csv
```
Weight optimization (after initial scoring run):
```bash
python3 optimize_weights.py \
  --scores final_result_data/flight_difficulty_scores.csv \
  --config config.yaml \
  --k 5 \
  --target difficulty
```
After review, copy updated weights from `final_result_data/optimized_weights.json` into `config.yaml` and rerun pipeline.

---
## 8. Optional Weather Enrichment
Add `resources/Weather_Data.csv` with columns:
```
station_code,observation_time,weather_severity_index
```
Times should be ISO or parseable; the system hour-aligns to flight departure. Missing file ⇒ feature auto-falls back to 0.

To test quickly:
```bash
python weather_enrichment_demo.py
```
Adjust weight (importance) in `config.yaml` under `weights.weather_severity_index`.

---
## 9. Testing
Pytest:
```bash
pytest -q
```
Direct quick sanity (no pytest output formatting):
```bash
python3 test_sanity.py
```
Tests validate presence of Priority 2 & 3 features and scoring integrity.

---
## 10. Feature Categories (Operational Lens)
| Category | Examples |
|----------|----------|
| Ground / Turn Stress | `ground_time_pressure`, `actual_turn_deficit`, `turn_deficit_ratio` |
| Passenger / Service Complexity | `ssr_count`, `ssr_mobility_count`, children & stroller features |
| Baggage Flow | `total_bags`, `transfer_ratio`, `hot_transfer_count`, `bag_issue_lead_mean` |
| Demand / Booking Dynamics | `booking_lead_days_mean`, `late_booking_ratio`, `basic_economy_ratio` |
| Structural / Capacity | `load_factor`, `bag_per_seat_ratio`, `widebody_flag` |
| Temporal / Volatility | `load_factor_volatility`, `ground_time_pressure_volatility` |
| Contextual Baseline | `destination_base_difficulty` |
| Interactions | `transfer_ground_interaction`, `load_pressure_interaction` |
| Environmental | `weather_severity_index` (optional) |

---
## 11. Interpreting Outputs
- `difficulty_score`: Composite weighted (0–1) difficulty intensity per day (after per-day scaling). Higher = more difficult.
- `daily_rank`: Dense rank within each departure date (1 = most difficult). Use ranks for daily operational triage.
- `difficulty_class`: Percentile-based triage buckets (tunable in config).
- `destination_consistency.csv`: Stability of difficulty at each station (systemic vs transient).
- `drivers.csv`: Feature lift (scaled) for top challenging destinations.

---
## 12. Weight Optimization Notes
- Uses a heuristic constrained gradient search + cross-validation to maximize correlation (or AUC) vs proxy target.
- Always inspect correlation matrix (`difficulty_feature_corr.csv`) before adopting new weights—avoid redundant overweighting in correlated clusters.

Shortcut to adopt optimized weights:
```bash
jq '.weights' final_result_data/optimized_weights.json > new_weights.json  # if jq installed
# Manually merge into config.yaml under weights:
```

---
## 13. CI Integration
Provided GitHub Actions workflow (`.github/workflows/ci.yml`) automatically:
1. Installs dependencies.
2. Runs pytest.
3. (If scores file exists) performs a light 3-fold weight optimization dry run.

To enable: push repository to GitHub on `main` or `master` branch.

---
## 14. Performance Tips
- Large bag datasets: sampling only affects histogram visuals; scoring always uses full set.
- If memory constrained, you may pre-filter to a date range with a wrapper script before scoring.
- Disable rarely informative features by zeroing their weights to shorten correlation matrix operations.

---
## 15. Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
| KeyError on a column | Header mismatch or missing CSV | Verify columns & no extra header row |
| All scores identical | Zero variance features per day | Inspect raw feature values; check date parsing |
| Missing new feature columns | Weight present but feature engineering prereq column absent | Check input data fields and spelling |
| CI fails on pytest | Incomplete resource data in workflow | Add sample minimal CSVs for CI or skip tests requiring full data |
| Optimizer yields extreme weights | Correlated inputs or noisy proxy target | Prune or group correlated features before re-run |
| ModuleNotFoundError: yaml | Virtual environment not activated | `source .venv/bin/activate` then re-run |
| tz-naive / tz-aware subtraction error | Mixed timezone baggage vs flight datetime | Ensure consistent timezone; code now normalizes to naive |

---
## 16. Common Customizations
| Task | How |
|------|-----|
| Change difficulty thresholds | Edit `thresholds.difficult` / `thresholds.medium` in `config.yaml` |
| Remove a feature | Delete or set its weight to 0 in `config.yaml` |
| Add a new feature | Engineer in `difficulty_scoring.engineer_features`, assign weight in config |
| Add alternative target for optimization | Extend `optimize_weights.py` to new `--target` logic |
| Compare two weight sets | Run optimizer, duplicate config, pipeline twice, diff key metrics |

---
## 17. Clean Re-run
To regenerate all artifacts from scratch:
```bash
rm -f final_result_data/*.csv
python3 all_analysis.py --config config.yaml
```
Add `--show` if you want interactive charts.

---
## 18. Minimal Example (Synthetic)
If you need a smoke-test without full data, create small dummy CSVs (a few rows each) with the required columns. Ensure at least 2 dates so scaling/ranking logic engages.

---
## 19. Security & Privacy Considerations
Keep PNR/customer-sensitive fields excluded or anonymized before sharing externally. The framework assumes de-identified operational aggregates.

---
## 20. Support / Next Steps
Potential follow-on improvements:
- SHAP / permutation importance layer.
- Automated weight adoption pipeline with rollback.
- Weather source ETL integration (METAR ingestion).

---
**You’re ready.** Adjust `config.yaml`, run the pipeline, iterate on weights, and convert insights into operational actions.

---
### Appendix: Environment & Timezone Notes
- Always verify interpreter: `which python` should point inside `.venv`.
- To run without activating: `./.venv/bin/python all_analysis.py --config config.yaml`.
- Timezone-aware timestamps in future extensions should be converted to UTC before merge for consistency.
