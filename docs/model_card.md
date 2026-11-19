# Model Card: Flight Delay Prediction (Airport Operations Coordination Engine)

**Status:** DRAFT. Items marked PENDING are not yet measured.
**Intended use:** decision support only. A person makes the final decision.
This model does not control aircraft or replace air traffic control.
**All outputs are estimates.**

## 1. What the model predicts

| Output                   | Type           | Meaning                                                        |
| ------------------------ | -------------- | -------------------------------------------------------------- |
| Delay probability        | Classification | Chance a flight arrives more than 15 minutes late (calibrated) |
| Expected delay (minutes) | Regression     | Estimated delay length, clipped to 0-300                       |
| Risk level               | Derived        | Low / Medium / High from probability cutoffs                   |
| Top factors              | Explanation    | Top 5 features pushing the prediction up or down (SHAP)        |

## 2. Data

- US only. BTS On-Time Performance, calendar year 2023; Open-Meteo weather
  (historical for training, forecast at prediction time).
- Airports: ATL, ORD, DFW, DEN, JFK, LAX, SFO, SEA, MIA, BOS
  (flights where both origin and destination are in this list).
- Cancelled and diverted flights are removed.
- Split by time, never randomly: months 1-9 train, month 10 validation,
  months 11-12 test (73,099 test rows).
- Dataset quality check (`04_check_dataset.py`): schema, leakage and missing-value
  checks passed on the 20,000-row sample. **PENDING:** rerun on the full dataset.

## 3. Features

- Categorical: airline, origin, dest.
- Numeric: month, day_of_week, dep_hour, is_holiday, distance, sched_duration,
  origin_dep_count.
- Weather at origin (departure hour) and destination (arrival hour): temperature,
  precipitation, cloud cover, wind speed, wind gusts, weather code.
- History features (computed on training months only): airline, origin,
  destination and route delay rates.

## 4. Excluded (leaky) features

Departure delay, taxi-out and taxi-in times, wheels-off and wheels-on times, actual
arrival time, cancellation codes and the delay-cause columns (carrier, weather,
NAS, late aircraft). These are not known before the flight.

## 5. Results on the test months (November-December 2023)

Measured from `data/test_predictions.parquet`:

| Metric                              | Value                                                              |
| ----------------------------------- | ------------------------------------------------------------------ |
| Actual delay rate                   | 15.3%                                                              |
| Mean calibrated probability         | 16.1%                                                              |
| ROC-AUC (LightGBM classifier)       | 0.659                                                              |
| Average precision (no-skill = 0.15) | 0.262                                                              |
| Delay minutes MAE / RMSE (LightGBM) | 16.34 / 29.96                                                      |
| Naive mean MAE / RMSE               | 20.22 / 31.09 (approximate: mean taken from the 20,000-row sample) |
| Mean predicted vs actual minutes    | 13.8 vs 9.7                                                        |

- The project target was an ROC-AUC of about 0.70 or higher. **The model did not
  reach it (0.659).** It ranks flights better than chance but only moderately.
- The regressor beats the naive mean, but over-predicts minutes at every level of
  prediction. Treat expected minutes as a rough relative guide, not an exact value.
- Calibration: probabilities follow the diagonal closely overall. The highest-risk
  group is over-predicted (about 0.37 predicted vs 0.32 actual, read from the plot).
  November was over-predicted and December was not. Two months are not enough to
  say more.

## 6. Comparison with other models

| Model                        | ROC-AUC                    | MAE     | RMSE    |
| ---------------------------- | -------------------------- | ------- | ------- |
| LightGBM                     | 0.659                      | 16.34   | 29.96   |
| Naive mean (approx.)         | n/a                        | 20.22   | 31.09   |
| Logistic regression baseline | PENDING                    | PENDING | PENDING |
| Community model              | not comparable (see below) |         |         |

Community model: Barvero/flight-delay-predictor (Hugging Face) was reviewed from its
README but not run. It is not comparable fairly: it was trained on about 96K US
domestic flights (year not stated), uses 33 scaled features that the README does
not list, and does not ship its scaler or encoders. Its reported R2 of about 0.89 is high enough
that leakage cannot be ruled out.

## 7. Risk levels
Provisional cutoffs on calibrated probability: Low below 0.15, High at 0.30 or
above, Medium between. **PENDING:** final cutoffs from `ml/risk_cutoffs.py`.
The current automatic suggestion (0.05 / 0.34) was rejected because it put 93.4% of
flights in Medium.

Measured on the test months (November-December 2023, overall delay rate 15.3%):

| Level | Share of flights | Delay rate in group | Share of all delays caught |
|---|---|---|---|
| Low | 57.9% | 10.1% | 38.4% |
| Medium | 37.0% | 20.4% | 49.3% |
| High | 5.1% | 36.7% | 12.3% |

The levels are ordered correctly but separate flights only moderately: most
delayed flights are not rated High, and 38.4% of delayed flights are rated Low.
These cutoffs were evaluated on the same months used to compare them, so the numbers
may be slightly optimistic for new data.

## 8. Explainability

**PENDING:** global SHAP importance on the trained classifier, reviewed for leakage.

## 9. Known limitations

- US airports only (10 listed). Not valid for other countries or other airports.
- Training data covers one year (2023).
- Forecast weather is less accurate than the historical weather used for training,
  so live predictions may be weaker than test results.
- Results are moderate because delays contain a lot of randomness.
- The delay rate changes with the season (in the 20,000-row sample: about 25%
  in months 1-9, about 16% in months 11-12). Class weighting inflates raw scores,
  so probabilities are calibrated on October only. They may be too low in busy
  months and too high in calm ones.
- Expected minutes over-predict on the test months (13.8 vs 9.7).
- Differences between small groups (for example small airlines, holidays) are
  within sampling noise and should not be interpreted.

## 10. Disclaimer

These outputs are estimates for decision support. They are not guarantees and must
not be the only basis for an operational decision.
