# Model Card: Flight Delay Prediction (Airport Operations Coordination Engine)

**Status:** Evaluation complete for the 2023, 10-airport test scope. Not validated
for other airports or future seasons.
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
- Full dataset: 444,127 rows. The dataset checker passed schema, leakage and
  coverage checks. Destination weather columns each had 0.01% missing values,
  below the 5% weather limit.

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
| Naive mean MAE / RMSE               | 20.29 / 31.11                                                        |
| Mean predicted vs actual minutes    | 13.8 vs 9.7                                                        |

- The selected classifier uses `num_leaves=63`, `min_child_samples=300`, and
  `learning_rate=0.05`; its October validation ROC-AUC was 0.6776.
- The project target was an ROC-AUC of about 0.70 or higher. **The model did not
  reach it (0.659).** It does beat the logistic regression baseline (0.632), but
  ranks flights only moderately better than chance.
- The regressor beats the naive mean, but over-predicts minutes at every level of
  prediction. Treat expected minutes as a rough relative guide, not an exact value.
- Calibration: probabilities follow the diagonal closely overall. The highest-risk
  group is over-predicted (about 0.37 predicted vs 0.32 actual, read from the plot).
  November was over-predicted and December was not. Two months are not enough to
  say more.

## 6. Comparison with other models

| Model                        | ROC-AUC | MAE   | RMSE  |
| ---------------------------- | ------- | ----- | ----- |
| Logistic regression baseline | 0.632   | n/a   | n/a   |
| LightGBM (raw scores)        | 0.659   | 16.34 | 29.96 |
| LightGBM (calibrated)        | 0.659   | n/a   | n/a   |
| Naive mean                   | n/a     | 20.29 | 31.11 |
| Community model              | n/a     | n/a   | n/a   |

Community model: Barvero/flight-delay-predictor (Hugging Face) was reviewed from its
README but not run. It is not comparable fairly: it was trained on about 96K US
domestic flights (year not stated), uses 33 scaled features that the README does
not list, and does not ship its scaler or encoders. Its reported R2 of about 0.89 is high enough
that leakage cannot be ruled out.

## 7. Risk levels
Selected cutoffs on calibrated probability: Low below 0.15, High at 0.30 or
above, Medium between. The analysis function's highest-precision suggestion
(0.05 / 0.34) was not selected because it put about 93% of flights in Medium.

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

Mean absolute SHAP importance was computed with TreeExplainer on a 2,000-flight
sample from the November-December test set. The leading features were reviewed
against the permitted feature list; none is a known post-departure/leaky field.

| Rank | Feature | Mean absolute SHAP |
| ---: | --- | ---: |
| 1 | dep_hour | 0.3485 |
| 2 | month | 0.1905 |
| 3 | o_temperature_2m | 0.1396 |
| 4 | airline | 0.1305 |
| 5 | route_delay_rate | 0.1097 |
| 6 | sched_duration | 0.0968 |
| 7 | day_of_week | 0.0875 |
| 8 | airline_delay_rate | 0.0863 |
| 9 | d_cloud_cover | 0.0848 |
| 10 | o_cloud_cover | 0.0829 |

These are associations used by the model, not causal explanations.

## 9. Known limitations

- US airports only (10 listed). Not valid for other countries or other airports.
- The predictor now allows an unseen airport code to proceed with a warning,
  general historical-rate fallbacks and no weather for that airport. This is a
  best-effort estimate outside the training scope and is not validated for
  operational use. The response includes an additive `warnings` field beyond
  the original PRD output schema; consumers should account for it.
- Training data covers one year (2023).
- Forecast weather is less accurate than the historical weather used for training,
  so live predictions may be weaker than test results.
- Results are moderate because delays contain a lot of randomness.
- The delay rate changes with the season (in the 20,000-row sample: about 25%
  in months 1-9, about 16% in months 11-12). Class weighting inflates raw scores,
  so probabilities are calibrated on October only. They may be too low in busy
  months and too high in calm ones.
- Expected minutes over-predict on the test months (13.8 vs 9.7).
- Expected minutes are clipped to 0-300, matching the training target range.
- Differences between small groups (for example small airlines, holidays) are
  within sampling noise and should not be interpreted.

## 10. Disclaimer

These outputs are estimates for decision support. They are not guarantees and must
not be the only basis for an operational decision.
