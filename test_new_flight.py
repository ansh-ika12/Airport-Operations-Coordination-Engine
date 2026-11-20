from ml.predictor import predict_flight

# New flight details
flight = {
  "airline": "DL",
  "origin": "ATL",
  "dest": "LTX",
  "flight_date": "2026-11-21",
  "sched_dep_time": 900,
  "sched_arr_time": 1045,
  "sched_duration_min": 105,
  "distance_miles": 404
}

# Generate prediction
result = predict_flight(flight)

# Display results
print("\n===== FLIGHT DELAY PREDICTION =====")

for key, value in result.items():
    print(f"{key}: {value}")