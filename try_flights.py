import datetime

from ml.predictor import predict_flight

DAYS_AHEAD = 5     # must stay within about 16 days so the forecast is available
day = (datetime.date.today() + datetime.timedelta(days=DAYS_AHEAD)).isoformat()

flights = [
    {"airline": "AA", "origin": "BOS", "dest": "MIA", "flight_date": day,
     "sched_dep_time": 600, "sched_arr_time": 905,
     "sched_duration_min": 185, "distance_miles": 1258},
    {"airline": "AA", "origin": "BOS", "dest": "MIA", "flight_date": day,
     "sched_dep_time": 1800, "sched_arr_time": 2105,
     "sched_duration_min": 185, "distance_miles": 1258},
    {"airline": "UA", "origin": "DEN", "dest": "SEA", "flight_date": day,
     "sched_dep_time": 1400, "sched_arr_time": 1550,
     "sched_duration_min": 170, "distance_miles": 1024},
]

for f in flights:
    r = predict_flight(f)
    top = r["top_factors"][0]
    print(f"{f['airline']} {f['origin']}->{f['dest']} on {f['flight_date']} "
          f"dep {f['sched_dep_time']:04d}: "
          f"probability {r['delay_probability']:.0%} | risk {r['risk_level']} | "
          f"~{r['expected_delay_minutes']} min | weather {r['weather_source']} | "
          f"biggest factor: {top['factor']} ({top['effect']})")