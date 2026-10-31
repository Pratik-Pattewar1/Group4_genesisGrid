# DeliveryIQ — Stage 6 App Validation Report

Generated 2026-10-31 by `src/validate_app.py` · **18/18 checks passed**

| # | Check | Result | Detail |
|---|---|---|---|
| 1 | saved model files unchanged (SHA-256 vs pre-Stage-6) | ✅ pass | all 6 identical |
| 2 | parity: 600 orders scored one at a time match the training pipeline exactly | ✅ pass | max /ΔETA/ 0.0, max /Δrisk/ 0.0; 300 with unknown/invalid inputs |
| 3 | app inputs contain no post-placement or leakage fields | ✅ pass | ['restaurant_lat', 'restaurant_lon', 'order_date', 'order_time', 'city_type', 'order_type', 'festival', 'customer_lat', 'customer_lon', 'distance_km'] |
| 4 | model features are the 13 order-placement features only | ✅ pass | ['distance_km', 'restaurant_lat', 'restaurant_lon', 'order_hour', 'order_hour_sin', 'order_hour_cos', 'order_day_of_month', 'order_is_weekend', 'city_type', 'order_type', 'festival', 'order_time_band', 'order_day_of_week |
| 5 | no leakage field exists as an app widget | ✅ pass | none |
| 6 | invalid inputs are rejected with a clear message | ✅ pass | 9/9 rejected |
| 7 | all-unknown order is scored with warnings, not a crash | ✅ pass | ETA 29.5, risk 0.477, 5 warnings |
| 8 | hours with no training data (01–07) produce an extrapolation warning | ✅ pass |  |
| 9 | business sense: evening 16 km order has a longer ETA and higher risk than morning 3 km order | ✅ pass |  |
| 10 | all scenario predictions are finite, ETA within 5–60 min, risk within 0–100% | ✅ pass |  |
| 11 | app loads without exceptions; 4 tabs render | ✅ pass | ['🔮 Predict an order', '📊 Business insights', '📈 Model performance', '🧹 Data quality & limitations'] |
| 12 | UI: 4 example orders show ETA, risk % and level that match the prediction service | ✅ pass | [('Weekday morning, short trip', '18.1 min', '2%', 'Low'), ('Evening peak, long trip', '32.9 min', '61%', 'High'), ('Afternoon, medium trip', '24.0 min', '16%', 'Low'), ('Late night, short trip', '18.0 min', '2%', 'Low') |
| 13 | UI: invalid store coordinates show an error message, not a crash | ✅ pass | ['This order cannot be scored: Store coordinates are outside the area covered by the data (I'] |
| 14 | UI: unknown order time is handled with a warning | ✅ pass |  |
| 15 | UI states '> 30 min' is a project benchmark, not an official SLA | ✅ pass |  |
| 16 | UI states risk scores are uncalibrated | ✅ pass |  |
| 17 | UI shows the three approved recommendations | ✅ pass | ['Priority 1 — Deploy order-placement late-risk alerts with ti', 'Priority 2 — Introduce a two-stage ETA: estimate at placemen', 'Priority 3 — Surge-staff riders in the 19:00–21:59 window an'] |
| 18 | raw, cleaned and engineered datasets unchanged | ✅ pass |  |

## Realistic test orders (prediction service)

| Scenario | ETA (min) | Risk score | Level | Distance km | Warnings | Biggest driver |
|---|---|---|---|---|---|---|
| Bengaluru, weekday 09:15, 3 km snack | 18.0 | 1.8% | Low | 3.0 | 0 | City type (vs Metropolitan) |
| Bengaluru, weekday 20:00, 16 km meal | 32.9 | 61.4% | High | 16.0 | 0 | Distance (vs 9.2 km) |
| Mumbai, Fri 16:30, 8 km drinks | 24.1 | 19.5% | Low | 8.0 | 0 | Time of day (vs 19:00) |
| Jaipur, Sat 23:10, 4 km buffet | 18.5 | 2.0% | Low | 4.0 | 0 | Time of day (vs 19:00) |
| Chennai, festival evening 19:45, 12 km | 45.5 | 99.8% | High | 12.0 | 1 | Festival (vs No) |
| Hyderabad, customer coordinates 21:00 | 27.2 | 31.5% | Elevated | 7.68 | 0 | Distance (vs 9.2 km) |

## Example orders through the Streamlit UI

| Preset | ETA shown | Risk shown | Level | Matches service |
|---|---|---|---|---|
| Weekday morning, short trip | 18.1 min | 2% | Low | ✅ |
| Evening peak, long trip | 32.9 min | 61% | High | ✅ |
| Afternoon, medium trip | 24.0 min | 16% | Low | ✅ |
| Late night, short trip | 18.0 min | 2% | Low | ✅ |

## Rejected invalid inputs

| Case | Message |
|---|---|
| store outside India | Store coordinates are outside the area covered by the data (India). |
| customer outside India | Customer coordinates are outside the area covered by the data (India). |
| only one store coordinate | Enter both store latitude and longitude, or neither. |
| only one customer coordinate | Enter both customer latitude and longitude, or neither. |
| distance 0 km | Delivery distance must be greater than 0 km. |
| distance 30 km (beyond data limit) | Delivery distance 30.0 km exceeds the 25 km limit of the data. |
| customer 60 km away | Delivery distance 59.7 km exceeds the 25 km limit of the data. |
| invalid order type | Invalid Order type: 'Pizza'. Allowed: Snack, Meal, Drinks, Buffet. |
| invalid city type | Invalid City type: 'Rural'. Allowed: Urban, Metropolitan, Semi-Urban. |

Risk scores are uncalibrated model scores; late = > 30 min is a project benchmark, not an official SLA.
