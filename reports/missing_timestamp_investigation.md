# DeliveryIQ — Missing Order Timestamp Investigation

**Scope:** the records whose `Time_Orderd` / `order_time` is missing. **Read-only:** no record was filled, deleted or modified. **Pickup time was not used** to reconstruct or describe order time.  
**Sources:** `data/raw/train.csv` (SHA-256 verified unchanged: True), `data/processed/cleaned_data.csv` (`3f2f1ee97163…`).  
**Reproduce:** `.venv\Scripts\python.exe -m src.investigate_missing_timestamps`. Numbers come from `reports/missing_timestamp_metrics.json`. Chart: `figures/11_missing_order_time.png`.

Labels: **[V]** verified from data · **[A]** interpretation · **[R]** recommendation

## Key findings

1. **1,731 records (3.8%)** have no order time. Their **order date is always present** (0 missing dates) [V].
2. They are spread **evenly over time**: every one of the 44 dates has some (daily rate 2.7–5.18%), and each month is about 3.71–3.82% [V].
3. **Delivery time is the same as for other records:** median 25 vs 26 min, mean 26.31 vs 26.29; Mann-Whitney p = 0.78, KS p = 0.84. Share over 30 min: 30.16% vs 29.83% [V]. The missingness is not related to the outcome.
4. **It is one block of incomplete records, not scattered gaps** [V]. In the cleaned data, every one of these records also lacks rider age and rating: 1,640 were already missing in the raw file, and the other 91 are the placeholder rider records blanked in Stage 2. (214 raw age-missing rows lie outside the block.) The block contains **every**:
   - missing-weather row (616/616) and missing-traffic row (601/601);
   - `vehicle_condition = 3` row (520/520) and bicycle (68/68);
   - placeholder rider record (91/91).
5. **No field in the dataset can reliably recover the missing times** (§2). Traffic level narrows the time to a 3–6 hour window at best and can't be checked on these rows.
6. **Side finding with Stage 3 impact:** traffic level is almost a pure function of order hour (97.48% of timed orders follow the bands below) [V].

## 1. Patterns

### By date

| Month | Missing | Rate % |
|---|---|---|
| 2022-02 | 272 | 3.76 |
| 2022-03 | 1,223 | 3.82 |
| 2022-04 | 236 | 3.71 |

Daily rate: 2.7–5.18%; dates with none missing: 0 of 44. Weekday association: chi-square p = 0.54.

### By city

City code (ID prefix, 22 cities): rates 2.88–5.02%, chi-square p = 0.71, so there is no city effect [V].

| City type | Rows | Missing time | Rate % |
|---|---|---|---|
| Urban | 10,136 | 376 | 3.71 |
| Metropolitan | 34,093 | 1,291 | 3.79 |
| <missing> | 1,200 | 56 | 4.67 |
| Semi-Urban | 164 | 8 | 4.88 |

Chi-square p = 0.36 [V].

### By weather

| Weather | Rows | Missing time | Rate % |
|---|---|---|---|
| Windy | 7,422 | 171 | 2.30 |
| Fog | 7,654 | 178 | 2.33 |
| Sunny | 7,284 | 175 | 2.40 |
| Stormy | 7,586 | 184 | 2.43 |
| Cloudy | 7,536 | 197 | 2.61 |
| Sandstorms | 7,495 | 210 | 2.80 |
| <missing> | 616 | 616 | 100.00 |

When weather is recorded, the rate is 2.3–2.8% for every condition. All missing-weather rows fall inside the block, so the high chi-square is co-missingness, not a weather effect [V/A].

### By traffic

| Traffic | Rows | Missing time | Rate % |
|---|---|---|---|
| High | 4,425 | 103 | 2.33 |
| Jam | 14,143 | 343 | 2.43 |
| Medium | 10,947 | 269 | 2.46 |
| Low | 15,477 | 415 | 2.68 |
| <missing> | 601 | 601 | 100.00 |

Same pattern as weather: about 2.3–2.7% for each recorded level, 100% where traffic itself is missing [V].

### By delivery duration (target)

| | Missing order time | Order time present |
|---|---|---|
| count | 1,731 | 43,862 |
| mean | 26.31 | 26.29 |
| std | 9.64 | 9.37 |
| min | 10 | 10 |
| 25% | 19 | 19 |
| 50% | 25 | 26 |
| 75% | 32 | 32 |
| max | 54 | 54 |
| > 30 min (project benchmark) | 30.16% | 29.83% |

Mann-Whitney p = 0.779; Kolmogorov-Smirnov p = 0.841. No detectable difference in delivery time [V].

### Other fields

| Vehicle condition | Rows | Missing time | Rate % |
|---|---|---|---|
| 2 | 15,034 | 392 | 2.61 |
| 0 | 15,009 | 406 | 2.71 |
| 1 | 15,030 | 413 | 2.75 |
| 3 | 520 | 520 | 100.00 |

| Vehicle type | Rows | Missing time | Rate % |
|---|---|---|---|
| motorcycle | 26,435 | 802 | 3.03 |
| scooter | 15,276 | 581 | 3.80 |
| electric_scooter | 3,814 | 280 | 7.34 |
| bicycle | 68 | 68 | 100.00 |

No association with order type (p = 0.98), festival (p = 0.67) or multiple deliveries (p = 0.06).

### Co-missingness

| Field | Missing inside block | Missing overall |
|---|---|---|
| `delivery_person_age_raw` | 1,640 | 1,854 |
| `delivery_person_rating_raw` | 1,640 | 1,908 |
| `weather` | 616 | 616 |
| `traffic_density` | 601 | 601 |
| `multiple_deliveries` | 50 | 993 |
| `festival` | 9 | 228 |
| `city_type` | 56 | 1,200 |

Also inside the block: 131 geo-invalid rows (not over-represented: 7.6% vs 7.98% overall).

**Interpretation [A]:** these look like orders whose rider/dispatch record failed to join. They lost order time, rider age and rating, and in some cases weather and traffic, and carry an unusual vehicle profile (all bicycles, all condition 3). Because delivery times are unaffected, the missingness is unrelated to the outcome, but it is **not random with respect to the other fields**.

## 2. Can the original data recover the missing times?

Pickup time was excluded by rule. Every other candidate in the raw file was tested:

| Candidate | Test (on rows *with* an order time) | Result | Usable? |
|---|---|---|---|
| `Order_Date` | completeness | 0 missing | ✅ The **date** is known for every row |
| Row order in the file | Spearman with order datetime | 0.0016 | ❌ File is not in time order |
| Hex `ID` | Spearman with datetime / with hour | -0.4 / 0.011 | ❌ Weakly tracks the date (already known), says nothing about time of day |
| Same `delivery_person_id` on same date | another order with identical time | 0.8% of 31,226 rows | ❌ Peers almost never share a time |
| `Road_traffic_density` | hour falls in the traffic level's band | 97.48% | ⚠️ Gives only a 3–6 hour **band** (33–55 distinct times per level). Available for only 1,130 of 1,731 block rows; can't be checked on the block |
| `Weatherconditions` | largest single-weather share in any hour | 0.189 | ❌ Weather is unrelated to hour |
| Other columns (rider, vehicle, order type, festival, city, coordinates) | no time content | — | ❌ |

**Traffic bands observed on timed orders** [V]:

| Traffic | Order hours |
|---|---|
| Low | 00, 08, 09, 10, 22, 23 |
| High | 11, 12, 13, 14 |
| Medium | 15, 16, 17, 18 |
| Jam | 19, 20, 21 |

**Conclusion:** the dataset holds **no reliable source for the missing order times**. The order **date** is fully available. Traffic level gives a coarse, unverifiable hour band and **should not be used to fill** `order_time`:
- it would recover a range, not a time;
- it can't be checked on the very rows it would fill;
- it would make the hour feature a copy of the traffic feature.

## 3. Handling in the chronological split [R]

1. **Split on `order_date`, not `order_datetime`.** `order_date` is complete, so every record (including these) can be placed on one side of the split. Sorting by `order_datetime` would push all NaT rows to one end and silently move them all into train or all into test.
2. **Cut at a whole-date boundary** (every order on a date goes to the same side), so no time-of-day ordering is needed within a date.
3. **The block splits proportionally on its own:** with an example 80/20 cut after 2022-03-29, train gets 1,414 (3.81%) and test gets 317 (3.74%). No stratification is needed. (The real cut date is chosen in Stage 4.)

## 4. Handling in model training [R]

1. **Keep the records by default.** Their delivery time matches the rest, and dropping them would remove every bicycle and every `vehicle_condition = 3` order from training.
2. **No filling outside the model pipeline.** Hour-based features get imputation fitted on the **training split only**, or are left as missing for models that handle it natively (e.g. gradient-boosted trees).
3. **Never back-fill from pickup time** (Tier C information) and never from traffic level (circular).
4. **No 'order time missing' indicator by default.** It carries no outcome signal and would mostly flag the incomplete-record block (bicycles and condition 3). `dq_pickup_day_assumed` marks exactly these rows but stays an audit column, never a feature.
5. **Sensitivity analysis:** train with and without the block (alongside the planned Festival comparison) and report test metrics separately for complete records and block records.
6. **Demo implication:** the Streamlit app always takes an order time, so the complete-record test metrics are the ones that represent live use.
7. **Stage 3 note:** traffic level and order hour carry almost the same information (97.48% overlap). Feature selection and importance analysis must treat them as collinear. It also suggests traffic was generated from order time, which bears on whether it is available at order time (Tier A\*) [A].

## 5. Limitations

- The cause of the missingness is unknown; the 'failed join' explanation is an interpretation [A].
- The traffic-to-hour bands come from timed records. Whether they hold for the block can't be tested.
- Statistical tests show no *detectable* difference in delivery time. That isn't proof of no difference.
