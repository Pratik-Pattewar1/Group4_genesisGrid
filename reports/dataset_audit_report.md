# DeliveryIQ — Stage 1 Dataset Audit Report

**Dataset:** `data/raw/train.csv` · **Audit date:** 2026-10-31 · **Status:** awaiting owner approval
**Source of every number:** `notebooks/01_EDA.ipynb` (executed) → `reports/audit_metrics.json`
**Raw file integrity:** SHA-256 `d293b6fc…cdde1` identical before and after the audit (verified in-notebook).

Each finding is labelled:
- **[V] Verified**: computed directly from the data.
- **[A] Assumption / interpretation**: plausible but not provable from the data alone.
- **[R] Recommendation**: a proposed action, not yet applied.

---

## 1. Dataset overview

| Property | Value |
|---|---|
| Rows × columns | **45,593 × 20** [V] |
| File size | 7,772,212 bytes [V] |
| Order dates | 2022-02-11 → 2022-04-06, 44 distinct dates [V] |
| Dates with no orders | 2022-02-19 → 2022-02-28 and 2022-03-22 [V] |
| Rows per month | Feb 7,242 · Mar 31,989 · Apr 6,362 [V] |
| Target | `Time_taken(min)`, present on 100% of rows [V] |

**Columns** (all loaded as text; raw pandas dtype is `str` for every column) [V]:

| # | Column | Parsed type | Unique | Notes |
|---|---|---|---|---|
| 1 | `ID` | string | 45,593 | Hex row key, unique |
| 2 | `Delivery_person_ID` | string | 1,320 | Pattern `<CITY>RES<nn>DEL<nn>`; **not a person** (§8) |
| 3 | `Delivery_person_Age` | float | 22 | 15, 20–39, 50 |
| 4 | `Delivery_person_Ratings` | float | 28 | 1.0, 2.5–5.0, 6.0 |
| 5–6 | `Restaurant_latitude/longitude` | float | 657 / 518 | Zeros & negative signs (§6) |
| 7–8 | `Delivery_location_latitude/longitude` | float | 4,373 / 4,373 | Near-zero where restaurant is (0,0) |
| 9 | `Order_Date` | date `%d-%m-%Y` | 44 | 0 parse failures |
| 10 | `Time_Orderd` | time `%H:%M:%S` | 176 | 1,731 missing |
| 11 | `Time_Order_picked` | time `%H:%M:%S` | 193 | 0 missing |
| 12 | `Weatherconditions` | category | 6 | Prefixed `"conditions "` |
| 13 | `Road_traffic_density` | category | 4 | Low / Medium / High / Jam |
| 14 | `Vehicle_condition` | int 0–3 | 4 | |
| 15 | `Type_of_order` | category | 4 | Snack / Meal / Drinks / Buffet |
| 16 | `Type_of_vehicle` | category | 4 | motorcycle / scooter / electric_scooter / bicycle |
| 17 | `multiple_deliveries` | int 0–3 | 4 | |
| 18 | `Festival` | category | 2 | Yes / No |
| 19 | `City` | category | 3 | Metropolitian (sic) / Urban / Semi-Urban |
| 20 | `Time_taken(min)` | int | 45 | Stored as `"(min) 24"` |

---

## 2. Data quality findings (summary)

| # | Finding | Rows | Severity |
|---|---|---|---|
| Q1 | Missing values stored as text tokens (`"NaN "`, `"conditions NaN"`), so pandas sees **0 nulls** | 4,225 rows have ≥1 | High |
| Q2 | Trailing whitespace in 11 columns (e.g. every value of `ID`, `City`, `Type_of_order`) | — | Low (mechanical) |
| Q3 | Restaurant coordinates `(0,0)` with delivery coordinates of 0.01–0.14 | 3,640 (7.98%) | High |
| Q4 | Negative restaurant latitude (431) / longitude (162); delivery coordinates positive on all of them | 431 | Medium |
| Q5 | `Delivery_person_ID` maps to many ages and vehicles, so it is not a person | all | High for rider segmentation |
| Q6 | Anomalous rider blocks: age 15 ⇔ rating 1.0 (38 rows); age 50 ⇔ rating 6.0 (53 rows) | 91 | Medium |
| Q7 | Rating 6.0 exceeds an assumed 1–5 scale | 53 | Medium |
| Q8 | Pickup time earlier than order time (midnight crossing) | 831 | Low (parsing) |
| Q9 | `Festival` missing ⇒ target always 10–14 min | 228 | Medium (artifact) |
| Q10 | Synthetic-looking structure in delivery offsets and target (§6, §7) | all | Affects interpretation |

No duplicate rows were found (§4).

---

## 3. Missing-value analysis

Figure: `figures/01_missing_values.png`, `figures/02_missing_cooccurrence.png`

| Column | Missing | % | Target mean if missing / present |
|---|---|---|---|
| `Delivery_person_Ratings` | 1,908 | 4.18 | 26.42 / 26.29 |
| `Delivery_person_Age` | 1,854 | 4.07 | 26.46 / 26.29 |
| `Time_Orderd` | 1,731 | 3.80 | 26.31 / 26.29 |
| `City` | 1,200 | 2.63 | **22.06** / 26.41 |
| `multiple_deliveries` | 993 | 2.18 | **22.77** / 26.37 |
| `Weatherconditions` | 616 | 1.35 | 26.55 / 26.29 |
| `Road_traffic_density` | 601 | 1.32 | 26.54 / 26.29 |
| `Festival` | 228 | 0.50 | **11.17** / 26.37 |

Rows with at least one missing field: **4,225 (9.27%)** [V].

**Structure [V]:**
- Age and Rating are missing together on all 1,854 age-missing rows (null-indicator correlation 0.99). `Time_Orderd` missingness correlates 0.90–0.91 with both.
- Weather and Traffic are missing together (correlation 0.99) and partly co-occur with the rider block (about 0.47–0.59).
- `Festival`, `City` and `multiple_deliveries` go missing independently of the others (correlation ≤ 0.02).

**Interpretation [A]:** the missingness is *not* completely at random. The rider/time block looks like a joined record that failed for some orders. Missing `Festival` rows have targets confined to 10–14 min (std 1.23), so this is almost certainly a dataset-generation artifact, not a real-world signal.

---

## 4. Duplicate analysis

| Check | Result |
|---|---|
| Fully identical rows | **0** [V] |
| Identical rows ignoring `ID` | **0** [V] |
| Duplicate `ID` values | **0** [V] |

No deduplication is needed.

---

## 5. Outlier and invalid-value analysis

Figures: `figures/04_numeric_distributions.png`, `figures/09_outliers.png`

**Descriptive statistics (non-missing values) [V]:**

| Column | Count | Mean | Std | Min | Median | Max |
|---|---|---|---|---|---|---|
| Delivery_person_Age | 43,739 | 29.57 | 5.82 | 15 | 30 | 50 |
| Delivery_person_Ratings | 43,685 | 4.63 | 0.33 | 1.0 | 4.7 | 6.0 |
| Vehicle_condition | 45,593 | 1.02 | 0.84 | 0 | 1 | 3 |
| multiple_deliveries | 44,600 | 0.74 | 0.57 | 0 | 1 | 3 |
| Time_taken(min) | 45,593 | 26.29 | 9.38 | 10 | 26 | 54 |

**Invalid / suspicious values [V]:**
- **Age:** values 20–39 each have about 2,100–2,260 rows (nearly uniform); outside that range, 15 (38 rows) and 50 (53 rows).
- **Rating:** 6.0 on 53 rows. Every one of these rows has age 50. Rating 1.0 on 38 rows, every one with age 15.
- **Vehicle_condition = 3:** 520 rows (others about 15,000 each). It is the only condition for bicycles (68/68).
- **Confound:** `Vehicle_condition = 0` occurs *only* with motorcycles (15,009 rows). Electric scooters only have conditions 2 or 3.

**IQR (1.5×) outliers [V]:**

| Column | Bounds | Outliers |
|---|---|---|
| Age | [10, 50] | 0 |
| Ratings | [3.9, 5.5] | 1,170 below, 53 above |
| Target | [−0.5, 51.5] | 270 above (52–54 min) |
| Audit distance (km) | [−8.9, 27.2] | 0 |
| multiple_deliveries | [−1.5, 2.5] | 361 above (value 3) |

**Interpretation [A]:** target values of 52–54 min fall within the dataset's natural 10–54 range and are not errors. Ratings below 3.9 are statistically unusual but plausible. The age-15/rating-1 and age-50/rating-6 blocks look like injected sentinel or test records.

---

## 6. Geographic data quality

Figure: `figures/07_geographic_distribution.png`

| Check | Rows |
|---|---|
| Restaurant `(0, 0)` | **3,640** [V] |
| Of those, delivery coords both < 1° (0.01–0.14) | 3,640 (exact overlap) [V] |
| Restaurant negative latitude / longitude | 431 / 162 (431 rows total) [V] |
| Those 431 rows: delivery coords positive and inside India | 431 [V] |
| Restaurant inside India bounding box (raw) | 41,522 [V] |
| Restaurant inside India after taking absolute values | 41,953 [V] |
| Delivery inside India | 41,953 [V] |
| Unique restaurant locations (after abs, incl. (0,0)) | 389 [V] |

*India bounding box used: lat 6–37.5°, lon 68–97.5° (a generous assumption [A]).*

**Delivery-location structure [V]:**
- Delivery latitude − restaurant latitude takes only about 14 discrete values (0.01, 0.02 … 0.14°).
- The latitude offset equals the longitude offset on **98.22%** of valid rows, so delivery points lie on a diagonal from the restaurant.
- Haversine distance on the 41,953 valid rows ranges from 1.47 to 20.97 km (median 9.19 km) and is clustered into discrete bands. Spearman correlation with the target is 0.32.

**Interpretation [A]:**
- The negative signs are sign-entry errors: the absolute values fall on known Indian cities, and the delivery coordinates for the same rows are positive.
- The `(0,0)` rows have no recoverable restaurant location. Their delivery coordinates are only the offsets.
- Delivery coordinates are synthetic offsets, not real customer addresses. Clustering will therefore mostly recover the 22 cities, and a fine-grained intra-city "zone" analysis has limited realism.

---

## 7. Target variable assessment

Figure: `figures/03_target_distribution.png`, `figures/06_target_by_segment.png`

- `Time_taken(min)`: integer, 10–54, mean 26.29, median 26, std 9.38, skew 0.49, **no missing values** [V].
- Quantiles: P75 32 · P80 34 · P85 37 · P90 40 · P95 44 [V].
- **Step pattern [V]:** counts are flat within 5-minute blocks (10–14: ~742/value; 15–19: ~1,760; 20–24: ~1,638; 25–29: ~2,014; 30–34: ~1,197; …; 50–54: ~87). [A] This looks like generated data.
- **Definition gap [A]:** the dataset does not state whether the clock starts at order placement or at pickup. This affects how the ETA is presented in the demo.
- **Segment medians [V], descriptive only:**
  - **Traffic:** Low 20 · Medium 27 · High 27 · Jam 31.
  - **Weather:** Sunny 20 vs. Cloudy/Fog 28.
  - **City type:** Urban 22 · Metropolitian 27 · Semi-Urban 49 (n = 164).
  - **Festival:** Yes 45 vs. No 25.
  - **multiple_deliveries:** 0 → 22, 3 → 48.
  - **Type_of_order:** no difference (25–26).

**Lateness / SLA [V]:** the dataset contains **no SLA and no late flag**. Share of orders above candidate cut-offs (information only, not a proposal): >30 min **29.84%** (13,606) · >35 min **17.56%** (8,006) · >40 min **8.85%** (4,037). **A threshold needs owner approval.**

---

## 8. Feature availability assessment

The key question for each column is whether it is known when the prediction is made.

| Column | Available at | Safe for order-time ETA? | Reasoning |
|---|---|---|---|
| `ID` | — | ✗ Drop | Unique row key, no signal |
| `Delivery_person_ID` | Rider assignment | ✗ Raw / ✓ prefix | [V] 1,320 IDs; each has **4–22 different ages** and ≥2 vehicle types, and maps to exactly **one** restaurant location; suffix only DEL01/02/03. [A] It encodes *city + restaurant + slot*, not a person. The city prefix (22 codes) is known at placement and is safe. The full ID risks memorisation. |
| `Delivery_person_Age` | Rider assignment | Tier B | Unknown until a rider is assigned |
| `Delivery_person_Ratings` | Rider assignment | Tier B, **leak risk** | [A] May include feedback on this very delivery; this can't be verified |
| Restaurant lat/lon | Placement | ✓ | Restaurant is known when ordering (after cleaning) |
| Delivery lat/lon | Placement | ✓ | Customer address is entered at checkout (but synthetic, §6) |
| `Order_Date` | Placement | ✓ | Calendar features |
| `Time_Orderd` | Placement | ✓ | Hour of day. [V] No orders 01:00–07:59 |
| `Time_Order_picked` | Pickup | ✗ for placement model | [V] Pickup − order is always exactly 5, 10 or 15 min. Post-placement information: usable only in a "re-estimate at pickup" model |
| `Weatherconditions` | Unknown | Tier A* | [A] Usable at placement only if sourced from a live feed. If recorded during or after the delivery, it leaks |
| `Road_traffic_density` | Unknown | Tier A* | Same as weather. Strongest categorical signal, so its provenance matters |
| `Vehicle_condition` | Rider assignment | Tier B | Attribute of the assigned vehicle; confounded with vehicle type |
| `Type_of_order` | Placement | ✓ | No target signal [V] |
| `Type_of_vehicle` | Rider assignment | Tier B | |
| `multiple_deliveries` | Dispatch / during trip | Tier B, **leak risk** | [A] Batching is often decided at dispatch or even mid-route. Strongest numeric correlate (Spearman 0.34) |
| `Festival` | Placement | ✓ | Calendar-known. Its missingness is an artifact (§3) |
| `City` (type) | Placement | ✓ | Known from the restaurant/customer address |
| `Time_taken(min)` | After delivery | ✗ **Target** | |

**Proposed tiers [R]:**
- **Tier A (placement):** date/time, coordinates/distance, city code and type, order type, festival.
- **A\*:** weather and traffic, conditional on provenance.
- **B (dispatch):** rider and vehicle fields, `multiple_deliveries`.
- **C (pickup):** pickup time.

---

## 9. Data leakage risks

1. **Target-derived:** none present besides the target itself. Never include `Time_taken(min)` or features computed from it. [V]
2. **Post-placement timing:** `Time_Order_picked` is a future event relative to placement. [V]
3. **Ambiguous recording time:** weather, traffic and `multiple_deliveries` may be observed during the delivery. [A]
4. **Rating feedback loop:** ratings may include this order's rating. [A]
5. **Identifier memorisation:** `Delivery_person_ID` (1,320 levels, restaurant-specific) could let models memorise restaurant-level averages. That mainly hurts generalisation to new restaurants. [V/A]
6. **Artifact leakage:** a "Festival is missing" flag would perfectly mark 10–14 min orders, giving a spurious gain. The same applies, more weakly, to City and multiple_deliveries missingness. [V]
7. **Preprocessing leakage:** imputers, encoders and scalers fitted on the full data before splitting. Prevent with pipelines fitted on train only. [R]
8. **Split leakage:** a random split mixes dates and restaurants. A temporal split (or a group split by restaurant location) gives a more honest estimate. [R]

---

## 10. Recommended cleaning actions (Stage 2, *not applied*) [R]

1. Strip whitespace; map `"NaN"` / `"conditions NaN"` to null; drop the `"conditions "` prefix; parse the target to int.
2. Parse dates and times; build `order_datetime` and `pickup_datetime` with midnight roll-over (+1 day when pickup < order).
3. Restaurant coordinates: take absolute values on the 431 negative-sign rows (pending approval). Flag the 3,640 `(0,0)` rows as `geo_valid = False`, set their distance to null, and **keep the rows** for non-geo models. Exclude them from clustering.
4. Flag the age-15/rating-1 and age-50/rating-6 blocks. Recommended handling: set those age and rating values to null (treat as missing). Owner decision.
5. Keep ratings 2.5–3.9 (plausible).
6. Missing values: impute inside the modelling pipeline (median or mode, or an explicit "Unknown" category for weather/traffic/city type). **Do not** add a Festival-missing indicator.
7. Normalise `"Metropolitian"` → `"Metropolitan"` (cosmetic, documented).
8. No row deletion is required. Every exclusion is logged with counts.

## 11. Recommended feature engineering (Stage 3) [R]

- **Distance:** haversine distance (km) and distance band.
- **Time:** order hour, time-of-day band (morning 08–11 / afternoon 12–16 / evening 17–23 / midnight 00), weekday, weekend flag.
- **Location:** city code from the ID prefix (22 levels) and an optional city × city-type interaction.
- **Context:** an ordinal traffic level (Low < Medium < High < Jam); an ordinal weather group, or one-hot weather.
- **Rider (Tier B):** age, rating, vehicle type, vehicle condition, `multiple_deliveries`.
- **Pickup (Tier C only):** order→pickup prep time (5/10/15).
- **Clustering:** K-Means on the cleaned restaurant (and/or delivery) coordinates of valid rows. Expect city-level structure.
- **Rider segmentation:** person-level segmentation is **not supported** (the ID is not a person). At most, segment *order-level rider profiles* (age × rating × vehicle) and label it clearly as such.

## 12. Risks and limitations

- **Synthetic characteristics** [V/A]: uniform ages 20–39, discrete diagonal delivery offsets, 5-minute target blocks, sentinel-like rider blocks. Model metrics may look better than they would on real operations, and business recommendations must be phrased with that caveat.
- **No SLA** in the data. Classification depends on an owner decision.
- **Unknown recording times** for weather, traffic and batching make the leakage assessment partly assumption-based.
- **Short horizon:** about 8 weeks with a 10-day gap. No seasonality beyond weekday and hour can be learned.
- **Small segments:** Semi-Urban (164 rows), bicycle (68), Festival = Yes (896). Their estimates will be unstable.
- **Correlation ≠ causation:** segment differences (e.g. Vehicle_condition 0 → slower) may reflect confounding (condition 0 is motorcycle-only).
- **Ambiguous target start point** (order vs. pickup).

---

### Figures produced
| File | Content |
|---|---|
| `figures/01_missing_values.png` | Missing % per column |
| `figures/02_missing_cooccurrence.png` | Missingness correlation heatmap |
| `figures/03_target_distribution.png` | Target histogram |
| `figures/04_numeric_distributions.png` | Numeric feature distributions |
| `figures/05_categorical_distributions.png` | Categorical distributions |
| `figures/06_target_by_segment.png` | Target by traffic, weather, city, festival, batching, vehicle condition |
| `figures/07_geographic_distribution.png` | Raw vs. valid coordinates, distance |
| `figures/08_time_patterns.png` | Orders by hour and by date |
| `figures/09_outliers.png` | IQR box plots |
| `figures/10_correlation.png` | Spearman correlation matrix |
