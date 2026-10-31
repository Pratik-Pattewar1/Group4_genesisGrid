# DeliveryIQ — Final Data Audit (cleaned vs. raw)

Independent check of `data/processed/cleaned_data.csv` against `data/raw/train.csv`. Produced by `src/audit_cleaned_data.py`, which does **not** import the cleaning pipeline.

**Cleaned file SHA-256:** `3f2f1ee971632876fe41d09da8fc35a618303b5f945cdd1816c63e6a2047bcfa` · **Raw file SHA-256:** `d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1`

| Check | Result | Evidence |
|---|---|---|
| 1. All 45,593 records present, order_id matches raw ID in the same order | **PASS** | raw 45,593 rows, cleaned 45,593 rows; order_id unique=True; position-by-position match=True |
| 2. 431 coordinate corrections match raw values and the approved validation rules | **PASS** | raw negative rows: 431; pass all 4 rules (recomputed): 431; flag == recomputed set: True; cleaned lat/lon == /raw/: True; *_raw == signed raw: True; positive-sign rows untouched: True |
| 3. 3,640 invalid geographic records have blank cleaned coordinates and flags | **PASS** | raw (0,0) rows: 3640; dq_geo_valid False: 3640; flag set == raw (0,0) set: True; all 4 cleaned coords blank: True; originals preserved in *_raw: True; no blank coords on valid rows: True; not sign-corrected: True |
| 4. 91 suspicious rider records: age & rating blank, originals preserved | **PASS** | raw sentinel rows (age15+r1 / age50+r6): 91 (38 / 53); flag == raw set: True; cleaned age & rating blank: True; originals preserved: True; all other rows' age/rating == raw: True |
| 5. Missing-value counts recomputed from the saved CSV | **PASS** | every column: empty cells == pandas NA, and == raw missing tokens + documented blanks; literal 'NaN'/'nan'/'None' strings left: 0 (table in §Missing values) |
| 6. 1,731 missing order timestamps identified & documented | **PASS** | raw 'NaN' Time_Orderd: 1731; cleaned order_time blank on same rows: True; order_datetime blank on same rows: True; dq_pickup_day_assumed on same rows: True; pickup_time still present: True; documented in cleaning report: True |
| 7. dq_* and *_raw columns excluded from future model features | **REVIEW** | 11 audit columns all tagged 'never a feature' in schema: True; target tagged: True; pickup_* tagged EXCLUDED: True. Documented in CLAUDE.md and the cleaning report, but no feature code exists yet, so exclusion is not yet *enforced*. Stage 3 must assert it in code. |
| 8. No target leakage introduced | **PASS** | target identical to raw: True; all 5 dq flags recomputed from non-target raw columns: True; categorical/int columns reproduce raw sources: True; extra target-like columns: none; max /Spearman/ of any numeric column with target: 0.339. Note: the raw dataset's pre-existing Festival-missing artifact (target 10–14 min) was kept as blanks, not created by cleaning; no indicator column was added. |
| 9. Raw file SHA-256 unchanged | **PASS** | expected = before = after = d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1 |
| 10. data/raw/test.csv exists? | **REVIEW** | test.csv NOT FOUND; data/raw contains: ['train.csv']. No external test set: Stage 4 must hold out a chronological test split from train.csv (already approved), and no test-set results can be reported against an official test file. |
| 11. City spelling-correction count = originally misspelled values only | **PASS** | raw 'Metropolitian': 34,093; raw already-correct 'Metropolitan': 0; reported count: 34,093; count now taken from raw values (step R5), not from cleaned rows |

## Missing values (computed from the saved CSV)

| Column | Empty cells | pandas NA | Expected (raw tokens + documented blanks) | Status |
|---|---|---|---|---|
| `delivery_person_age` | 1,945 | 1,945 | 1,945 | ok |
| `delivery_person_rating` | 1,999 | 1,999 | 1,999 | ok |
| `restaurant_lat` | 3,640 | 3,640 | 3,640 | ok |
| `restaurant_lon` | 3,640 | 3,640 | 3,640 | ok |
| `delivery_lat` | 3,640 | 3,640 | 3,640 | ok |
| `delivery_lon` | 3,640 | 3,640 | 3,640 | ok |
| `order_time` | 1,731 | 1,731 | 1,731 | ok |
| `order_datetime` | 1,731 | 1,731 | — | ok |
| `weather` | 616 | 616 | 616 | ok |
| `traffic_density` | 601 | 601 | 601 | ok |
| `multiple_deliveries` | 993 | 993 | 993 | ok |
| `festival` | 228 | 228 | 228 | ok |
| `city_type` | 1,200 | 1,200 | 1,200 | ok |
| `delivery_person_age_raw` | 1,854 | 1,854 | — | ok |
| `delivery_person_rating_raw` | 1,908 | 1,908 | — | ok |

Columns not listed have no missing values.

**Summary:** 9 PASS · 2 REVIEW · 0 FAIL
