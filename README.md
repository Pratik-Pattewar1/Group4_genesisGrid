# DeliveryIQ — AI-Powered Quick Commerce Delivery Intelligence System

DeliveryIQ is our submission for the **KLU + PATH3 ML-01 Quick-Commerce Delivery Intelligence** hackathon.
From order-level delivery data it predicts delivery time (ETA) and late-order risk at the moment an order
is placed, analyses delivery zones, and turns the evidence into three prioritised management actions,
with an interactive Streamlit demo.

> **Status:** final submission. All six stages are complete and the Streamlit demo runs and is validated **locally**. The app has **not** been deployed online.
> **Final reports:** [`reports/business_insights_report.pdf`](reports/business_insights_report.pdf) (Business Insights Report / Business Solution Document) ·
> [`reports/business_report.pdf`](reports/business_report.pdf) (Business Report).
> **Run the demo:** `.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py` → http://localhost:8501

## Team
**Team Name:** Genesis Grid  
**Group Number:** Group 4

| S. No. | Team Member |
|---|---|
| 1 | Manda Lohitha |
| 2 | Pratik Jankiram Pattewar |
| 3 | K Bonishira Akshaya |
| 4 | Yateesh Chandra Maram |
| 5 | Sanapalli Vikas |

## Results at a glance (held-out test period 2022-03-30 → 04-06, order-placement features only)
| Task | Result | Baseline |
|---|---|---|
| ETA regression (gradient boosting) | MAE **6.19 min**, RMSE 7.73, R² 0.310 | median: MAE 7.47 (model is 17.1% better) |
| Late risk > 30 min (*project benchmark, not an official SLA*) | ROC-AUC **0.791**, PR-AUC 0.608, recall 0.813, precision 0.467 | PR-AUC 0.287 (base rate) |
| Geographic K-Means | k = 22, silhouette 0.948 — **zones = the 22 cities**; no dark-store conclusions | — |
| Rider segmentation | **Not feasible**: rider IDs are restaurant/slot codes, not people | — |

**Three recommendations:**
1. Order-placement late-risk alerts with tiered interventions.
2. A two-stage ETA, refreshed at rider assignment (comparison evidence: MAE 3.67 min without rider rating; not production-ready).
3. Surge-staff riders 19:00–21:59 (those hours are about 50% late vs a 29.8% network rate) and plan promises around peaks.

Key limitations: food-delivery (not grocery quick-commerce) data with synthetic patterns, an 8-day test window, and uncalibrated probabilities.

## Submission documents (per `Hackathon_Challenge_Guide.pdf` §13)
| Deliverable | File |
|---|---|
| **Final Business Insights Report** (Business Solution Document, ≥ 5 pages; 23 pages) | [`reports/business_insights_report.pdf`](reports/business_insights_report.pdf) |
| **Final Business Report** (4–6 pages; 6 pages) | [`reports/business_report.pdf`](reports/business_report.pdf) |
| Editable sources of the two PDFs (single source of truth) | [`reports/business_insights_report.md`](reports/business_insights_report.md), [`reports/business_report.md`](reports/business_report.md) |
| Three prioritised recommendations | [`reports/business_recommendations.md`](reports/business_recommendations.md) |
| Model comparison and selection table | Solution Document §5.1 |
| Requirement-by-requirement compliance check | [`reports/challenge_compliance.md`](reports/challenge_compliance.md) |
| Streamlit demo (ETA + delay-risk %) | [`app/streamlit_app.py`](app/streamlit_app.py), validated by [`reports/app_validation_report.md`](reports/app_validation_report.md) |

## Assumptions
- **Late = delivery time > 30 minutes.** This is a project-defined benchmark: neither the data nor the challenge guide gives an official SLA. Sensitivity is reported at 35 and 40 minutes.
- **Restaurants stand in for dark stores**, as the guide frames it. The data is food delivery, not grocery quick commerce.
- **Restaurant coordinates with a negative sign are data-entry errors** and were corrected only when 4 geographic checks agreed. (0, 0) locations are invalid and are never estimated.
- **Rider ratings run from 1 to 5** (assumed; not documented). Age 15 / rating 1.0 and age 50 / rating 6.0 are placeholder records.
- **Weather and traffic have unknown recording times,** so they are excluded from order-placement models.
- **`Delivery_person_ID` is a restaurant/slot code, not a person,** so no rider-level conclusions are drawn.
- **Distance is the straight-line Haversine distance,** not road distance.

## Project overview
DeliveryIQ turns historical delivery records into operational intelligence for quick-commerce and
food delivery. It predicts *when* an order will arrive, *how likely* it is to be late, *where*
delivery demand clusters, and *what* the business should change.

## Problem statement
Customers expect fast, accurate ETAs, and late deliveries drive cancellations, refunds and churn.
Operations teams need to know which orders are at risk **before** they become late, and which
conditions (traffic, weather, distance, batching, rider and vehicle factors) drive delays.

## Objectives
1. **ETA regression:** predict delivery time in minutes.
2. **Late-risk classification:** score the risk that an order takes more than 30 minutes (a project-defined benchmark, not an official SLA).
3. **Delivery zones:** discover geographic clusters with K-Means.
4. **Business output:** exactly three prioritised, evidence-backed recommendations.
5. **Demo:** a Streamlit app that returns the predicted ETA and delay probability for an order.

## Dataset
- Location: `data/raw/train.csv` (**read-only, never modified**)
- 45,593 orders × 20 columns; order dates 2022-02-11 → 2022-04-06 across 22 Indian city codes.
- Target: `Time_taken(min)` (10–54 min, median 26).
- Key quality issues (see the audit report): missing values stored as text `"NaN"`, 3,640 rows with
  `(0,0)` restaurant coordinates, 431 rows with sign-flipped coordinates, and a
  `Delivery_person_ID` that does not identify a single person.
- **No official SLA exists in the data.** The project uses a *project-defined* late benchmark of
  **> 30 min**, with sensitivity checks at 35 and 40 min.
- Cleaned data: `data/processed/cleaned_data.csv` (all 45,593 rows kept; issues flagged in `dq_*` columns).
  See [`reports/data_cleaning_report.md`](reports/data_cleaning_report.md).

| Data-quality issue | Count | Treatment |
|---|---|---|
| Rows with missing values (stored as text `NaN`) | 4,225 | Converted to blanks; filled only inside train-fitted model pipelines |
| Duplicate rows | 0 | — |
| Restaurant location recorded as (0, 0) | 3,640 | Flagged invalid; never estimated; rows kept |
| Sign-flipped restaurant coordinates | 431 | Corrected after 4 geographic checks; originals kept |
| Placeholder rider records (age 15 / rating 1, age 50 / rating 6) | 91 | Age and rating blanked; originals kept |
| Missing order time | 1,731 | Kept; never rebuilt from pickup time |
| Rows deleted | 0 | All 45,593 rows retained |

Engineered features: `data/processed/feature_engineered_data.csv`
([`reports/feature_dictionary.md`](reports/feature_dictionary.md)).

Full audit: [`reports/dataset_audit_report.md`](reports/dataset_audit_report.md)

## Project architecture
```
Project1/
├── README.md
├── requirements.txt                 # pinned dependencies
├── .gitignore
├── data/
│   ├── raw/train.csv                # immutable source (SHA-256 checked by every stage)
│   └── processed/                   # cleaned_data.csv, cleaned_data_schema.json, feature_engineered_data.csv
├── notebooks/01_EDA.ipynb           # Stage 1 audit (executed)
├── src/
│   ├── data_cleaning.py             # Stage 2: cleaning + validation
│   ├── audit_cleaned_data.py        # Stage 2: independent audit
│   ├── investigate_missing_timestamps.py
│   ├── feature_engineering.py       # Stage 3: features, leakage guard, chronological split
│   ├── regression.py, classification.py, clustering.py, evaluation.py
│   ├── modelling.py, modelling_report.py            # Stage 4
│   ├── business_insights.py, insights_report.py, challenge_gaps.py   # Stage 5
│   ├── predict.py                   # Stage 6: order-placement prediction service
│   ├── validate_app.py              # Stage 6: app validation suite
│   └── build_pdfs.py                # final PDFs from the Markdown reports
├── app/streamlit_app.py             # Streamlit demo
├── models/                          # saved models + model_card.json
├── reports/                         # PDFs, Markdown reports, metrics JSON
└── figures/                         # charts 01–29
```
`CLAUDE.md` (internal engineering notes) and the challenge guide PDF are kept locally but excluded from the
submission repository through `.gitignore`.

## ML tasks and final models
| Task | Models compared | Final choice (test result) |
|---|---|---|
| ETA regression | Mean and median baselines, ridge, random forest, gradient boosting | Gradient boosting: MAE 6.19 min, RMSE 7.73, R² 0.310 |
| Late-risk classification (> 30 min) | Prior baseline, logistic regression, gradient boosting | Gradient boosting: ROC-AUC 0.791, PR-AUC 0.608, precision 0.467, recall 0.813, F1 0.593 at threshold 0.268 |
| Delivery zones | K-Means, k = 2–30 (elbow + silhouette) | k = 22, silhouette 0.948 (zones = cities) |
| Rider segmentation | Not feasible (no person identifier) | Order-level rider-attribute profiles only (k = 3) |

Full comparison table and selection reasons: Business Insights Report §5.1.

## Evaluation metrics
- **Regression:** MAE, RMSE, R², compared with a naive baseline, plus error by segment.
- **Classification:** precision, recall, F1, ROC-AUC (plus PR-AUC), with the confusion matrix at the chosen threshold. Accuracy is never used alone.
- **Clustering:** inertia (elbow) and silhouette score.

## Development workflow
| Stage | Scope | Status |
|---|---|---|
| 1 | Audit & EDA | ✅ Approved |
| 2 | Data cleaning | ✅ Approved |
| 3 | Feature engineering (leakage-tiered) | ✅ Approved |
| 4 | Regression, classification, clustering | ✅ Approved |
| 5 | Business insights, explainability, 3 recommendations | ✅ Approved |
| 6 | Streamlit demo, validation & final PDFs | ✅ Complete (local; not deployed online) |

Each stage requires explicit approval before the next starts (see `CLAUDE.md`).

## Installation (Windows PowerShell)
Requires Python 3.12. The project's virtual environment is `.venv/`.

```powershell
cd D:\Project1
# Only if .venv does not exist yet (it already exists on the development machine):
#   py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```
If PowerShell blocks `Activate.ps1`, you don't need it: call `.\.venv\Scripts\python.exe` directly, as below.

## Run the Streamlit demo
```powershell
cd D:\Project1
.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py
```
Open **http://localhost:8501**. Stop the app with `Ctrl+C` in the terminal.
On another port: `.\.venv\Scripts\python.exe -m streamlit run app/streamlit_app.py --server.port 8502`

**The app needs** these files, already produced by Stages 3–5: `models/*.joblib`, `models/model_card.json`,
`data/processed/feature_engineered_data.csv`, `reports/stage5_metrics.json` and `figures/`.

**What the app does:**
- **Predict an order.** Pick the store (by city), the customer location (distance or coordinates), the order date and time, order type, city type and festival flag. A quick-example menu fills realistic orders. It returns:
  - **ETA** in minutes, with a likely window of ± test MAE;
  - a **delay-risk score** (> 30 min);
  - a **risk level:** Low, Elevated (above the 0.268 alert threshold) or High (riskiest 10% of training orders);
  - suggested action and a per-order "why this prediction" breakdown.
- **Inputs are order-placement only.** Pickup time, rider age and rating, vehicle, multiple deliveries, weather and traffic are never requested.
- **Unknown values** (distance, order time, city type, festival) are filled by the models' training-fitted imputers, with a warning.
- **Invalid inputs** (coordinates outside India, distance ≤ 0 or > 25 km, invalid categories) show an error, never a crash.
- **Other tabs:** the three findings and three recommendations, model performance and selection, data quality and limitations.
- **Caveats are always shown:** > 30 min is a **project benchmark, not an official SLA**, and risk scores are **uncalibrated**.

**Validate the app** (≈ 1–2 min: parity with training, leakage, invalid inputs, UI tests through Streamlit AppTest):
```powershell
.\.venv\Scripts\python.exe -m src.validate_app
```

## Optional: deploying online (not done)
The demo has only been run locally. To publish it on **Streamlit Community Cloud** (free):
1. **Create a dedicated Git repository** for `D:\Project1`. The current Git root is the whole `D:\` drive, which must not be pushed.
   Run `git init` inside `D:\Project1` and push it to a new GitHub repository.
2. **Commit the artifacts the app reads.** `.gitignore` currently excludes `models/*` and `data/processed/*`, so either:
   - remove those two lines and commit `models/` and `data/processed/feature_engineered_data.csv` (about 9.5 MB); or
   - add a start-up step that rebuilds them by running Stages 2–4.

   Also commit `reports/stage5_metrics.json` and `figures/`.
3. **Check the dataset licence** before committing `data/raw/train.csv` or derived data to a public repository.
4. **On share.streamlit.io:** "New app", choose the repository and branch, set the main file to `app/streamlit_app.py` and Python 3.12.
   The root `requirements.txt` is installed automatically.
5. **After deploying,** run the same example orders as `reports/app_validation_report.md` and confirm the numbers match.

Alternatives: Hugging Face Spaces (Streamlit SDK), or any VM / container running the command above with
`--server.address 0.0.0.0`.

## Rebuild the pipeline (Stages 1–5)
```powershell
cd D:\Project1

# Stage 2: rebuild the cleaned dataset + cleaning report (raw file is read-only)
.\.venv\Scripts\python.exe -m src.data_cleaning

# Stage 3: rebuild engineered features, feature dictionary and report
.\.venv\Scripts\python.exe -m src.feature_engineering

# Stage 4: train + evaluate all models, write models/, figures and modelling report (~4-5 min)
.\.venv\Scripts\python.exe -m src.modelling

# Stage 5: verify Stage 4, explainability, geography/rider checks, management reports (~1-2 min)
.\.venv\Scripts\python.exe -m src.business_insights

# Final PDFs from the Markdown reports (needs Google Chrome or Microsoft Edge installed; ~30 s)
.\.venv\Scripts\python.exe -m src.build_pdfs

# Re-run the audit notebook (regenerates figures/ and reports/audit_metrics.json):
.\.venv\Scripts\python.exe -c "import nbformat; from nbclient import NotebookClient; nb=nbformat.read('notebooks/01_EDA.ipynb',4); NotebookClient(nb,timeout=600,kernel_name='python3',resources={'metadata':{'path':'notebooks'}}).execute(); nbformat.write(nb,'notebooks/01_EDA.ipynb')"
```
