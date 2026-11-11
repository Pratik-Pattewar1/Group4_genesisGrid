# DeliveryIQ — Complete Project Audit

**Team Genesis Grid · Group 4 · KLU + PATH3 ML-01 Quick-Commerce Delivery Intelligence**
**Audit date:** 2026-11-11 · **Project location audited:** `D:\Group4_GenesisGrid` · **Type:** audit only

---

## How this audit was done (read first)

- **Audit-only.** No project file was modified, retrained, rebuilt or regenerated. The only file created in the project is
  this report. Every Python run used `PYTHONDONTWRITEBYTECODE=1`, and no `__pycache__` was written.
- **Re-executed checks wrote only to the session scratchpad.** Validation suites, the notebook and pipeline rebuilds were
  run with their outputs redirected there.
- **One local service was started and stopped.** A Streamlit server on port 8510 (PIDs 7028 and 1720) was stopped after
  testing; no other process was touched.
- **Every number below was independently recomputed** from saved artifacts in this audit, unless it is labelled
  **[Reported]**. [Reported] means it is read from an earlier run's output file and was not re-executed here, usually
  because re-running would mean retraining, which this audit forbids.
- **Integrity:** all checksums match the values recorded during development. Raw data:
  `d293b6fc…cdde1`; cleaned data: `3f2f1ee9…`; engineered data: `86f90bec…`; all 6 model files; both Markdown reports.

### Executive summary of findings

| Area | Verdict | One-line reason |
|---|---|---|
| ML results | ✅ Verified | All test metrics recomputed from the saved models match the reports exactly |
| Data pipeline | ✅ Verified | Cleaned and engineered CSVs rebuilt in memory from raw data are **byte-identical** to the saved files |
| Notebook | ✅ Verified | Re-executed copy: 0 errors; `audit_metrics.json` identical; 10/10 figures byte-identical |
| Streamlit app | ✅ Verified | Live server, real browser prediction (ETA 27 min · 32% risk), AppTest suite 18/18 |
| Leakage | ✅ None found | Feature list, guard, imputer statistics and correlation checks all clean |
| PDFs | ✅ Verified | 23 and 6 pages; every source heading, number and figure is present |
| GitHub repo | ⚠️ Attention | Local branch is **2 commits behind GitHub**; GitHub holds an extra Power BI PDF branded "SwiftOpsAI" that uses a **different late definition (40+ min)** |
| Documentation | ⚠️ Stale spots | `D:\Project1` paths, "Git root is D:\" text, "Stage 6 pending" text inside the Solution Document |
| Problem Statements 6 and 7 | ⚠️ Data-limited | Correctly documented as not fully achievable; labelled alternatives provided |
| Test automation | ⚠️ Gap | Strong validation scripts, but no pytest suite or CI |

---

# PART 1 — Official hackathon requirements

Source: `Hackathon_Challenge_Guide.pdf`, 8 pages. All requirements below are taken from the guide; the section number is
in brackets. Items that come only from `CLAUDE.md` are listed separately in Table 1b.

### Deliverable 1 — Official Requirements Master Table

| ID | Official requirement | What must be analysed | What must be implemented | Required calculations / metrics | Expected output / evidence |
|---|---|---|---|---|---|
| R1 | PS1 Data Quality & Reliability (§4) | Missing values, duplicates, invalid GPS (0,0 / outside city), impossible ratings/ages, inconsistent labels, malformed timestamps, delivery-time outliers | Data audit + documented treatment of each issue | Counts per issue; retained / excluded / transformed | Data-quality summary with justified treatment |
| R2 | PS2 Define the Delay Lens (§4) | Definition of "late" and of delivery time; target; distance method; post-delivery columns | Target extraction; late label; Haversine distance; leakage exclusion list | SLA threshold used; distance formula | Written definitions + excluded columns |
| R3 | PS3 Diagnose Delay Patterns (§4) | Delivery time and late rate by hour, day of week, weather, traffic, distance, order type, rider experience | EDA comparisons; association-only language | Late rate and delivery time per group | Strongest patterns; where delays concentrate; no causal claims |
| R4 | PS4 Regression (§4) | Delivery-time prediction | Compare Linear Regression / Random Forest / XGBoost or Gradient Boosting vs a naive baseline (e.g. mean) | MAE, RMSE, R² on held-out test | Comparison + most important features |
| R5 | PS5 Classification (§4) | "Will this order be late?" **at order placement** | Classifier; class-imbalance handling; cost-based threshold | Recall, precision, F1, ROC-AUC (not accuracy alone) | Metrics + justified threshold |
| R6 | PS6 Dark-store zones (§4) | Order locations and demand | K-Means (or approved alternative); k by elbow + silhouette; cluster map | Zone demand, average delivery time, late rate vs network benchmark | Candidate dark-store zones |
| R7 | PS7 Rider segmentation (§4) | Rider behaviour: rating, experience, average delivery time, late rate, deliveries per day | Cluster riders; name and profile segments | Segment profiles | Support / roster / training recommendations per segment |
| R8 | PS8 Select, explain, act (§4) | Final model selection with documented criteria; explainability | Feature importance or SHAP; exactly 3 prioritised recommendations | — | Each recommendation links finding → evidence → action; observed vs projected kept separate |
| R9 | Challenge statement (§3) | — | Observed patterns not presented as causation; projected benefits labelled as estimates | — | Labelled estimates |
| R10 | Metric framework (§6) | — | Late rate = late ÷ total × 100; MAE; RMSE; R²; recall; precision; F1/ROC-AUC; silhouette | Formulas as defined | No silent change of definitions; alternative SLA labelled as additional |
| R11 | Mandatory evidence (§7) | — | (a) data-quality summary; (b) ≥ 3 delay comparisons; (c) baseline + ≥ 2 regression models on held-out test; (d) classifier with recall/precision/F1/ROC-AUC + justified threshold; (e) K-Means with elbow + silhouette; (f) model selection table with choice + reason; (g) 3 findings + 3 recommendations; (h) limitations section | As listed | All eight items present |
| R12 | Recommended comparisons (§8) | Hour vs late rate; weather vs traffic; distance vs delivery time; rider experience vs delay; baseline vs model; model vs model; zone vs network | Recommended (not strictly mandatory) | Gap to average; added delay; R² of distance; MAE improvement | Answers to the example questions |
| R13 | Visualisation set (§9) | — | Delivery-time distribution + late rate by hour; late rate by weather and traffic; actual vs predicted; feature importance / SHAP; PR or ROC with threshold marked; elbow + silhouette + cluster map; rider segment profile chart | — | Few, decision-oriented charts |
| R14 | Workflow (§10) | — | Includes "peak-hour flag" and "rider experience bands" in feature engineering (step 4); tune models; build demo; present | — | Process followed |
| R15 | Technology (§12) | — | Python stack; explainability via importance/permutation/SHAP; Streamlit; Git/GitHub recommended | — | Optional choices |
| R16 | Submission package (§13) | — | Source code/notebook (runnable); model comparison table; Business Report 4–6 pages; Streamlit demo returning ETA + delay-risk %; Business Solution Document ≥ 5 pages; README with setup, dependencies, assumptions, execution | Page counts | All six deliverables |
| R17 | Report structure (§14) | — | Executive summary; business understanding; dataset & quality; EDA; modelling approach; model selection & explainability; 3 recommendations; limitations; reproducibility | — | Recommended structure |
| R18 | Judging rubric (§15) | — | Business understanding; data quality & EDA; supervised modelling without leakage; unsupervised with justified k and useful zone/rider segments; metrics for business cost; recommendations; reproducibility & demo; code clarity | — | Rubric coverage |
| R19 | Challenge boundaries (§16) | — | No training/tuning on test; no post-delivery features; no accuracy-only; no causal claims; no unapproved external data; no arbitrary deletion; no complexity without value; observed vs projected separated | — | Compliance |
| R20 | Expected outcome (§17) | — | Predict delivery time; flag risky orders; identify where new dark stores are needed; show when to surge-staff | — | Working demo + 3 actions |

### Table 1b — Additional project-level requirements (from `CLAUDE.md`, not in the guide)

| ID | Project requirement | Source |
|---|---|---|
| P1 | Raw file immutable (SHA-256 pinned); all 45,593 rows preserved | CLAUDE.md §2 |
| P2 | > 30 min = project benchmark (not an official SLA); sensitivity at > 35 and > 40 | §6 |
| P3 | Chronological split; preprocessing fitted on train only; `SEED = 42` | §4, §8 |
| P4 | Order-placement (Tier A) features only in the primary model; `Time_Order_picked` excluded | §7 |
| P5 | Report MAE/RMSE/R² by segment; PR-AUC and confusion matrix; zone profiles | §8 |
| P6 | Models with and without missing-Festival rows; with and without `order_month` | §8, Stage 3 rules |
| P7 | Explainability on training data only; zones never presented as dark-store evidence; rider profiles labelled "not rider segmentation" | Stage 5 rules |
| P8 | App: exact parity with training; disclaimers on every screen; risk tiers from training data only | Stage 6 rules |

---

# PART 2 — Current project inventory

**Repository:** dedicated Git repository at `D:\Group4_GenesisGrid` (branch `main`, remote
`origin = github.com/Pratik-Pattewar1/Group4_genesisGrid`). 85 tracked files; `CLAUDE.md` and
`Hackathon_Challenge_Guide.pdf` exist locally only (ignored, as intended).

### Deliverable 2 — Project Inventory

| File / component | Purpose | Exists? | Inspected? | Execution status (this audit) | Observations |
|---|---|---|---|---|---|
| `data/raw/train.csv` | Immutable source (45,593 × 20) | ✅ | ✅ | Hash verified | SHA-256 matches CLAUDE.md |
| `data/processed/cleaned_data.csv` (+ schema JSON) | Stage 2 output (45,593 × 33) | ✅ | ✅ | **Rebuilt in memory: byte-identical** | 26/26 cleaning checks pass |
| `data/processed/feature_engineered_data.csv` | Stage 3 output (45,593 × 32) | ✅ | ✅ | **Rebuilt in memory: byte-identical** | 24/24 feature checks pass |
| `notebooks/01_EDA.ipynb` | Stage 1 audit | ✅ | ✅ | **Copy re-executed: 0 errors** | Outputs reproduce exactly |
| `src/data_cleaning.py` | Cleaning + validation | ✅ | ✅ | `clean()` + `validate()` run | Report text says `cd D:\Project1` (stale) |
| `src/audit_cleaned_data.py` | Independent cleaned-data audit | ✅ | ✅ | Run (output redirected) | 9 PASS · 2 REVIEW (documented) |
| `src/investigate_missing_timestamps.py` | Missing-time study | ✅ | ✅ | Imported only (it writes reports) | Output JSON present |
| `src/feature_engineering.py` | Features, leakage guard, split, preprocessor | ✅ | ✅ | `build_features()` + `validate()` run | `PRIMARY_FEATURES` = 14 (Stage 3 set); deployed models use 13 (no `order_month`), as documented |
| `src/regression.py`, `classification.py`, `clustering.py`, `evaluation.py` | Stage 4 components | ✅ | ✅ | Imported; used by recomputation | Small and documented |
| `src/modelling.py`, `modelling_report.py` | Stage 4 orchestrator | ✅ | ✅ | **NOT RUN** (retrains models; forbidden in this audit) | Outputs verified via saved models |
| `src/business_insights.py`, `insights_report.py`, `challenge_gaps.py` | Stage 5 | ✅ | ✅ | **NOT RUN** (overwrites reports); key numbers recomputed independently | All recomputed numbers match |
| `src/predict.py` | Prediction service | ✅ | ✅ | Executed (parity, edge, latency) | Exact parity with training |
| `src/validate_app.py` | App validation suite | ✅ | ✅ | **Executed, 18/18** (outputs redirected) | Takes 280 s |
| `src/build_pdfs.py` | PDF builder | ✅ | ✅ | NOT RUN (overwrites PDFs); PDFs inspected | Relies on `tornado` (not pinned directly) |
| `app/streamlit_app.py` | Demo UI | ✅ | ✅ | **Live server + browser + AppTest** | Works |
| `models/*.joblib`, `model_card.json` | Saved models (ETA; late > 30/35/40; K-Means) | ✅ (6) | ✅ | Loaded and evaluated | Hashes unchanged |
| `reports/*.md` (14), `*.json` (8), `*.pdf` (2) | Documentation and metrics | ✅ | ✅ | PDFs parsed and checked | Stale wording (Part 7) |
| `figures/01–29_*.png` (29) | Charts | ✅ | ✅ (spot) | 10 notebook charts reproduced byte-identically | All referenced figures exist |
| `README.md` | Documentation | ✅ | ✅ | Links resolved | Stale `D:\Project1`, Git-root text, CLAUDE.md references |
| `requirements.txt` | Pinned dependencies | ✅ | ✅ | `pip check` clean | `tornado` used directly but not pinned |
| `.gitignore` | Repository hygiene | ✅ | ✅ | Verified | Correctly excludes `.venv`, caches, CLAUDE.md, guide |
| `.venv/` | Environment (Python 3.12.6) | ✅ | ✅ | `python.exe` works | **Moved from `D:\Project1`: `.exe` launchers (`streamlit.exe`, `pip.exe`) point to the old path and fail.** `python -m …` works |
| Tests (`tests/`, pytest, CI) | Automated test suite | ❌ | — | — | Validation is script-based; no pytest or CI |
| Deployment config (`.streamlit/config.toml`, `runtime.txt`) | Cloud deployment | ❌ | — | — | Not needed locally; not deployed |
| **GitHub-only:** `SwiftOpsAI-Food-Delivery-Prediction.pdf` | Power BI dashboard (6 pages) uploaded on GitHub | Remote only | ✅ (extracted read-only) | — | **Different product name; "% Late Deliveries (40+)" = 10.1%** conflicts with the > 30 min benchmark |

**Missing items:** none required for running. No missing datasets or models, no broken imports (17/17 modules import),
no syntax errors (18/18 files parse).

**Obsolete or duplicate items:**
- `reports/pdf_build_check.json` and `reports/app_validation.json` are generated logs, not duplicates.
- No unused scripts, although `audit_cleaned_data.py` and `investigate_missing_timestamps.py` are one-off audit scripts.

---

# PART 3 — What we have completed (verified)

### Deliverable 3 — Requirement Completion Matrix

| ID | Requirement | Expected work | Actual implementation | Evidence / file | Status | Remaining work |
|---|---|---|---|---|---|---|
| R1 | PS1 Data quality | Audit + treatment | All issues found and treated (counts in Part 5); no rows deleted | `dataset_audit_report.md`, `data_cleaning_report.md`, `final_data_audit.md`; rebuild byte-identical | **COMPLETE & VERIFIED** | — |
| R2 | PS2 Delay lens | Target, late definition, distance, leakage | Target parsed; > 30 min project benchmark (+35/+40); Haversine; Tier A/B/C table; guard rejects 17/17 forbidden columns | `feature_dictionary.md`, `feature_engineering.py` | **COMPLETE & VERIFIED** | — |
| R3 | PS3 EDA | Hour, weekday, weather, traffic, distance, order type, rider experience | All seven done; association language used | Solution Doc §4, §4.1–4.5; figures 03–10, 26–27 | **COMPLETE & VERIFIED** | Rider "experience" uses an age proxy (no tenure field) |
| R4 | PS4 Regression | Baseline + LR/RF/GB; MAE/RMSE/R²; importance | Mean and median baselines, ridge (regularised linear), random forest, gradient boosting; test metrics; permutation importance | `modelling_metrics.json`; recomputed exactly | **COMPLETE & VERIFIED** | XGBoost not used (optional); plain OLS not run (ridge used) |
| R5 | PS5 Classification | Placement-time classifier; imbalance; metrics; threshold | Prior baseline, logistic, gradient boosting; PR-AUC selection + threshold moving; 0.268 threshold from training out-of-fold F1 | Recomputed exactly; figures 16–18, 24, 29 | **COMPLETE & VERIFIED** | Accuracy never reported (see Part 4B) |
| R6 | PS6 Dark-store zones | K-Means; elbow + silhouette; map; zone vs network; candidate zones | K-Means k = 22 (silhouette 0.948; elbow k = 6); map; zone-vs-network table with Bonferroni tests; watch-list (HYD, RANCHI, JAP), none significant | figures 19–20; Solution Doc §8, §8.1 | **BLOCKED BY DATA LIMITATION** (analysis done; "ideal zones" not supportable) | Explain to judges (below) |
| R7 | PS7 Rider segmentation | Cluster riders; profile; support/roster | Shown infeasible; order-level rider-attribute profiles (k = 3, silhouette 0.458), clearly labelled | Solution Doc §9, §9.1; figure 28 | **BLOCKED BY DATA LIMITATION** (alternative provided) | Explain to judges |
| R8 | PS8 Selection + explanation + 3 actions | Documented selection; importance; 3 recommendations | Model selection table; grouped permutation + partial dependence (training only); exactly 3 recommendations, each with 9 fields; observed and scenario labels | `business_recommendations.md`; Solution Doc §5.1, §7 | **COMPLETE & VERIFIED** | "Which zones need a new dark store" is answered honestly as "not determinable" |
| R9 | No causation; estimates labelled | — | [Observed] / [Scenario] labels throughout | Reports | **COMPLETE & VERIFIED** | — |
| R10 | Metric framework | Definitions | Late rate formula used; all listed metrics present | Reports | **COMPLETE & VERIFIED** | — |
| R11a–h | Mandatory evidence (8 items) | — | All 8 present | `challenge_compliance.md` (checked) | **COMPLETE & VERIFIED** | — |
| R12 | Recommended comparisons (7) | — | All 7 done | Solution Doc §4.1–4.5, §5.1, §8.1 | **COMPLETE & VERIFIED** | — |
| R13 | Visualisation set (7) | — | 6 fully; rider segment chart as order-level profiles | figures 03, 13, 15/22, 19–20, 26, 27, 28, 29 | **COMPLETE & VERIFIED** (rider chart data-limited) | — |
| R14 | Workflow features (peak-hour flag, experience bands) | — | `order_time_band` covers peak; age bands in EDA only | `feature_engineering.py`, `challenge_gaps.py` | **PARTIALLY COMPLETE** | No explicit binary "peak-hour flag" feature (time band serves this purpose) |
| R16 | Submission package (6) | — | Code ✅; table ✅; Business Report PDF 6 pp ✅; Streamlit (local) ✅; Solution Doc PDF 23 pp ✅; README ✅ | Part 7 | **COMPLETE & VERIFIED** | Stale text fixes (Part 7) |
| R17 | Report structure | — | Business Report follows §14 exactly (9 sections) | `business_report.md` | **COMPLETE & VERIFIED** | — |
| R19 | Boundaries (8) | — | All satisfied (Part 5, Part 6) | Validation outputs | **COMPLETE & VERIFIED** | — |
| R20 | Expected outcome | — | ETA ✅; risky orders ✅; dark-store need: not supportable ⚠️; surge-staff window ✅ (19:00–21:59) | Reports, app | **PARTIALLY COMPLETE** (data-limited) | — |

### Problem Statements 6 and 7 — independent verification and how to explain them

**PS6 (dark-store zones).** The finding that the zones are just the cities is verified independently:
- **Zones = cities:** the K-Means zones are 100% pure by city code.
- **The high score is city separation:** the silhouette of the *city labels themselves* is **0.9483**, equal to the
  clusters' 0.9482.
- **Zones don't differ in delay:** the zone late-rate difference is not significant (chi-square p = 0.103). After
  Bonferroni correction (α = 0.05/22 = 0.00227), **0 of 22 zones** differ in late rate or ETA; the smallest p is 0.0084.
- **Customer points are synthetic:** latitude offset equals longitude offset on 98.2% of rows (Stage 1).
- **Acceptability:** the guide allows "K-Means or an approved alternative" and requires zone vs network comparison,
  which was done.
- **What to tell judges:** "We ran the required analysis. The data cannot validate store sites, so we report a
  provisional watch-list and list the data needed (real customer coordinates, demand density, costs, service radius)."
  This is consistent with §16, which bars presenting patterns as proof.

**PS7 (rider segmentation).** Also verified independently:
- **The IDs are not people:** there are 1,320 `delivery_person_id`s, each with **4–22 distinct ages**; 100% have more
  than one vehicle type; each maps to exactly one restaurant location; the suffixes are only DEL01–03.
- **Per-rider metrics can't be built:** deliveries per day, experience and per-rider late rate need a person
  identifier, which doesn't exist.
- **Alternative provided:** order-level rider-attribute profiles, labelled as such.
- **What to tell judges:** "Segmenting this ID would profile restaurant slots, not riders. We show the evidence and an
  honest order-level alternative." A low rating may be a *consequence* of lateness (reverse causality).

---

# PART 4 — Machine-learning metrics audit (all recomputed unless marked)

**Split (verified):**
- **Chronological, on whole dates.** Train 2022-02-11 → 2022-03-29 (37,123 rows); test 2022-03-30 → 2022-04-06
  (8,470 rows). No date overlap.
- **Cross-validation:** 3 forward-chaining folds within train; validation windows 03-03→03-11, 03-12→03-20 and
  03-21→03-29 [Reported].

### A. Regression

| Model | Dataset / split | MAE | RMSE | R² | Training status | Evaluation status |
|---|---|---|---|---|---|---|
| Mean baseline (train mean 26.34 min) | Test 8,470 | **7.5115** | 9.3137 | −0.0007 | Trivial | **Recomputed** |
| Median baseline | Test | **7.4737** | 9.3112 | −0.0001 | Trivial | **Recomputed** (= reported) |
| Distance-only linear regression | Test | 7.222 | 8.894 | 0.087 | Trained in Stage 5 | [Reported] |
| Ridge (α = 100) | Test | 6.4186 | 7.9965 | 0.2624 | Not saved | [Reported]; CV MAE 6.6213 |
| Random forest (300 trees, min_samples_leaf = 20) | Test | 6.2023 | 7.7639 | 0.3047 | Not saved | [Reported]; CV MAE 6.2985 |
| **Hist. gradient boosting (final, saved)** | Test | **6.1940** | **7.7335** | **0.3101** | Saved, `random_state = 42`, 150 iterations | **Recomputed exactly**; CV MAE 6.2946 |
| Comparison only: dispatch-time GB | Test | 3.4982 | 4.3732 | 0.7794 | Not saved | [Reported]; not deployable at placement |
| Comparison only: dispatch-time GB without rating | Test | 3.6708 | 4.5435 | 0.7619 | Not saved | [Reported] |
| XGBoost / plain OLS | — | NOT CALCULATED | | | Not implemented | Optional in guide |

- **Selection:** lowest CV MAE on training data, a practical tie with random forest (difference 0.004 min).
- **Improvement on the test set:** 17.1% lower MAE than the median baseline and **17.5%** lower than the mean baseline.
- **The final model was evaluated once on an untouched test set.** Tuning, selection and threshold choice used training
  CV only (code inspected: `src/modelling.py` uses only `train` rows inside `cv_score` and `tune`).
- **Business meaning:**
  - **MAE 6.19 min:** a typical ETA is off by about 6 minutes on a 26-minute average delivery.
  - **RMSE 7.73:** large misses are somewhat more common than MAE alone suggests.
  - **R² 0.31:** order-time information explains about a third of the variation; the rest depends on post-dispatch factors.

### B. Classification (late = > 30 min; test set; saved model recomputed)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | Brier | Threshold | TN / FP / FN / TP |
|---|---|---|---|---|---|---|---|---|---|
| Prior baseline (constant) | 0.7129* | 0.000 | 0.000 | 0.000 | 0.500 | 0.2871 | 0.2049 | 0.5 | 6038 / 0 / 2432 / 0 |
| Logistic (C = 10) [Reported] | 0.5875* | 0.4024 | 0.8997 | 0.5560 | 0.7656 | 0.5702 | 0.1660 | 0.2126 | 2788 / 3250 / 244 / 2188 |
| **Hist. gradient boosting (final)** | **0.6797** | **0.4668** | **0.8133** | **0.5932** | **0.7911** | **0.6084** | 0.1587 | **0.2681** | **3779 / 2259 / 454 / 1978** |
| Final, > 35 min | 0.7653 | 0.3928 | 0.7382 | 0.5127 | 0.8314 | 0.5399 | 0.1041 | 0.2160 | 5436 / 1617 / 371 / 1046 |
| Final, > 40 min | 0.8639 | 0.3357 | 0.5767 | 0.4244 | 0.8519 | 0.4721 | 0.0586 | 0.2010 | 6892 / 841 / 312 / 425 |

\*Accuracy was **never calculated or reported** by the project. The values here are computed in this audit: from the
stored confusion matrices for the baseline and logistic model, and recomputed for the saved models.

- **Class distribution:** test 2,432 late / 6,038 on-time (28.71%); train 30.10% late. At > 35 min: 16.73% test; at
  > 40 min: 8.70% test.
- **Imbalance handling:** models are selected on PR-AUC, and the decision threshold is moved to the F1-best point on
  training out-of-fold predictions. No resampling.
- **⚠️ Accuracy below the trivial baseline:** the final model's accuracy (0.680) is lower than "always predict on-time"
  (0.713).
  - This is a deliberate trade-off: the low threshold buys 81% recall.
  - It is *correct* for a costly-miss problem, but it is undocumented and a judge may ask. Prepare the answer:
    "Accuracy rewards ignoring late orders; at our threshold we catch 1,978 of 2,432 late orders, while the baseline
    catches 0."

### C. Clustering

| Clustering model | Algorithm | k | Inertia | Silhouette | Interpretation |
|---|---|---|---|---|---|
| Delivery zones (saved) | K-Means, km-projected delivery coordinates, `n_init = 10`, seed 42, fitted on 34,016 training geo-valid rows | 22 (silhouette maximum; elbow at 6) | 1,906,291 km² (recomputed) | **0.9482** (recomputed 0.948220; city labels 0.9483) | Zones = cities; no significant late-rate differences; not dark-store evidence |
| Rider-attribute profiles | K-Means on standardised age + rating (order level) | 3 (silhouette maximum over 2–8) | [Reported] | 0.458 [Reported scan] | "Younger, higher-rated" 13.3% late; "Older, higher-rated" 27.0%; "Older, lower-rated" 86.4% late. **Not rider segments** |

- **Silhouette scan [Reported]:** 0.511 (k = 2) … 0.928 (21), **0.948 (22)**, 0.916 (23) … 0.701 (30).
- **Cluster sizes:** 478–2,672 training orders (478–3,443 on all valid rows).
- **Supports the business objective?** Only at city level. Location explains little: late rates span 28.7–34.3% across
  zones, against 2.2–35.5% across time bands.

### D. Explainability

| Item | Done? | Evidence | Training data only? |
|---|---|---|---|
| Single-feature permutation importance (ETA) | ✅ | `modelling_metrics.json`, figure 15. Top: `order_hour_sin` 0.676, `distance_km` 0.588, `festival` 0.358, `city_type` 0.277 (min MAE increase) | ❌ computed on **test** (post-hoc; labelled as such; not used for decisions) |
| Grouped permutation importance (ETA + classifier) | ✅ | `stage5_metrics.json`, figure 22. ETA: time of day 1.002, distance 0.642, festival 0.390, city type 0.263, location 0.042, calendar 0.038, order type 0.010. Classifier ROC-AUC drop: 0.160 / 0.085 / 0.018 / 0.017 / 0.008 / 0.004 / 0.002 | ✅ 37,123 training rows |
| Partial dependence | ✅ | figure 23. ETA 21.8 min at 09:00 vs 29.1 at 20:00; late probability 4% → 42%; festival 26.1 → 40.2 min; Semi-Urban 41.7 min | ✅ 5,000 training rows |
| Model built-in feature importance | ❌ NOT CALCULATED | HistGradientBoosting has none | — |
| SHAP | ❌ NOT IMPLEMENTED | Not installed (deliberate) | — |

### E. Additional analytical calculations

| Calculation | Actual value (recomputed in this audit unless marked) | Status | Source |
|---|---|---|---|
| Late rate > 30 / > 35 / > 40 min (all orders) | 29.84% / 17.56% / 8.85% | ✅ | `audit_metrics.json` |
| Sensitivity models at > 35 and > 40 | See 4B | ✅ | `models/late_classifier_gt35/40` |
| Late rate by hour (peak) | 19:00 49.4%, 20:00 50.6%, 21:00 50.1% (network 29.8%) | ✅ | `stage5_metrics.json`, figure 26 |
| Late rate by weekday | 25.2% (Thu) – 36.2% (Wed); chi-square p = 1.4e-57 [Reported] | ✅ | Solution Doc §4.2 |
| Weather vs traffic | Fog − Sunny within Jam +58.3 pp; within Low +4.1 pp | ✅ | figure 27 |
| Distance-only R² | 0.103 (squared correlation, all valid rows); 0.087 (test, train-fitted model) [Reported] | ✅ | Solution Doc §4.4 |
| Rider rating / age on similar routes | Rating 4.8–5.0 vs < 4.5: −8.93 min, −58.9 pp; age 35–39 vs 20–24: +6.59 min, +24.2 pp [Reported] | ✅ | Solution Doc §4.5 |
| Zone vs network + Bonferroni | 0/22 significant (smallest p 0.0084 vs α 0.00227); overall chi-square p 0.103 | ✅ | Solution Doc §8.1 |
| Missing-Festival analysis | 228 rows: 0% late, mean 11.2 min (artifact); Yes: 896 rows, 100% late; No: 28.6% late | ✅ | Stage 1/5 reports |
| With vs without missing-Festival rows | CV MAE 6.2781 vs 6.2878 (0.16%, below the 0.5% rule → kept) [Reported] | ✅ | `modelling_metrics.json` |
| With vs without `order_month` | CV MAE 6.2946 vs 6.2878 → dropped [Reported] | ✅ | same |
| With vs without the 1,731 missing-time rows | CV MAE 6.3175 vs 6.2878 → kept [Reported] | ✅ | same |
| Class-balance statement | 30.1% train / 28.7% test | ✅ | Solution Doc §6 |
| Calibration (reliability curve) | NOT CALCULATED (Brier only) | ⚠️ | — |
| Accuracy | NOT CALCULATED by the project (computed here: 0.680) | ⚠️ | — |

**Code support:** every value in this table is produced by code (`src/modelling.py`, `src/business_insights.py`,
`src/challenge_gaps.py`) and stored in JSON. None appears only in prose.

---

# PART 5 — Dataset and data-pipeline audit

### Deliverable 4 — Data Pipeline Verification Table

| Pipeline stage | Expected behaviour | Actual behaviour | Rows / columns | Evidence | Status |
|---|---|---|---|---|---|
| Raw load | Read as text, hash-checked | `load_raw()` verifies SHA-256 | 45,593 × 20 | Hash OK | ✅ |
| Missing values | Text `NaN` tokens → nulls; no imputation in data | 4,225 rows with ≥ 1 missing; imputation only inside train-fitted pipelines | — | Cleaning checks | ✅ |
| Duplicates | Detect | 0 full, 0 excluding ID, 0 duplicate IDs | — | `audit_metrics.json` | ✅ |
| Invalid GPS | Flag (0,0); validated sign fixes | 3,640 flagged invalid (41,953 valid); 431 sign fixes, all passing 4 checks | — | `dq_*` flags recomputed | ✅ |
| Outliers | Document | 270 target IQR outliers kept (within 10–54) | — | Reports | ✅ |
| Cleaning | All rows kept; originals preserved | 26/26 checks; 91 placeholder riders blanked; 831 midnight roll-overs | 45,593 × 33 | **Rebuild byte-identical** | ✅ |
| Feature engineering | Tier A features; missing stays missing | 24/24 checks; shuffled-target test passes | 45,593 × 32 | **Rebuild byte-identical** | ✅ |
| Split | Chronological, whole dates | Cut 2022-03-30; no overlap | 37,123 / 8,470 | Recomputed | ✅ |
| Preprocessing | Fitted on train only | Imputer medians equal **train** medians (and differ from full-data medians) | — | Recomputed | ✅ |
| Training | CV on train; seed 42 | Code inspected; `random_state = 42` | — | `src/modelling.py` | ✅ (NOT re-run) |
| Evaluation | Test used once | Metrics recomputed exactly | — | Part 4 | ✅ |
| Serialization | Saved pipelines + metadata | 6 artifacts load; hashes pinned | — | joblib | ✅ |
| Prediction pipeline | Same features as training | Exact parity on 600 single orders (max difference 0.0) | — | `validate_app` | ✅ |
| Streamlit | Uses saved models via `predict.py` | Verified live and with AppTest | — | Part 6 | ✅ |

**Project rules:**

| Rule | Result |
|---|---|
| Raw unchanged | ✅ |
| 45,593 rows preserved | ✅ |
| SHA-256 matches CLAUDE.md | ✅ |
| Chronological split | ✅ |
| Train-only preprocessing | ✅ |
| No leakage | ✅ |
| No target-derived features | ✅ (shuffled-target test; max \|Spearman\| of any feature with target = 0.322) |
| `Time_Order_picked` excluded | ✅ (not carried into the feature data) |
| `Delivery_person_ID` not a rider key | ✅ (raw ID and `city_code` rejected by the guard) |
| GPS handling per owner decisions | ✅ |
| `SEED = 42` | ✅ in code |

**Reproducibility:**
- Raw → cleaned → engineered: **reproduced byte-for-byte** in this audit.
- Engineered → models: **NOT re-run** (it would retrain).
- Stage 4 [Reported] includes a deterministic-refit check, and this audit confirmed repeated predictions are identical.

---

# PART 6 — Testing and QA audit

### Deliverable 5 — Testing and QA Report

| Test ID | Category | Test case | Expected | Actual | Result | Evidence |
|---|---|---|---|---|---|---|
| T01 | Python | Parse all 18 source files | No syntax errors | 18/18 parse | **PASS** | AST parse |
| T02 | Python | Import all 17 `src` modules | No import errors | 17/17 | **PASS** | importlib |
| T03 | Python | Hard-coded machine paths in code logic | None | None in logic; `D:\Project1` only inside report text strings (4 files) | **PASS** (doc staleness noted) | grep |
| T04 | Python | Public functions documented | Docstrings | Missing on ~40 small helpers (`sha`, `main`, `check`, …) | **PARTIAL** | AST scan |
| T05 | Notebook | Stored notebook | 0 errors, sequential | 39 cells / 22 code; 0 errors; counts 1–22 | **PASS** | nbformat |
| T06 | Notebook | Re-execute copy end to end | 0 errors; same outputs | 0 errors in 56 s; `audit_metrics.json` identical; 10/10 figures byte-identical | **PASS** | nbclient (scratchpad) |
| T07 | Dataset | Raw hash and shape | Match | Match; 45,593 × 20 | **PASS** | sha256 |
| T08 | Dataset | Cleaning validation suite | All pass | 26/26 | **PASS** | `validate()` |
| T09 | Dataset | Independent cleaned-vs-raw audit | Pass | 9 PASS, 2 REVIEW (no `test.csv`; feature exclusion enforced later) | **PASS** | `audit_cleaned_data` |
| T10 | Dataset | Feature validation suite | All pass | 24/24 | **PASS** | `validate()` |
| T11 | Dataset | Cleaned and engineered reproducibility | Byte-identical | Both identical | **PASS** | In-memory rebuild |
| T12 | ML | Saved models load | Load | 6/6 | **PASS** | joblib |
| T13 | ML | Test metrics reproduce reports | Exact | Exact (MAE, ROC-AUC, confusion) | **PASS** | Part 4 |
| T14 | ML | Leakage guard | Rejects forbidden columns | 17/17 rejected | **PASS** | `assert_no_leakage` |
| T15 | ML | Imputer fitted on train only | Train medians | True (≠ full-data medians) | **PASS** | Recompute |
| T16 | ML | Repeated prediction stability | Identical | Identical (batch and 50 single runs) | **PASS** | Recompute |
| T17 | ML | Label-permutation sanity (20 / 10 shuffles) | No skill | [Reported] passed in Stage 4 | **NOT RUN** (retrains) | `modelling_metrics.json` |
| T18 | ML | Full retraining reproducibility | Same metrics | — | **NOT RUN** (forbidden) | — |
| T19 | Integration | App path vs training path (600 single orders incl. 300 unknown/invalid) | Identical | max \|ΔETA\| = 0.0, max \|Δrisk\| = 0.0 | **PASS** | `validate_app` |
| T20 | App (AppTest) | Loads; 4 tabs render | No exception | OK | **PASS** | `validate_app` |
| T21 | App (AppTest) | 4 example orders: UI equals service | Match | Match (18.1 min/2%/Low; 32.9/61%/High; 24.0/16%/Low; 18.0/2%/Low) | **PASS** | `validate_app` |
| T22 | App (AppTest) | Invalid store coordinates | Error message, no crash | Error shown | **PASS** | `validate_app` |
| T23 | App (AppTest) | Unknown order time | Warning | Warning shown | **PASS** | `validate_app` |
| T24 | App (AppTest) | Disclaimers (SLA, uncalibrated) and 3 recommendations | Visible | Visible | **PASS** | `validate_app` |
| T25 | Validation | 9 invalid inputs rejected | Rejected | 9/9 | **PASS** | `validate_app` |
| T26 | Boundaries | 10 edge inputs (25.0 / 25.01 km, 0.01 km, NaN, negative, lat 91, 00:00, 03:00, year 2030, lower-case category) | Safe handling | All handled; **year 2030 scored with no warning** | **PASS** (minor gap) | Audit script |
| T27 | Live server | Start and health | 200 | Healthy after 9.2 s; root page 200 | **PASS** | Port 8510 |
| T28 | Live E2E (browser) | Load UI, click Predict | Result shown | First render 25.0 s; result in 3.5 s: "ETA 27 min · 32% risk · Elevated" | **PASS** | Headless Chrome + screenshot |
| T29 | Live E2E (browser) | Switch tabs | Each tab renders | Tab selector did not match, so not verified live (AppTest covers it) | **NOT VERIFIED** | — |
| T30 | Performance | Single prediction latency (incl. 6-factor explanation) | Interactive | Median 321 ms, p95 402 ms | **PASS** | 50 runs |
| T31 | Performance | Batch inference | Fast | 0.30 ms per row | **PASS** | 8,470 rows |
| T32 | Performance | Server memory after predictions | Reasonable | 236 MB working set (680 MB private) | **PASS** | Get-Process |
| T33 | Environment | `pip check` | Consistent | No broken requirements | **PASS** | pip |
| T34 | Environment | `streamlit.exe` launcher in moved venv | Works | Points to the deleted `D:\Project1` path; `python -m streamlit` works | **FAIL** (workaround exists) | Launcher inspection |
| T35 | Environment | Fresh install on another machine | Installs and runs | — | **NOT RUN** (no new installs allowed) | — |
| T36 | Docs | PDFs vs Markdown | Complete | 378/378 and 76/76 numbers; 26/26 and 10/10 headings; 17/17 and 4/4 figures; footer on every page | **PASS** | pypdf |
| T37 | Security | Credentials in tracked files and all commits | None | None found; no e-mail addresses | **PASS** | git grep |
| T38 | Automation | pytest suite / CI | Present | Absent | **NOT IMPLEMENTED** | — |

**Distinction:**
- **Implemented and executed here, passed:** T01–T16, T19–T28 and T30–T33.
- **Implemented and executed here, partial:** T04 (some helpers lack docstrings).
- **Failed:** T34.
- **Executed but not verified:** T29.
- **Not run (by rule):** T17, T18, T35.
- **Not implemented:** T38.

Previous claims of "tested" were confirmed for the validation suites. There is no frontend-unit, API or CI layer; the
app is a single Streamlit process with no separate backend or API.

---

# PART 7 — Documentation and submission audit

| Check | Result | Evidence |
|---|---|---|
| Business Report 4–6 pages | ✅ 6 pages | pypdf |
| Business Solution Document ≥ 5 pages | ✅ 23 pages | pypdf |
| PDF readability | ✅ | Text extraction; no clipped text blocks (earlier rasterised check) |
| Cover + team | ✅ All 5 names, team, group | Page 1 text |
| Table of contents | ✅ | "Contents" present |
| Page numbering | ✅ "Page X of N" on every page | pypdf |
| Referenced charts included | ✅ 17/17 and 4/4 | Image count |
| Tables readable / no cut-off | ✅ | Earlier visual inspection + edge check |
| README install and run | ✅ | Sections present |
| Requirements complete | ⚠️ `tornado` used by `build_pdfs.py` but only installed through Streamlit | AST import scan |
| Source runnable | ✅ (with `python -m`) | T02, T27 |
| Models and data available in repo | ✅ Tracked (85 files; largest 11.8 MB) | git ls-files |
| No secrets | ✅ | T37 |
| `.gitignore` keeps required artifacts | ✅ | Only `.venv`, caches, CLAUDE.md, guide and logs excluded |

**Stale or incorrect documentation found:**
1. **Solution Document (MD and PDF):** the objectives table still says "Interactive demo … ⏳ Stage 6", and the
   conclusion says "The next step is Stage 6". The demo is complete.
2. **README:**
   - `cd D:\Project1` (×3) and a `Project1/` tree root; the folder is now `D:\Group4_GenesisGrid`.
   - The deployment section says "the current Git root is the whole `D:\` drive". It is now a dedicated repository
     that has been pushed.
   - It references `CLAUDE.md` (lines 130, 158) and the guide PDF, neither of which is in the repository.
3. **Reports:** `cd D:\Project1` in the reproduce blocks of `business_report.md` and 3 stage reports; `CLAUDE.md`
   references in 3 reports.
4. **`CLAUDE.md` (local only):** still says "never rename the project folder `D:\Project1`". The folder was renamed. The
   code is unaffected because all paths are relative.

---

# PART 8 — GitHub and deployment readiness

| Question | Finding |
|---|---|
| Repository root | `D:\Group4_GenesisGrid` (dedicated). ✅ The earlier risk of the whole `D:\` drive being the repository is resolved |
| Remote | `origin` = `github.com/Pratik-Pattewar1/Group4_genesisGrid` |
| Sync state | **Local `main` is 2 commits behind `origin/main`** (by the last fetched refs; no fetch performed in this audit): `30f8ae8 Add files via upload` (adds `SwiftOpsAI-Food-Delivery-Prediction.pdf`) and `3455c5f Update README.md` (by a teammate; no content difference from the local README) |
| Required files included | ✅ 85 tracked: code, data (raw, cleaned, engineered), models, figures, reports, PDFs |
| Large files | ✅ Largest 11.8 MB (below GitHub's 100 MB per-file limit) |
| Excluded correctly | ✅ `.venv`, `__pycache__`, CLAUDE.md, guide, logs |
| Sensitive data in history | ✅ None found in any commit |
| README public-ready | ⚠️ Stale paths and statements (Part 7) |
| Runs on another computer | Likely (relative paths, pinned requirements), but **not tested** (T35). Needs Windows-specific steps only for the PDF build (Chrome or Edge) |
| Deployed? | **No.** No public URL exists. Optional per guide (§13 asks for a *working* app, not a hosted one) |
| Deployment needs | Streamlit Community Cloud: repository ✅, `requirements.txt` ✅ (Python 3.12 selected in UI). Optional: `.streamlit/config.toml`. Model and data artifacts are already tracked. The first page load takes about 25 s (loads 9.5 MB of reference data) |

**Risk:** the GitHub-only `SwiftOpsAI-Food-Delivery-Prediction.pdf` (a Power BI dashboard) uses another product name and
a "% Late Deliveries (40+)" = 10.1% headline that conflicts with the project's > 30 min benchmark (29.84%). Judges
browsing the repository will see two inconsistent definitions. Decide whether to keep it (and explain it in the README
as an additional dashboard) or remove it. Then `git pull` before any further local commit, to avoid a diverged history.

---

# PART 9 — Consolidated audit tables

### Table 1 — Official hackathon requirements

| No. | Requirement | Required analysis / implementation | Required metrics / calculations | Expected deliverable |
|---|---|---|---|---|
| 1 | PS1 Data quality | Audit missing, duplicates, GPS, impossible values, labels, timestamps, outliers | Counts; treatment | Data-quality summary |
| 2 | PS2 Delay lens | Target, late definition, distance, leakage columns | SLA threshold; Haversine | Definitions |
| 3 | PS3 EDA | Hour, weekday, weather, traffic, distance, order type, rider experience | Late rate / delivery time per group | ≥ 3 comparisons, association only |
| 4 | PS4 Regression | Baseline + LR/RF/GB | MAE, RMSE, R² (test); importance | Comparison + features |
| 5 | PS5 Classification | Placement-time classifier; imbalance | Recall, precision, F1, ROC-AUC; threshold | Metrics + threshold justification |
| 6 | PS6 Zones | K-Means on locations and demand | Elbow, silhouette; zone demand / ETA / late vs network | Candidate zones + map |
| 7 | PS7 Riders | Cluster riders | Rating, experience, avg time, late rate, deliveries/day | Named segments + actions |
| 8 | PS8 Selection & actions | Documented selection; importance/SHAP | — | Exactly 3 actions; observed vs projected |
| 9 | Visualisations | 7 chart types | — | Decision-oriented charts |
| 10 | Submission | Code, comparison table, Business Report (4–6 pp), Streamlit (ETA + risk %), Solution Doc (≥ 5 pp), README | Page counts | Package |
| 11 | Boundaries | No test tuning, leakage, accuracy-only, causation, external data, deletion | — | Compliance |

### Table 2 — Actual project completion

| No. | Requirement | What we have done | Actual results / values | Evidence | Status |
|---|---|---|---|---|---|
| 1 | PS1 | Full audit + reproducible cleaning | 4,225 missing rows; 0 duplicates; 3,640 invalid GPS; 431 sign fixes; 91 placeholders; 1,731 missing times; 0 rows deleted | Rebuild byte-identical; 26/26 | COMPLETE & VERIFIED |
| 2 | PS2 | Benchmark + Haversine + guard | > 30 min (29.84%); +35 / +40 sensitivity; 17/17 forbidden columns rejected | Leakage tests | COMPLETE & VERIFIED |
| 3 | PS3 | 7 comparison groups | Peak 19–21 h ≈ 50% late vs 29.8%; Fog−Sunny in Jam +58.3 pp; distance R² 0.087; weekday 25.2–36.2% | Recomputed | COMPLETE & VERIFIED |
| 4 | PS4 | 2 baselines + ridge + RF + GB | GB MAE 6.194 / RMSE 7.734 / R² 0.310; −17.5% vs mean baseline | Recomputed | COMPLETE & VERIFIED |
| 5 | PS5 | Prior + logistic + GB; threshold 0.268 | Precision 0.467 / recall 0.813 / F1 0.593 / ROC-AUC 0.791 / PR-AUC 0.608 | Recomputed | COMPLETE & VERIFIED |
| 6 | PS6 | K-Means k = 22 + zone-vs-network tests | Silhouette 0.948 = city separation; 0/22 zones significant | Recomputed | BLOCKED BY DATA LIMITATION |
| 7 | PS7 | Infeasibility proof + order-level profiles | 1,320 IDs with 4–22 ages each; profiles k = 3 (silhouette 0.458) | Recomputed | BLOCKED BY DATA LIMITATION |
| 8 | PS8 | Selection table, explainability, 3 actions | Time of day top driver (ΔMAE 1.00); 3 recommendations × 9 fields | Files | COMPLETE & VERIFIED |
| 9 | Visualisations | 29 figures | All 7 guide types (rider chart at order level) | `figures/` | COMPLETE & VERIFIED |
| 10 | Submission | All 6 items | PDFs 6 pp / 23 pp; app works locally | Part 7 | COMPLETE & VERIFIED (stale text to fix) |
| 11 | Boundaries | All 8 respected | — | Parts 4–6 | COMPLETE & VERIFIED |

### Table 3 — Missing calculations and metrics

| No. | Metric / calculation | Required? | Already calculated? | Actual value | Where missing | Action needed |
|---|---|---|---|---|---|---|
| 1 | Classifier accuracy (with baseline comparison) | Not required (guide says "not accuracy alone") | No (audit computed it) | 0.680 vs all-negative 0.713 | All reports | Add one sentence explaining why accuracy is below baseline |
| 2 | Probability calibration (reliability curve / calibrated model) | No (project limitation) | Brier only | Brier 0.159 | Reports, app | Optional; already disclosed as uncalibrated |
| 3 | XGBoost model | Optional ("or Gradient Boosting") | No | — | — | None required |
| 4 | Plain OLS linear regression | Guide names "Linear Regression"; ridge is its regularised form | Ridge only | Ridge MAE 6.419 | Model table | Optional: label ridge "regularised linear regression" (already) or add OLS |
| 5 | Deliveries per day / per-rider late rate (PS7) | Yes | Impossible (no person ID) | — | — | Keep documented limitation |
| 6 | Explicit binary peak-hour flag feature | Workflow suggestion | No (time band used) | — | Features | Optional |
| 7 | SHAP | Optional ("where time permits") | No | — | — | Optional |
| 8 | Fresh-machine install test | Implied by reproducibility | No | — | — | Run once on a clean PC or venv |
| 9 | Model retraining reproducibility (this audit) | Good practice | [Reported] in Stage 4 | — | — | Optional re-run in a sandbox (needs approval) |

### Table 4 — Testing summary

| No. | Test category | Tests available | Executed (this audit) | Passed | Failed | Not tested |
|---|---|---|---|---|---|---|
| 1 | Python code (syntax, imports, paths, docstrings) | 4 | 4 | 3 (+1 partial) | 0 | 0 |
| 2 | Notebook | 2 | 2 | 2 | 0 | 0 |
| 3 | Dataset (hash, cleaning suite, independent audit, feature suite, rebuild) | 5 (60+ assertions) | 5 | 5 | 0 | 0 |
| 4 | Machine learning (load, metrics, leakage, imputer, stability, permutation, retrain) | 7 | 5 | 5 | 0 | 2 (by rule) |
| 5 | Integration (parity) | 1 | 1 | 1 | 0 | 0 |
| 6 | Streamlit (AppTest + live) | 9 | 9 | 8 | 0 | 1 (live tab switch not verified) |
| 7 | Validation / boundary inputs | 2 (19 cases) | 2 | 2 (1 minor gap) | 0 | 0 |
| 8 | Performance / reliability | 3 | 3 | 3 | 0 | 0 |
| 9 | Environment / portability | 3 | 2 | 1 | 1 (`streamlit.exe` launcher) | 1 (fresh install) |
| 10 | Documentation / PDF / security | 2 | 2 | 2 | 0 | 0 |
| 11 | Automated suite / CI | 0 | — | — | — | Not implemented |
| **Total** | | **38** | **35** | **32 (+1 partial)** | **1** | **4 + CI** |

### Table 5 — Remaining work

| Priority | Remaining task | Why it is needed | Expected output | Estimated effort |
|---|---|---|---|---|
| **P0** | Resolve the GitHub divergence: decide on `SwiftOpsAI-Food-Delivery-Prediction.pdf` (remove, or rename and explain its 40+ definition), then `git pull` | The public repository shows a conflicting late definition and product name; local branch is behind | Consistent repository | 15 min |
| **P0** | Fix stale Stage 6 wording in `business_insights_report.md` / PDF (objectives table + conclusion) and rebuild the PDF | Judges read "demo pending" in the final document | Corrected PDF (23 pp) | 15 min |
| **P1** | Update README: `D:\Project1` → generic `cd <project folder>`; remove the "Git root is D:\" text; drop or explain `CLAUDE.md` references | Public README must be correct | Clean README | 20 min |
| **P1** | Recreate `.venv` in place, or document "always use `python -m`" | `streamlit.exe` and `pip.exe` launchers are broken after the move | Working launchers | 10 min |
| **P1** | Add an accuracy note (0.680 vs 0.713 baseline) to the classification sections | Likely judge question | One paragraph | 10 min |
| **P1** | Prepare the demo: pre-load the app before presenting (first load takes about 25 s); rehearse PS6/PS7 answers | Live demo reliability | Rehearsed demo | 30 min |
| **P1** | Fresh-machine test: clone → venv → `pip install -r requirements.txt` → run app | Proves reproducibility claims | Confirmed instructions | 30 min |
| **P2** | Pin `tornado` in `requirements.txt` (used by `build_pdfs.py`) | Explicit dependency | 1 line | 2 min |
| **P2** | Warn when the order date is outside the training period (e.g. 2030) | Input robustness | Warning message | 15 min |
| **P2** | Wrap validation scripts in a pytest suite (+ optional GitHub Actions) | Professional test automation | `tests/` | 1–2 h |
| **P2** | Optional deployment to Streamlit Community Cloud | Public demo link | URL | 30–60 min |
| **P2** | Optional: calibration curve, SHAP, OLS, peak-hour flag | Extra depth | Charts / tables | 1–2 h |
| **P2** | Update local `CLAUDE.md` path rules to the new folder | Internal accuracy | — | 5 min |

---

# PART 10 — Final senior-level review

1. **All eight problem statements addressed?** Yes. PS1–5 and PS8 are fully done. PS6 and PS7 were attempted, and the
   data-imposed limits are proven and documented with labelled alternatives.
2. **Official requirements satisfied?** Yes for every mandatory evidence item and submission deliverable, with the
   PS6/PS7 caveats. Hosting the demo is not required.
3. **All mandatory metrics calculated?** Yes: MAE, RMSE, R², recall, precision, F1, ROC-AUC, silhouette and late rate.
   Accuracy was never reported (allowed, but should be explained).
4. **Models justified?** Yes. Selection used training-CV only with pre-registered rules. Random forest ties gradient
   boosting, which should be stated openly.
5. **Clustering defensible?** Yes as an honest negative result. It is *not* defensible as dark-store placement, and the
   project does not claim it is.
6. **Limitations documented?** Yes, thoroughly: synthetic data, 8-day test window, invalid GPS, missing times, benchmark
   not an SLA, uncalibrated probabilities, PS6/PS7.
7. **Any data leakage?** **None found.** Checked by feature list, guard, train-only imputer, shuffled-target test and
   chronological split. Dispatch-time models are comparison-only.
8. **Pipeline reproducible?** Raw → engineered: reproduced **byte-identically**. Modelling: deterministic by design;
   not re-run here.
9. **All Python scripts tested?** All parse and import. The validators were executed. Report generators and
   `modelling.py` were not executed (they would overwrite outputs or retrain).
10. **All notebooks executed?** Yes: the single notebook re-executed cleanly, with identical outputs.
11. **Models independently validated?** Yes: every test metric was recomputed from the saved artifacts and matches.
12. **Streamlit tested end to end?** Yes: live server and browser click → prediction, plus 18 AppTest/service checks.
    Live tab switching was not separately verified.
13. **App functioning correctly?** Yes. Exact parity with training, safe on invalid input, and disclaimers shown.
14. **PDFs and README submission-ready?** PDFs yes, apart from the stale "Stage 6" sentences. README needs path and
    staleness fixes.
15. **Ready for GitHub submission?** Almost: resolve the 2-commit divergence and the SwiftOpsAI PDF first.
16. **Ready for a live demo?** Yes, provided the app is pre-loaded and started with `python -m streamlit`.
17. **Likely judge questions:**
    - Why is R² only 0.31?
    - Why not use traffic and weather in the main model?
    - Why is accuracy below the always-on-time baseline?
    - How was the 0.268 threshold chosen, and what does a false alarm cost?
    - Why 22 clusters, and why isn't that a dark-store answer?
    - Why no rider segments?
    - Is the 100% festival late rate real?
    - How do you know there is no leakage?
    - Why a chronological split?
    - Why gradient boosting over random forest when they tie?
    - How would the two-stage ETA work in production?
18. **Top five weaknesses:**
    1. Modest order-time accuracy (R² 0.31, F1 0.59), by design.
    2. PS6 and PS7 cannot deliver what the rubric's wording expects ("useful zone and rider segments").
    3. Synthetic-looking data with a short 8-day test window.
    4. Repository inconsistency (SwiftOpsAI PDF using 40+, local behind remote) and stale documentation text.
    5. No automated test suite or CI, and no fresh-machine install test; also uncalibrated probabilities.

### Final verdict

| Dimension | Status | Basis |
|---|---|---|
| Technical implementation | **READY WITH MINOR IMPROVEMENTS** | Clean architecture, exact parity; broken exe launchers after the folder move |
| Machine-learning quality | **READY WITH MINOR IMPROVEMENTS** | Sound, leakage-free, verified; modest accuracy by design; accuracy note missing |
| Data quality and reliability | **READY** | Fully audited, reproducible byte-for-byte |
| Testing and reproducibility | **READY WITH MINOR IMPROVEMENTS** | Strong executed validation; no pytest/CI or fresh-install test |
| Business analysis | **READY** | Evidence-backed, labelled estimates, honest limits |
| Documentation | **READY WITH MINOR IMPROVEMENTS** | Stale paths and "Stage 6" sentences |
| Hackathon compliance | **READY WITH MINOR IMPROVEMENTS** | All mandatory items; PS6/PS7 data-limited |
| Submission readiness | **READY WITH MINOR IMPROVEMENTS** | Fix the repository divergence + stale text (P0) |
| Live demo readiness | **READY WITH MINOR IMPROVEMENTS** | Works; pre-load to avoid the 25 s first render |

### Readiness score — method and result

**Method:**
- **Requirements scored:** the 22 mandatory items in the guide: 8 problem statements (§4), 8 mandatory evidence items
  (§7) and 6 submission deliverables (§13).
- **Points:** COMPLETE & VERIFIED = 1; BLOCKED BY DATA LIMITATION with a documented alternative, or PARTIAL = 0.5; NOT
  IMPLEMENTED = 0.
- **What it measures:** coverage of requirements only, not the quality of the results.

| Group | Items | Score |
|---|---|---|
| Problem statements | 8 | 7.0 (PS6 = 0.5, PS7 = 0.5) |
| Mandatory evidence | 8 | 8.0 |
| Submission deliverables | 6 | 6.0 |
| **Total** | **22** | **21.0 / 22 = 95.5%** |

The 4.5% gap is entirely the PS6/PS7 data limitation and cannot be closed honestly with this dataset. The P0/P1 items
above don't change the score, but they affect how judges perceive the submission.

---

*Evidence scripts and outputs from this audit are in the session scratchpad (not part of the project):
`audit_ml.py/.json`, `audit_pipeline.py/.json`, `audit_nb.py`, `audit_misc.py/.json`, `audit_stage5.py/.json`, `cdp_live.py`
and live screenshots.*
