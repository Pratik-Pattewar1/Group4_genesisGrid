# DeliveryIQ — Compliance with Hackathon_Challenge_Guide.pdf

Generated 2026-10-31 by `src/business_insights.py`. Each mandatory item in the guide is mapped to its evidence.

| Requirement (guide section) | Status | Evidence |
|---|---|---|
| PS1 Data quality | ✅ | dataset_audit_report.md, data_cleaning_report.md, final_data_audit.md |
| PS2 Delay lens (target, late definition, Haversine, leakage columns) | ✅ | feature_dictionary.md; > 30 min = project benchmark (no official SLA given) |
| PS3 EDA: hour, day of week, weather, traffic, distance, order type, rider experience | ✅ | Solution Doc §4, §4.1–4.5; figures 26–27 |
| PS4 Regression: LR / RF / GB vs naive baseline; MAE, RMSE, R²; feature importance | ✅ | modelling_report.md; Solution Doc §5.1; figures 12–15, 22 |
| PS5 Classifier at placement: imbalance, recall/precision/F1/ROC-AUC, cost-based threshold | ✅ | Solution Doc §6; figures 16–18, 24, 29 |
| PS6 K-Means zones: elbow + silhouette, map, zone vs network | ⚠️ done; data cannot validate dark-store sites | Solution Doc §8, §8.1; figures 19–20, 25 |
| PS7 Rider segmentation | ⚠️ not feasible as specified; order-level alternative provided | Solution Doc §9, §9.1; figure 28 |
| PS8 Model selection, explainability, exactly 3 recommendations | ✅ | Solution Doc §5.1, §7, §10; business_recommendations.md |
| Mandatory: data-quality summary | ✅ | Business Report §3 |
| Mandatory: ≥ 3 EDA delay comparisons | ✅ | hour, weather × traffic, distance, rider rating/age, day of week |
| Mandatory: baseline + ≥ 2 regression models on held-out test | ✅ | mean & median baselines, ridge, RF, GB |
| Mandatory: classifier with recall, precision, F1, ROC-AUC + justified threshold | ✅ | threshold 0.268 (training OOF, F1-max, recall-favoured) |
| Mandatory: K-Means with elbow and silhouette | ✅ | figure 19 |
| Mandatory: model selection table with final choice and reason | ✅ | Solution Doc §5.1; Business Report §6 |
| Mandatory: 3 findings + 3 prioritised recommendations | ✅ | Solution Doc §4.6; business_recommendations.md |
| Mandatory: limitations section | ✅ | Solution Doc §12; Business Report §8 |
| Visual: delivery-time distribution + late rate by hour | ✅ | figures 03, 26 |
| Visual: late rate by weather and traffic | ✅ | figure 27 |
| Visual: actual vs predicted | ✅ | figure 13 |
| Visual: feature importance | ✅ | figures 15, 22 |
| Visual: PR/ROC with chosen threshold marked | ✅ | figure 29 |
| Visual: elbow, silhouette, cluster map | ✅ | figures 19, 20 |
| Visual: rider segment profile chart (rating, experience, late rate) | ⚠️ order-level profiles, labelled as such | figure 28 |
| Submission: source code / notebook | ✅ | src/*.py, notebooks/01_EDA.ipynb |
| Submission: model comparison table | ✅ | Solution Doc §5.1 |
| Submission: Business Report 4–6 pages | ✅ (by word-count estimate) | business_report.md — 2,123 words, about 4.2 pages at about 500 words/page |
| Submission: Business Solution Document ≥ 5 pages | ✅ (by word-count estimate) | business_insights_report.md — 6,218 words, about 12.4 pages |
| Submission: Streamlit demo (ETA + delay-risk %) | ✅ working locally (not deployed online) | app/streamlit_app.py; reports/app_validation_report.md (python -m src.validate_app) |
| Submission: README with setup, dependencies, assumptions, execution | ✅ | README.md |
| Boundaries: no test training/tuning, no post-delivery features, no accuracy-only, no causal claims, no deletion, observed vs projected | ✅ | Stage 4/5 validation checks; [Observed]/[Scenario] labels |
| Final formatting of reports as PDF/DOCX | ⏳ Stage 6 polish | reports are Markdown |

## Requirements the data cannot satisfy as worded

- **Ideal dark-store locations (PS6, §17):**
  - The clusters are the 22 cities (silhouette of the city labels = cluster silhouette).
  - No zone differs significantly from the network: 0 significant after correction.
  - Delivery points are synthetic offsets.
  - We report a provisional watch-list instead, and the data needed.
- **Rider segments for rostering and training (PS7):**
  - `Delivery_person_ID` is a restaurant/slot code with 4–22 ages per ID.
  - Per-rider history, experience and deliveries per day cannot be built.
  - We provide order-level rider-attribute profiles, clearly labelled.
- **Official SLA:** neither the data nor the guide gives one. The > 30 min benchmark is labelled project-defined, with 35 and 40 min sensitivity, consistent with guide §6.

## Optional items not used

- XGBoost: sklearn histogram gradient boosting is used, the same model family, with no extra dependency.
- SHAP: grouped permutation importance and partial dependence instead, both allowed by guide §12.
- Folium/Plotly maps: static matplotlib map instead.
