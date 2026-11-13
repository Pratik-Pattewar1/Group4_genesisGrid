# 🚀 DeliveryIQ — AI-Powered Quick Commerce Delivery Intelligence

<p align="center">
  <strong>Transforming Delivery Data into Actionable Business Intelligence</strong>
</p>

<p align="center">
  <a href="https://deliveryiq-genesisgrid.streamlit.app/">
    <strong>🌐 LIVE APPLICATION — LAUNCH DELIVERYIQ</strong>
  </a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12-blue?logo=python" alt="Python 3.12"/>
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--learn-orange" alt="Machine Learning"/>
  <img src="https://img.shields.io/badge/Framework-Streamlit-red?logo=streamlit" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/Status-Deployed-success" alt="Deployed"/>
  <img src="https://img.shields.io/badge/Project-DeliveryIQ-purple" alt="DeliveryIQ"/>
</p>

---

## 📌 Project Overview

**DeliveryIQ** is an AI-powered E-commerce Delivery Intelligence platform developed for the Calibo Hackathon.

The platform transforms historical delivery records into actionable operational intelligence by combining machine learning, predictive analytics, geographic clustering and interactive visualizations.

DeliveryIQ helps businesses understand:

- **When** an order is likely to arrive through Estimated Time of Arrival (ETA) prediction.
- **How likely** an order is to experience a delay through late-delivery risk classification.
- **Where** delivery demand is concentrated through geographic clustering.
- **What** operational improvements businesses should implement through evidence-based recommendations.

The solution includes an interactive Streamlit application that allows users to explore delivery intelligence and generate predictions.

## 🌐 Live Application

**[Launch DeliveryIQ – Live Streamlit Application](https://deliveryiq-genesisgrid.streamlit.app/)**

The application provides an interactive interface for delivery ETA prediction, delay-risk estimation, model insights and business recommendations.

## 🏆 Project Information

| Attribute | Details |
|---|---|
| Project Name | DeliveryIQ |
| Team Name | Genesis Grid |
| Group Number | Group 4 |
| Competition | Calibo Hackathon |
| Domain | Artificial Intelligence and Machine Learning |
| Primary Language | Python |
| ML Framework | Scikit-learn |
| Application Framework | Streamlit |
| Deployment | Streamlit Community Cloud |
| Application Status | Deployed |

---

## 👥 Team Genesis Grid

| S. No. | Team Member |
|---|---|
| 1 | Manda Lohitha |
| 2 | Pratik Jankiram Pattewar |
| 3 | K Bonishira Akshaya |
| 4 | Yateesh Chandra Maram |
| 5 | Sanapalli Vikas |

---

## 🎯 Problem Statement

In modern quick-commerce and food-delivery operations, customers expect fast and accurate deliveries.

Unpredictable delivery times and delayed orders can lead to:

- Customer dissatisfaction.
- Order cancellations and refunds.
- Increased operational costs.
- Inefficient rider allocation.
- Reduced customer retention.

Traditional delivery systems often lack the predictive intelligence needed to identify potentially delayed orders before they become late.

DeliveryIQ addresses this challenge by using historical delivery data and machine learning to predict delivery outcomes and provide actionable insights to operational teams.

## 🎯 Project Objectives

1. **ETA Prediction:** Estimate delivery time in minutes using regression models.
2. **Late-Delivery Risk Prediction:** Identify orders with a high probability of exceeding the project-defined 30-minute benchmark.
3. **Geographic Analysis:** Discover delivery zones using K-Means clustering.
4. **Business Intelligence:** Generate three prioritised, evidence-backed operational recommendations.
5. **Interactive Application:** Provide an accessible Streamlit interface for delivery predictions and analytical insights.

---

## 🧠 Solution Architecture

DeliveryIQ follows a structured machine-learning workflow:

1. Data auditing and exploratory data analysis.
2. Data cleaning and quality validation.
3. Feature engineering and leakage prevention.
4. Model development, comparison and evaluation.
5. Business intelligence and explainability.
6. Streamlit application development, validation and deployment.

### Technology Stack

| Technology | Purpose |
|---|---|
| Python | Core development and machine learning |
| Pandas | Data manipulation and preprocessing |
| NumPy | Numerical computations |
| Scikit-learn | Machine learning and model evaluation |
| Streamlit | Interactive web application |
| Matplotlib | Data visualization |
| Seaborn | Exploratory data visualization |
| Jupyter Notebook | Experimentation and analysis |
| Joblib | Model persistence |
| Git and GitHub | Version control and project management |
| Streamlit Community Cloud | Application deployment |

---

## 📊 Dataset Overview

The project uses historical delivery data containing:

| Attribute | Details |
|---|---|
| Total Records | 45,593 |
| Original Features | 20 |
| Delivery Period | February 11, 2022 – April 6, 2022 |
| Geographic Coverage | 22 Indian city codes |
| Prediction Target | `Time_taken(min)` |
| Target Range | 10–54 minutes |
| Target Median | 26 minutes |

### Data Quality and Cleaning

A comprehensive dataset audit identified several data-quality issues.

| Data Quality Issue | Records | Treatment |
|---|---:|---|
| Rows containing missing values | 4,225 | Converted text NaN values into missing values |
| Duplicate rows | 0 | No duplicate removal required |
| Restaurant coordinates recorded as (0, 0) | 3,640 | Flagged as invalid |
| Sign-flipped restaurant coordinates | 431 | Corrected after four geographic checks |
| Placeholder rider records | 91 | Invalid age and rating values blanked |
| Missing order timestamps | 1,731 | Preserved without reconstructing from pickup time |
| Deleted records | 0 | All 45,593 records retained |

All original records were retained, and data-quality indicators were preserved for analysis.

### Feature Engineering

Feature engineering was performed to create meaningful predictive variables while preventing data leakage.

Important considerations:

- Order-placement information was used for the primary prediction models.
- Features unavailable when an order is placed were excluded from order-placement predictions.
- Missing values were handled using training-fitted preprocessing pipelines.
- Chronological splitting was used to evaluate future-period performance.

---

## 🤖 Machine Learning Models

DeliveryIQ implements three primary analytical tasks.

### 1. Delivery ETA Prediction

**Objective:** Predict the estimated delivery time in minutes.

Models compared:

- Mean and median baselines.
- Ridge Regression.
- Random Forest Regressor.
- Gradient Boosting Regressor.

**Final Model:** Gradient Boosting Regressor.

| Evaluation Metric | Result |
|---|---:|
| Mean Absolute Error (MAE) | 6.19 minutes |
| Root Mean Squared Error (RMSE) | 7.73 minutes |
| R² Score | 0.310 |
| Median Baseline MAE | 7.47 minutes |
| Improvement over Median Baseline | 17.1% |

### 2. Late-Delivery Risk Classification

**Objective:** Estimate the probability that an order takes more than 30 minutes.

Models compared:

- Prior-probability baseline.
- Logistic Regression.
- Gradient Boosting Classifier.

**Final Model:** Gradient Boosting Classifier.

| Evaluation Metric | Result |
|---|---:|
| ROC-AUC | 0.791 |
| PR-AUC | 0.608 |
| Recall | 0.813 |
| Precision | 0.467 |
| F1 Score | 0.593 |
| Selected Threshold | 0.268 |
| Baseline PR-AUC | 0.287 |

The classifier helps identify potentially delayed orders so that operational teams can prioritise interventions.

### 3. Geographic Delivery-Zone Clustering

**Objective:** Identify geographic patterns in delivery locations.

Algorithm: K-Means Clustering.

| Evaluation Metric | Result |
|---|---:|
| Selected Clusters | 22 |
| Silhouette Score | 0.948 |
| Geographic Interpretation | Clusters correspond to the 22 cities |

The clustering results identify geographic groupings in the dataset. They should not be interpreted as evidence of individual dark-store locations.

### Rider Segmentation

Rider-level segmentation was not considered feasible because the available `Delivery_person_ID` values behave as restaurant/slot codes rather than reliable individual rider identifiers.

No unsupported individual rider-performance conclusions were made.

---

## 💡 Key Business Insights and Recommendations

The analysis produced three prioritised operational recommendations.

### 1. Introduce Order-Placement Delay-Risk Alerts

Use the late-risk classification model to identify potentially delayed orders at the time of placement.

Recommended actions:

- Flag high-risk orders.
- Prioritise operational intervention.
- Communicate potential delays proactively.
- Monitor risk patterns across locations and order periods.

### 2. Implement a Two-Stage ETA Prediction Strategy

Use an initial order-placement ETA, followed by a refreshed prediction when additional information becomes available at rider assignment.

The comparison analysis achieved an MAE of 3.67 minutes without rider rating in the second-stage comparison.

This result is experimental and is not presented as a production-ready deployment result.

### 3. Improve Rider Capacity During Peak Hours

The analysis identified the 19:00–21:59 period as a significant delivery-risk window.

Approximately 50% of deliveries during these hours were late, compared with a network late-delivery rate of 29.8%.

Recommended actions:

- Increase rider availability during peak periods.
- Monitor order volume and delivery demand.
- Optimise operational capacity.
- Adjust customer delivery promises based on observed demand patterns.

---

## 📈 Model Evaluation Methodology

Model evaluation used a held-out chronological test period:

**March 30, 2022 – April 6, 2022**

The evaluation focused on order-placement features.

### Regression Evaluation

- Mean Absolute Error (MAE).
- Root Mean Squared Error (RMSE).
- R² Score.
- Comparison against a naive baseline.
- Segment-level error analysis.

### Classification Evaluation

- Precision.
- Recall.
- F1 Score.
- ROC-AUC.
- PR-AUC.
- Confusion matrix.
- Threshold-based evaluation.

Accuracy was not used as the sole classification evaluation metric.

### Clustering Evaluation

- Elbow method.
- Inertia.
- Silhouette score.

---

## 🖥️ Application Features

The DeliveryIQ Streamlit application provides an interactive interface for exploring delivery predictions and analytical results.

### 1. Delivery ETA Prediction

Users can provide order-level information such as:

- Store location by city.
- Customer location or delivery distance.
- Order date and time.
- Order type.
- City type.
- Festival indicator.

The application generates:

- Predicted delivery ETA.
- Expected delivery-time window.
- Delay-risk probability.
- Risk classification.
- Suggested operational action.
- Explanation of the prediction.

### 2. Delay-Risk Classification

The application identifies delivery-risk levels:

- Low Risk.
- Elevated Risk.
- High Risk.

The classification uses the selected alert threshold and training-period risk distribution.

### 3. Business Intelligence

Explore:

- Three prioritised business recommendations.
- Delivery trends.
- Model performance.
- Data-quality findings.
- Project limitations.

### 4. Model Performance

The application provides access to analytical information about the predictive models and their evaluation.

### 5. Input Validation

The application validates inputs and handles invalid geographic coordinates, distances and categorical values.

### Important Prediction Considerations

- Predictions use information available at order placement.
- Pickup time, rider age, rider rating, vehicle, multiple deliveries, weather and traffic are not requested as order-placement inputs.
- Unknown values are handled using model-training-fitted imputers.
- The delay threshold of more than 30 minutes is a project-defined benchmark, not an official delivery SLA.
- Predicted risk probabilities are uncalibrated.

---

## 🏗️ Project Structure

```text
Group4_GenesisGrid/
│
├── .streamlit/
│   └── config.toml
│
├── app/
│   └── streamlit_app.py
│
├── data/
│   ├── raw/
│   │   └── train.csv
│   │
│   └── processed/
│       ├── cleaned_data.csv
│       └── feature_engineered_data.csv
│
├── figures/
│   └── Analytical charts and visualizations
│
├── models/
│   ├── Saved machine learning models
│   └── model_card.json
│
├── notebooks/
│   └── 01_EDA.ipynb
│
├── presentation/
│   ├── DeliveryIQ_Final_Hackathon_Presentation.pptx
│   └── DeliveryIQ_Speaker_Script_and_QA.md
│
├── reports/
│   ├── business_insights_report.pdf
│   ├── business_report.pdf
│   ├── business_insights_report.md
│   ├── business_report.md
│   ├── business_recommendations.md
│   ├── challenge_compliance.md
│   ├── dataset_audit_report.md
│   ├── data_cleaning_report.md
│   ├── feature_dictionary.md
│   ├── app_validation_report.md
│   └── Other analytical reports
│
├── src/
│   ├── data_cleaning.py
│   ├── audit_cleaned_data.py
│   ├── investigate_missing_timestamps.py
│   ├── feature_engineering.py
│   ├── regression.py
│   ├── classification.py
│   ├── clustering.py
│   ├── evaluation.py
│   ├── modelling.py
│   ├── modelling_report.py
│   ├── business_insights.py
│   ├── insights_report.py
│   ├── challenge_gaps.py
│   ├── predict.py
│   ├── validate_app.py
│   └── build_pdfs.py
│
├── .gitignore
├── README.md
├── requirements.txt
└── run_app.bat
```

---

## ⚙️ Installation and Local Setup

### Prerequisites

- Python 3.12.
- Git.
- Visual Studio Code or another Python-compatible IDE.

### Step 1: Clone the Repository

```powershell
git clone https://github.com/Pratik-Pattewar1/Group4_genesisGrid.git

cd Group4_genesisGrid
```

### Step 2: Create a Virtual Environment

```powershell
py -3.12 -m venv .venv
```

### Step 3: Activate the Environment

```powershell
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use the virtual environment's Python executable directly.

### Step 4: Install Dependencies

```powershell
python -m pip install --upgrade pip

pip install -r requirements.txt
```

### Step 5: Run the Application

```powershell
python -m streamlit run app/streamlit_app.py
```

Open the following URL in your browser:

```text
http://localhost:8501
```

Stop the application using `Ctrl + C` in the terminal.

---

## ☁️ Deployment

DeliveryIQ is deployed using **Streamlit Community Cloud**.

**Live application:**

https://deliveryiq-genesisgrid.streamlit.app/

| Deployment Configuration | Value |
|---|---|
| Hosting Platform | Streamlit Community Cloud |
| Source Repository | Group4_genesisGrid |
| Branch | main |
| Application Entrypoint | `app/streamlit_app.py` |
| Dependency File | `requirements.txt` |
| Application URL | deliveryiq-genesisgrid.streamlit.app |

The application source code is maintained in GitHub. Changes pushed to the connected repository can trigger application updates through Streamlit Community Cloud.

---

## 📄 Project Reports and Deliverables

The repository includes the following project deliverables.

| Deliverable | Location |
|---|---|
| Final Business Insights Report | `reports/business_insights_report.pdf` |
| Final Business Report | `reports/business_report.pdf` |
| Business Insights Source | `reports/business_insights_report.md` |
| Business Report Source | `reports/business_report.md` |
| Business Recommendations | `reports/business_recommendations.md` |
| Challenge Compliance Report | `reports/challenge_compliance.md` |
| Dataset Audit Report | `reports/dataset_audit_report.md` |
| Data Cleaning Report | `reports/data_cleaning_report.md` |
| Feature Dictionary | `reports/feature_dictionary.md` |
| Application Validation Report | `reports/app_validation_report.md` |
| Final Presentation | `presentation/DeliveryIQ_Final_Hackathon_Presentation.pptx` |
| Speaker Script and Q&A | `presentation/DeliveryIQ_Speaker_Script_and_QA.md` |

---

## 🔬 Reproducibility and Validation

The project contains scripts for reproducing the data-processing and machine-learning workflow.

### Data Cleaning

```powershell
python -m src.data_cleaning
```

### Feature Engineering

```powershell
python -m src.feature_engineering
```

### Model Training and Evaluation

```powershell
python -m src.modelling
```

### Business Insights

```powershell
python -m src.business_insights
```

### Generate Final Reports

```powershell
python -m src.build_pdfs
```

### Application Validation

```powershell
python -m src.validate_app
```

Run these commands from the repository root with the required dependencies and input files available.

---

## ⚠️ Assumptions and Limitations

The results should be interpreted within the limitations of the available dataset and experimental setup.

1. **Project-defined late threshold:** A delivery time greater than 30 minutes is treated as late. This is not an official service-level agreement (SLA). Sensitivity checks were also performed at 35 and 40 minutes.

2. **Dataset domain:** The source dataset represents food delivery rather than actual grocery quick-commerce operations.

3. **Geographic assumptions:** Restaurants are used as proxies for dark stores where required by the project framing.

4. **Coordinate corrections:** Negative restaurant coordinates were corrected only after four geographic checks agreed. Invalid (0, 0) coordinates were not estimated.

5. **Rider identifiers:** Delivery person identifiers behave as restaurant/slot codes and cannot reliably identify individual riders.

6. **Weather and traffic:** Their recording times are unknown, so they were excluded from order-placement prediction models.

7. **Uncalibrated probabilities:** The classifier's risk probabilities are not calibrated production probabilities.

8. **Test period:** The held-out test period covers eight days, which limits the scope of generalisation.

9. **Synthetic patterns:** The source data contains synthetic patterns that may not fully represent real-world quick-commerce operations.

10. **Production readiness:** Experimental results should not be interpreted as guaranteed real-world performance.

---

## 🔮 Future Enhancements

Potential future improvements include:

- Integrating real-time traffic and weather information.
- Incorporating live rider-assignment data.
- Implementing continuous ETA updates.
- Improving probability calibration.
- Extending the dataset to real grocery quick-commerce operations.
- Adding real-time delivery monitoring.
- Introducing more advanced geographic and demand forecasting models.
- Evaluating model performance across larger chronological test periods.
- Integrating the predictions with operational order-management systems.

---

## 📌 Conclusion

DeliveryIQ demonstrates how machine learning and predictive analytics can transform historical delivery records into actionable operational intelligence.

By combining ETA prediction, late-delivery risk classification, geographic analysis and business recommendations, the platform provides a practical foundation for improving delivery visibility and operational decision-making.

The interactive Streamlit application makes these analytical capabilities accessible through a user-friendly interface.

**DeliveryIQ — Predict. Prevent. Deliver Better.**

---

<p align="center">
  <strong>Developed by Team Genesis Grid | Group 4</strong>
</p>

<p align="center">
  <a href="https://deliveryiq-genesisgrid.streamlit.app/">
    🌐 Explore DeliveryIQ
  </a>
  &nbsp; | &nbsp;
  <a href="https://github.com/Pratik-Pattewar1/Group4_genesisGrid">
    💻 GitHub Repository
  </a>
</p>
