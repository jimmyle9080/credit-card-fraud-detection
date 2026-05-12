# Credit Card Fraud Detection & AML Analytics

**Author:** Jimmy Le-Nguyen  
**GitHub:** [github.com/jimmyle9080](https://github.com/jimmyle9080)

## Overview

An end-to-end fraud detection and AML analytics pipeline built on the Kaggle Credit Card Fraud Detection dataset (ULB Machine Learning Group). The dataset contains **284,807 real transactions** from European cardholders in September 2013, with **492 fraud cases (0.173%)** — a severely imbalanced classification problem.

This project mirrors real-world AML and fraud detection work, covering the full analytics lifecycle from data ingestion and SQL analysis through machine learning model development, evaluation, and Power BI-ready reporting exports.

## Key Results

| Model | AUC-ROC | AUPRC | F1 (Fraud) |
|---|---|---|---|
| Logistic Regression | ~0.97 | ~0.73 | ~0.75 |
| Random Forest | ~0.98 | ~0.85 | ~0.84 |
| Gradient Boosting | ~0.98 | ~0.82 | ~0.81 |

- **Class imbalance handled** via oversampling and class weighting
- **AUPRC used as primary metric** (recommended over accuracy for imbalanced datasets)
- **284,807 transactions scored** with ensemble fraud probability
- **Power BI-ready exports** generated for executive dashboard

## Dataset

- **Source:** [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **284,807 transactions** | **492 fraud cases** | **0.173% fraud rate**
- Features V1-V28 are PCA-transformed (anonymized for confidentiality)
- Only `Time`, `Amount`, and `Class` are original features

> Download `creditcard.csv` from Kaggle and place it in the `data/` folder before running.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run everything (VS Code: just press the play button on main.py)
python main.py
```

Everything runs from `main.py` — no additional setup needed.

## Project Structure

```
credit_fraud_project/
├── data/
│   └── creditcard.csv          # Kaggle dataset (download separately)
├── outputs/
│   ├── charts/                 # 6 professional visualizations
│   └── exports/                # Power BI-ready CSV exports
├── main.py                     # Single entry point — run this
├── requirements.txt
└── README.md
```

## Outputs

### Charts (outputs/charts/)
| File | Description |
|---|---|
| 1_class_distribution.png | Fraud vs legitimate transaction counts |
| 2_fraud_by_hour.png | Hourly fraud activity pattern |
| 3_amount_analysis.png | Amount distribution & fraud rate by bucket |
| 4_model_comparison.png | AUC, AUPRC, F1 across 3 models |
| 5_feature_importance.png | Top 15 predictive features (Random Forest) |
| 6_evaluation_curves.png | Precision-Recall & ROC curves |

### Power BI Exports (outputs/exports/)
| File | Description |
|---|---|
| scored_transactions.csv | All 284,807 transactions with fraud probability scores |
| flagged_transactions.csv | High-risk flagged transactions for review |
| fraud_by_hour.csv | Hourly fraud analysis |
| fraud_by_amount_bucket.csv | Amount tier breakdown |
| model_comparison.csv | Model performance metrics |
| feature_importance.csv | Top predictive features |
| anomaly_detection.csv | Risk-tiered transaction sample |
| summary_stats.csv | Executive summary statistics |

## Technical Approach

### Class Imbalance Handling
The dataset has a 577:1 imbalance ratio (legitimate:fraud). This project addresses it through:
- **Oversampling** the minority (fraud) class in training data
- **Class weighting** in Logistic Regression
- **AUPRC as primary metric** (accuracy is misleading with extreme imbalance)
- **Stratified train/test split** to preserve fraud ratio

### Feature Engineering
Beyond the 28 PCA components, engineered features include:
- `Hour` — derived from transaction timestamp (fraud peaks at specific hours)
- `AmountLog` — log-transformed amount to reduce skewness
- `AmountBucket` — categorical amount tiers for segment analysis
- `RiskScore` — composite score from key PCA components

### Models
Three models trained and benchmarked:
1. **Logistic Regression** — interpretable baseline with class weighting
2. **Random Forest** — ensemble method, strong feature importance output
3. **Gradient Boosting** — sequential ensemble, high precision on fraud class

Final scoring uses an **ensemble probability** (average of RF + GB) for production flagging.

## Skills Demonstrated

- **SQL** — SQLite queries with window functions, aggregations, and risk tiering
- **Python** — Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Machine Learning** — Classification, ensemble methods, imbalanced data handling
- **Model Evaluation** — AUC-ROC, AUPRC, Precision-Recall, F1, confusion matrix
- **Feature Engineering** — Time-based features, log transforms, risk scoring
- **Data Storytelling** — 6 professional charts + Power BI-ready exports
- **Financial Domain** — AML, fraud detection, risk tiering, transaction monitoring

## License

Dataset licensed under [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/).  
Code: MIT License.
