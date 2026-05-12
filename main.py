"""
Credit Card Fraud Detection & AML Analytics
Author: Jimmy Le-Nguyen
GitHub: https://github.com/jimmyle9080

Dataset: Kaggle Credit Card Fraud Detection (ULB Machine Learning Group)
284,807 transactions | 492 fraud cases | 0.173% fraud rate

How to run:
    1. pip install -r requirements.txt
    2. Open this folder in VS Code
    3. Run main.py (press the play button)

Outputs:
    - outputs/exports/ : CSV files ready for Power BI
    - outputs/charts/  : 6 professional visualizations
"""

import os
import sqlite3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              roc_auc_score, average_precision_score,
                              precision_recall_curve, roc_curve)
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import warnings
warnings.filterwarnings('ignore')

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(ROOT, "data", "creditcard.csv")
DB_PATH    = os.path.join(ROOT, "data", "fraud.db")
EXPORT_DIR = os.path.join(ROOT, "outputs", "exports")
CHART_DIR  = os.path.join(ROOT, "outputs", "charts")

os.makedirs(EXPORT_DIR, exist_ok=True)
os.makedirs(CHART_DIR,  exist_ok=True)

FRAUD_COLOR  = "#E63946"
LEGIT_COLOR  = "#457B9D"
ACCENT_COLOR = "#2A9D8F"
BG_COLOR     = "#F8F9FA"

# ── Step 1: Load & Clean ───────────────────────────────────────────────────────
print("=" * 60)
print("  CREDIT CARD FRAUD DETECTION & AML ANALYTICS")
print("  Author: Jimmy Le-Nguyen")
print("=" * 60)

print("\n[1/6] Loading and cleaning data...")
df = pd.read_csv(DATA_PATH)

# Feature engineering
df['Hour']           = (df['Time'] // 3600) % 24
df['AmountLog']      = np.log1p(df['Amount'])
df['AmountBucket']   = pd.cut(df['Amount'],
                               bins=[0,10,50,200,500,2000,df['Amount'].max()+1],
                               labels=['<$10','$10-50','$50-200','$200-500','$500-2K','>$2K'])
df['IsFraud']        = df['Class'].map({0: 'Legitimate', 1: 'Fraud'})
df['RiskScore']      = (df[[f'V{i}' for i in [4,11,12,14,17]]].abs().mean(axis=1) * 10).round(2)

# Data quality check
null_count  = df.isnull().sum().sum()
dupe_count  = df.duplicated().sum()
total_tx    = len(df)
fraud_count = df['Class'].sum()
fraud_rate  = fraud_count / total_tx * 100

print(f"   Transactions loaded : {total_tx:,}")
print(f"   Fraud cases         : {fraud_count:,}  ({fraud_rate:.3f}%)")
print(f"   Null values         : {null_count}")
print(f"   Duplicate rows      : {dupe_count}")
print(f"   Class imbalance     : {int((total_tx-fraud_count)/fraud_count)}:1 (legit:fraud)")

# ── Step 2: Load into SQLite ───────────────────────────────────────────────────
print("\n[2/6] Loading into SQLite database...")
conn = sqlite3.connect(DB_PATH)
df.to_sql("transactions", conn, if_exists="replace", index=False)

# SQL Analysis queries
queries = {
    "fraud_by_hour": """
        SELECT Hour,
               COUNT(*) AS total_transactions,
               SUM(Class) AS fraud_count,
               ROUND(SUM(Class)*100.0/COUNT(*), 3) AS fraud_rate_pct,
               ROUND(AVG(Amount), 2) AS avg_amount
        FROM transactions
        GROUP BY Hour
        ORDER BY Hour
    """,
    "fraud_by_amount_bucket": """
        SELECT AmountBucket,
               COUNT(*) AS total_transactions,
               SUM(Class) AS fraud_count,
               ROUND(SUM(Class)*100.0/COUNT(*), 3) AS fraud_rate_pct,
               ROUND(AVG(Amount), 2) AS avg_amount,
               ROUND(MAX(Amount), 2) AS max_amount
        FROM transactions
        GROUP BY AmountBucket
        ORDER BY avg_amount
    """,
    "top_fraud_transactions": """
        SELECT Time, Amount, RiskScore, Hour,
               V1, V2, V3, V4, V14, V17
        FROM transactions
        WHERE Class = 1
        ORDER BY Amount DESC
        LIMIT 50
    """,
    "hourly_risk_summary": """
        SELECT Hour,
               COUNT(*) AS total_tx,
               SUM(Class) AS fraud_count,
               ROUND(AVG(RiskScore), 2) AS avg_risk_score,
               ROUND(AVG(Amount), 2) AS avg_amount,
               ROUND(SUM(CASE WHEN Class=1 THEN Amount ELSE 0 END), 2) AS fraud_dollar_loss
        FROM transactions
        GROUP BY Hour
        ORDER BY fraud_count DESC
    """,
    "anomaly_detection": """
        SELECT Time, Amount, RiskScore, Class, Hour,
               CASE
                   WHEN RiskScore > 15 AND Amount > 100 THEN 'High Risk'
                   WHEN RiskScore > 10 OR Amount > 500   THEN 'Medium Risk'
                   ELSE 'Low Risk'
               END AS risk_tier
        FROM transactions
        ORDER BY RiskScore DESC
        LIMIT 500
    """,
    "summary_stats": """
        SELECT
            COUNT(*) AS total_transactions,
            SUM(Class) AS total_fraud,
            ROUND(SUM(Class)*100.0/COUNT(*), 3) AS fraud_rate_pct,
            ROUND(SUM(CASE WHEN Class=1 THEN Amount ELSE 0 END), 2) AS total_fraud_dollars,
            ROUND(AVG(CASE WHEN Class=1 THEN Amount END), 2) AS avg_fraud_amount,
            ROUND(AVG(CASE WHEN Class=0 THEN Amount END), 2) AS avg_legit_amount,
            ROUND(MAX(Amount), 2) AS max_transaction,
            ROUND(AVG(RiskScore), 2) AS avg_risk_score
        FROM transactions
    """
}

results = {}
for name, query in queries.items():
    results[name] = pd.read_sql_query(query, conn)
    export_path = os.path.join(EXPORT_DIR, f"{name}.csv")
    results[name].to_csv(export_path, index=False)

conn.close()

# Print summary stats
stats = results["summary_stats"].iloc[0]
print(f"   Total fraud dollars  : ${stats['total_fraud_dollars']:,.2f}")
print(f"   Avg fraud amount     : ${stats['avg_fraud_amount']:,.2f}")
print(f"   Avg legit amount     : ${stats['avg_legit_amount']:,.2f}")
print(f"   Max transaction      : ${stats['max_transaction']:,.2f}")

# ── Step 3: Machine Learning ───────────────────────────────────────────────────
print("\n[3/6] Building fraud detection models...")

features = [f'V{i}' for i in range(1, 29)] + ['Amount', 'Hour', 'AmountLog']
X = df[features]
y = df['Class']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Handle class imbalance with SMOTE-style oversampling
X_tr_fraud = X_train[y_train == 1]
y_tr_fraud = y_train[y_train == 1]
X_tr_legit = X_train[y_train == 0]
y_tr_legit = y_train[y_train == 0]

X_fraud_up, y_fraud_up = resample(X_tr_fraud, y_tr_fraud,
                                   replace=True, n_samples=len(X_tr_legit)//10,
                                   random_state=42)
X_balanced = pd.concat([X_tr_legit, X_fraud_up])
y_balanced = pd.concat([y_tr_legit, y_fraud_up])

scaler   = StandardScaler()
X_tr_sc  = scaler.fit_transform(X_balanced)
X_te_sc  = scaler.transform(X_test)

# Logistic Regression baseline
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
lr.fit(X_tr_sc, y_balanced)
lr_probs = lr.predict_proba(X_te_sc)[:, 1]
lr_preds = lr.predict(X_te_sc)

# Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced', n_jobs=-1)
rf.fit(X_balanced, y_balanced)
rf_probs = rf.predict_proba(X_test)[:, 1]
rf_preds = rf.predict(X_test)

# Gradient Boosting
gb = GradientBoostingClassifier(n_estimators=100, random_state=42, learning_rate=0.1)
gb.fit(X_balanced, y_balanced)
gb_probs = gb.predict_proba(X_test)[:, 1]
gb_preds = gb.predict(X_test)

models = {
    'Logistic Regression': (lr_preds, lr_probs),
    'Random Forest':       (rf_preds, rf_probs),
    'Gradient Boosting':   (gb_preds, gb_probs)
}

model_results = []
print(f"\n   {'Model':<25} {'AUC-ROC':>8} {'AUPRC':>8} {'Precision':>10} {'Recall':>8} {'F1':>6}")
print("   " + "-"*65)
for name, (preds, probs) in models.items():
    auc   = roc_auc_score(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    report = classification_report(y_test, preds, output_dict=True)
    fraud_metrics = report.get('1', {})
    prec = fraud_metrics.get('precision', 0)
    rec  = fraud_metrics.get('recall', 0)
    f1   = fraud_metrics.get('f1-score', 0)
    print(f"   {name:<25} {auc:>8.4f} {auprc:>8.4f} {prec:>10.4f} {rec:>8.4f} {f1:>6.4f}")
    model_results.append({'Model': name, 'AUC_ROC': auc, 'AUPRC': auprc,
                          'Precision': prec, 'Recall': rec, 'F1': f1})

model_df = pd.DataFrame(model_results)
model_df.to_csv(os.path.join(EXPORT_DIR, "model_comparison.csv"), index=False)

# Feature importance from RF
feat_imp = pd.DataFrame({'Feature': features, 'Importance': rf.feature_importances_})
feat_imp = feat_imp.sort_values('Importance', ascending=False).head(15)
feat_imp.to_csv(os.path.join(EXPORT_DIR, "feature_importance.csv"), index=False)

# ── Step 4: Visualizations ─────────────────────────────────────────────────────
print("\n[4/6] Generating visualizations...")
plt.rcParams.update({'font.family': 'DejaVu Sans', 'figure.facecolor': BG_COLOR,
                     'axes.facecolor': 'white', 'axes.spines.top': False,
                     'axes.spines.right': False})

# Chart 1: Class Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)
fig.suptitle('Credit Card Transaction Class Distribution', fontsize=14, fontweight='bold', y=1.02)

counts = df['Class'].value_counts()
bars = axes[0].bar(['Legitimate', 'Fraud'], [counts[0], counts[1]],
                    color=[LEGIT_COLOR, FRAUD_COLOR], width=0.5, edgecolor='white')
axes[0].set_title('Transaction Count by Class')
axes[0].set_ylabel('Count')
for bar, count in zip(bars, [counts[0], counts[1]]):
    axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1000,
                 f'{count:,}', ha='center', fontweight='bold')

axes[1].pie([counts[0], counts[1]], labels=['Legitimate\n99.83%', 'Fraud\n0.17%'],
             colors=[LEGIT_COLOR, FRAUD_COLOR], autopct='', startangle=90,
             wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Fraud Rate (Highly Imbalanced Dataset)')

plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, '1_class_distribution.png'), dpi=150, bbox_inches='tight')
plt.close()

# Chart 2: Fraud by Hour of Day
fig, ax = plt.subplots(figsize=(14, 5), facecolor=BG_COLOR)
hourly = results['fraud_by_hour']
ax2 = ax.twinx()
bars = ax.bar(hourly['Hour'], hourly['fraud_count'], color=FRAUD_COLOR, alpha=0.7, label='Fraud Count')
line = ax2.plot(hourly['Hour'], hourly['fraud_rate_pct'], color=ACCENT_COLOR,
                marker='o', linewidth=2, markersize=4, label='Fraud Rate %')
ax.set_xlabel('Hour of Day')
ax.set_ylabel('Fraud Count', color=FRAUD_COLOR)
ax2.set_ylabel('Fraud Rate (%)', color=ACCENT_COLOR)
ax.set_title('Fraud Activity by Hour of Day', fontsize=13, fontweight='bold')
ax.set_xticks(range(0, 24))
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, '2_fraud_by_hour.png'), dpi=150, bbox_inches='tight')
plt.close()

# Chart 3: Transaction Amount Distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)
fig.suptitle('Transaction Amount Analysis', fontsize=14, fontweight='bold')

fraud_amounts = df[df['Class'] == 1]['Amount']
legit_amounts = df[df['Class'] == 0]['Amount'].sample(5000, random_state=42)

axes[0].hist(legit_amounts, bins=50, color=LEGIT_COLOR, alpha=0.7, label='Legitimate', density=True)
axes[0].hist(fraud_amounts, bins=50, color=FRAUD_COLOR, alpha=0.7, label='Fraud', density=True)
axes[0].set_xlabel('Transaction Amount ($)')
axes[0].set_ylabel('Density')
axes[0].set_title('Amount Distribution (Log Scale)')
axes[0].set_xscale('log')
axes[0].legend()

bucket_data = results['fraud_by_amount_bucket'].dropna(subset=['AmountBucket'])
bucket_data['AmountBucket'] = bucket_data['AmountBucket'].astype(str)
axes[1].barh(bucket_data['AmountBucket'], bucket_data['fraud_rate_pct'],
              color=FRAUD_COLOR, alpha=0.8, edgecolor='white')
axes[1].set_xlabel('Fraud Rate (%)')
axes[1].set_title('Fraud Rate by Amount Bucket')
for i, v in enumerate(bucket_data['fraud_rate_pct']):
    axes[1].text(v + 0.005, i, f'{v:.3f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, '3_amount_analysis.png'), dpi=150, bbox_inches='tight')
plt.close()

# Chart 4: Model Comparison
fig, axes = plt.subplots(1, 3, figsize=(15, 5), facecolor=BG_COLOR)
fig.suptitle('Fraud Detection Model Performance Comparison', fontsize=14, fontweight='bold')

metrics = ['AUC_ROC', 'AUPRC', 'F1']
titles  = ['AUC-ROC Score', 'AUPRC Score', 'F1 Score (Fraud Class)']
colors  = [LEGIT_COLOR, ACCENT_COLOR, FRAUD_COLOR]

for ax, metric, title, color in zip(axes, metrics, titles, colors):
    bars = ax.bar(model_df['Model'], model_df[metric], color=color, alpha=0.85, edgecolor='white')
    ax.set_title(title, fontweight='bold')
    ax.set_ylim(0, 1.1)
    ax.set_xticklabels(model_df['Model'], rotation=15, ha='right', fontsize=8)
    for bar, val in zip(bars, model_df[metric]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, '4_model_comparison.png'), dpi=150, bbox_inches='tight')
plt.close()

# Chart 5: Feature Importance
fig, ax = plt.subplots(figsize=(12, 6), facecolor=BG_COLOR)
colors_feat = [FRAUD_COLOR if 'V' in f else ACCENT_COLOR for f in feat_imp['Feature']]
bars = ax.barh(feat_imp['Feature'], feat_imp['Importance'], color=colors_feat, edgecolor='white')
ax.set_xlabel('Feature Importance Score')
ax.set_title('Top 15 Most Predictive Features (Random Forest)', fontsize=13, fontweight='bold')
ax.invert_yaxis()
red_patch   = mpatches.Patch(color=FRAUD_COLOR,  label='PCA Feature (V1-V28)')
green_patch = mpatches.Patch(color=ACCENT_COLOR, label='Engineered Feature')
ax.legend(handles=[red_patch, green_patch])
plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, '5_feature_importance.png'), dpi=150, bbox_inches='tight')
plt.close()

# Chart 6: Precision-Recall Curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor=BG_COLOR)
fig.suptitle('Model Evaluation Curves', fontsize=14, fontweight='bold')

model_colors = [LEGIT_COLOR, ACCENT_COLOR, FRAUD_COLOR]
for (name, (preds, probs)), color in zip(models.items(), model_colors):
    prec_vals, rec_vals, _ = precision_recall_curve(y_test, probs)
    auprc = average_precision_score(y_test, probs)
    axes[0].plot(rec_vals, prec_vals, color=color, linewidth=2,
                 label=f'{name} (AUPRC={auprc:.4f})')
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc_val = roc_auc_score(y_test, probs)
    axes[1].plot(fpr, tpr, color=color, linewidth=2,
                 label=f'{name} (AUC={auc_val:.4f})')

axes[0].set_xlabel('Recall')
axes[0].set_ylabel('Precision')
axes[0].set_title('Precision-Recall Curve')
axes[0].legend(fontsize=8)
axes[0].axhline(y=fraud_rate/100, color='gray', linestyle='--', alpha=0.5, label='Baseline')

axes[1].plot([0,1], [0,1], 'k--', alpha=0.5, label='Random Classifier')
axes[1].set_xlabel('False Positive Rate')
axes[1].set_ylabel('True Positive Rate')
axes[1].set_title('ROC Curve')
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(CHART_DIR, '6_evaluation_curves.png'), dpi=150, bbox_inches='tight')
plt.close()

# ── Step 5: Export Final Scored Dataset ───────────────────────────────────────
print("\n[5/6] Exporting Power BI ready datasets...")
df_export = df[['Time', 'Hour', 'Amount', 'AmountBucket', 'Class', 'IsFraud', 'RiskScore']].copy()
df_export['RF_FraudProbability']  = rf.predict_proba(X)[:, 1]
df_export['GB_FraudProbability']  = gb.predict_proba(X)[:, 1]
df_export['EnsembleProbability']  = (df_export['RF_FraudProbability'] + df_export['GB_FraudProbability']) / 2
df_export['ModelFlag'] = (df_export['EnsembleProbability'] > 0.5).astype(int)
df_export['AlertTier'] = pd.cut(df_export['EnsembleProbability'],
                                  bins=[0, 0.3, 0.6, 0.8, 1.0],
                                  labels=['Low', 'Medium', 'High', 'Critical'])
df_export.to_csv(os.path.join(EXPORT_DIR, "scored_transactions.csv"), index=False)

alerts = df_export[df_export['ModelFlag'] == 1].sort_values('EnsembleProbability', ascending=False)
alerts.to_csv(os.path.join(EXPORT_DIR, "flagged_transactions.csv"), index=False)

# ── Step 6: Summary ───────────────────────────────────────────────────────────
print("\n[6/6] Done! Summary of outputs:")
print(f"\n   CHARTS (outputs/charts/):")
print(f"   1_class_distribution.png   - Fraud vs legitimate transaction counts")
print(f"   2_fraud_by_hour.png        - Hourly fraud activity pattern")
print(f"   3_amount_analysis.png      - Amount distribution & fraud rate by bucket")
print(f"   4_model_comparison.png     - AUC, AUPRC, F1 across 3 models")
print(f"   5_feature_importance.png   - Top 15 predictive features")
print(f"   6_evaluation_curves.png    - Precision-Recall & ROC curves")
print(f"\n   EXPORTS (outputs/exports/):")
print(f"   scored_transactions.csv    - All 284,807 transactions with fraud scores")
print(f"   flagged_transactions.csv   - {len(alerts):,} flagged high-risk transactions")
print(f"   fraud_by_hour.csv          - Hourly fraud analysis for Power BI")
print(f"   fraud_by_amount_bucket.csv - Amount tier breakdown")
print(f"   model_comparison.csv       - Model performance metrics")
print(f"   feature_importance.csv     - Top predictive features")
print(f"   anomaly_detection.csv      - Risk-tiered transaction sample")
print(f"   summary_stats.csv          - Executive summary statistics")

best_model = model_df.loc[model_df['AUC_ROC'].idxmax(), 'Model']
best_auc   = model_df['AUC_ROC'].max()
best_auprc = model_df.loc[model_df['AUC_ROC'].idxmax(), 'AUPRC']
print(f"\n   BEST MODEL: {best_model}")
print(f"   AUC-ROC : {best_auc:.4f}")
print(f"   AUPRC   : {best_auprc:.4f}")
print(f"\n   Total fraud exposure detected : ${stats['total_fraud_dollars']:,.2f}")
print(f"   Flagged transactions          : {len(alerts):,}")
print("\n" + "=" * 60)
print("  Connect outputs/exports/ to Power BI for live dashboard")
print("=" * 60)
