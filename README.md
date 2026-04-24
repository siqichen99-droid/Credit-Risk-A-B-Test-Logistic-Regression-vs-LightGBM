# Credit Risk A/B Test: Logistic Regression vs LightGBM

> An end-to-end machine learning A/B testing framework for credit default prediction — comparing a traditional Logistic Regression scorecard against a LightGBM gradient boosting model on 300K+ real loan applicants. Validated with production-grade statistical methods and translated into quantified business impact.

**Author:** Siqi Chen | [LinkedIn](https://www.linkedin.com/in/siqi-chen-3159431b6) | siqichen99@gmail.com

---

## Results at a Glance

| Metric | Model A — Logistic Regression | Model B — LightGBM | Winner |
|---|---|---|---|
| AUC-ROC | 0.6992 | 0.7378 | ✅ LightGBM (+5.52%) |
| Gini Coefficient | 0.3985 | 0.4757 | ✅ LightGBM |
| KS Statistic | 0.2962 | 0.3547 | ✅ LightGBM |
| Recall | 0.8479 | 0.8530 | ✅ LightGBM |
| Precision | 0.1071 | 0.1204 | ✅ LightGBM |
| Optimal Threshold | 0.261 | 0.168 | — |
| Estimated Cost | $37.66M | $34.15M | ✅ LightGBM (-$3.51M) |
| DeLong Z-stat | — | 18.40 | ✅ Significant |
| P-value | — | p < 0.001 | ✅ Reject H₀ |
| Cohen's d | — | 12.85 (Large) | ✅ Large effect |

**Bottom line:** LightGBM at threshold 0.168 reduces estimated credit losses by **$3.51M on the test set (~$11.7M annualized)**, catches 38 more defaults, and approves 6,265 more creditworthy borrowers — validated with the DeLong test (p < 0.001) and 1,000-iteration bootstrap confidence intervals.

---

## Project Structure

```
credit_risk_ab_test/
│
├── notebooks/
│   ├── 01_eda.ipynb                  # Data loading, EDA, feature engineering, SMOTE
│   ├── 02_models.ipynb               # Model training, SHAP, initial evaluation
│   ├── 03_ab_testing.ipynb           # DeLong test, bootstrap CI, threshold optimization
│   └── 04_business_dashboard.ipynb   # Business impact, executive summary, visuals
│
├── outputs/
│   ├── section3_dashboard.png        # Statistical A/B test results dashboard
│   ├── section4_portfolio_hero.png   # Full project summary (6-panel)
│   ├── section4_business_impact.png  # Waterfall + metrics comparison
│   └── section4_risk_segments.png    # Risk segment calibration chart
│
├── data/
│   └── (download from Kaggle — see instructions below)
│
├── models/
│   ├── model_a_logistic.pkl          # Trained Logistic Regression
│   ├── model_b_lightgbm.pkl          # Trained LightGBM
│   └── scaler.pkl                    # StandardScaler for Model A
│
├── results/
│   ├── phase2_summary.csv            # Section 2 metrics comparison
│   ├── section3_results.csv          # Full statistical test results
│   └── section4_impact.csv           # Annualized financial impact
│
└── README.md
```

---

## Methodology

### Section 1 — Data & Feature Engineering (`01_eda.ipynb`)

**Dataset:** [Home Credit Default Risk](https://www.kaggle.com/c/home-credit-default-risk) — 307K loan applicants, 120+ features, ~8% default rate.

**Feature engineering:** 9 domain-driven financial features created from raw columns:

| Feature | Formula | Financial rationale |
|---|---|---|
| `DEBT_TO_INCOME` | AMT_CREDIT / AMT_INCOME | Measures loan burden relative to earnings |
| `PAYMENT_TO_INCOME` | AMT_ANNUITY / AMT_INCOME | Annual repayment affordability |
| `LOAN_TO_VALUE` | AMT_CREDIT / AMT_GOODS_PRICE | Collateral coverage ratio |
| `CREDIT_TERM` | AMT_CREDIT / AMT_ANNUITY | Loan duration in months |
| `AGE_YEARS` | −DAYS_BIRTH / 365 | Applicant age |
| `EMPLOYED_YEARS` | −DAYS_EMPLOYED / 365 | Employment stability |
| `EMPLOYMENT_TO_AGE` | EMPLOYED_YEARS / AGE_YEARS | Career stability ratio |
| `EXT_SOURCE_MEAN` | Mean of 3 bureau scores | Combined external credit signal |
| `EXT_SOURCE_MIN` | Min of 3 bureau scores | Downside credit risk signal |

**Class imbalance:** Handled with SMOTE (sampling_strategy=0.3) on training set only. Test set preserved at the real-world 8% default rate.

**Split:** Stratified 70/30 train/test split (seed=42).

---

### Section 2 — Model Development (`02_models.ipynb`)

**Model A — Logistic Regression (Control)**
- Industry standard in credit risk; every coefficient is auditable
- StandardScaler applied; C=0.1 regularization; class_weight='balanced'
- Coefficients visualized as a credit scorecard

**Model B — LightGBM (Treatment)**
- Gradient boosting with 500 trees, early stopping (50 rounds), learning_rate=0.05
- No feature scaling required
- SHAP TreeExplainer used for feature-level explainability on 2,000-applicant sample

**Evaluation metrics:** AUC-ROC, Gini coefficient, KS statistic, average precision, F1, precision, recall, cost-sensitive confusion matrix at threshold=0.35.

**Section 2 finding:** At a fixed threshold of 0.35, Logistic Regression appeared $2.3M cheaper — not because LightGBM was worse, but because its threshold was uncalibrated. This was resolved in Section 3.

---

### Section 3 — A/B Test Statistical Framework (`03_ab_testing.ipynb`)

#### DeLong Test
Compares two correlated AUC-ROC curves on the same test population.

- **H₀:** AUC(Model A) = AUC(Model B)
- **H₁:** AUC(Model B) > AUC(Model A)
- **Result:** Z = 18.40, p < 0.001 → **Reject H₀**
- LightGBM's AUC advantage is statistically significant and not due to chance

> The extreme Z-statistic reflects both a genuinely large AUC difference (+0.0386) and a large test population (90K+ applicants) that gives the test high power to detect real differences.

#### Bootstrap Confidence Intervals
1,000 resamples with replacement to estimate AUC stability:

| Model | AUC | 95% CI |
|---|---|---|
| Model A — Logistic Regression | 0.6992 | Entirely below Model B |
| Model B — LightGBM | 0.7378 | Entirely above Model A |
| Difference (B − A) | +0.0386 | Entirely above zero |

The CI for the AUC difference is entirely above zero — confirming LightGBM consistently outperforms Logistic Regression across all resamples.

#### Effect Size — Cohen's d

| Metric | Value |
|---|---|
| Pooled std | 0.003005 |
| Cohen's d | **12.85** |
| Classification | **Large** |

Cohen's d of 12.85 (threshold for "large" is 0.8) confirms the AUC gap is not only statistically significant but practically enormous and stable.

#### Threshold Optimization
Swept 200 threshold values (0.01–0.99) and minimized total cost (FN × $10,000 + FP × $500):

| Model | Optimal Threshold | Minimum Cost |
|---|---|---|
| Model A — Logistic Regression | 0.261 | $37,662,500 |
| Model B — LightGBM | 0.168 | $34,150,000 |
| **Cost saving** | | **$3,512,500** |

At calibrated thresholds, LightGBM wins on every metric: lower cost, higher recall, higher precision, more defaults caught, more good borrowers approved.

---

### Section 4 — Business Impact (`04_business_dashboard.ipynb`)

#### Performance at Optimal Thresholds

| Metric | Model A (t=0.261) | Model B (t=0.168) | Δ |
|---|---|---|---|
| Defaults caught | 6,315 (84.8%) | 6,353 (85.3%) | +38 |
| Good loans approved | 32,141 | 38,406 | +6,265 |
| Good loans rejected | 52,665 | 46,400 | −6,265 |
| Defaults missed | 1,133 | 1,095 | −38 |
| Total cost | $37,662,500 | $34,150,000 | **−$3,512,500** |

#### Annualized Financial Impact

| Item | Amount |
|---|---|
| Test-set cost saving | $3,512,500 |
| Scale factor (test = 30%) | 3.33× |
| **Estimated annual saving** | **~$11,700,000** |
| Extra defaults caught per year | ~127 |
| Extra good borrowers approved per year | ~20,900 |

#### Cost Breakdown (where does the saving come from?)
- **Fewer missed defaults:** 38 fewer × $10,000 = $380,000 (test) | ~$1.27M/yr
- **Fewer wrongly rejected borrowers:** 6,265 fewer × $500 = $3,132,500 (test) | ~$10.4M/yr

#### Deployment Recommendation
Deploy **LightGBM at threshold 0.168**. It is statistically validated (DeLong p < 0.001), cost-optimized (threshold sweep), and operationally defensible (SHAP explainability for regulatory review). The Logistic Regression scorecard is retained as a benchmark and for SOX documentation purposes.

---

## How to Reproduce

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/credit_risk_ab_test.git
cd credit_risk_ab_test
```

### 2. Create virtual environment and install dependencies
```bash
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

pip install numpy pandas scikit-learn lightgbm imbalanced-learn shap \
            matplotlib seaborn scipy jupyter ipykernel joblib
```

### 3. Download the dataset
1. Go to https://www.kaggle.com/c/home-credit-default-risk/data
2. Download and unzip into the `data/` folder
3. The required file is `application_train.csv`

### 4. Run notebooks in order
```
01_eda.ipynb               → generates X_train, X_test, y_train, y_test CSVs
02_models.ipynb            → generates predictions.csv, model .pkl files
03_ab_testing.ipynb        → generates section3_results.csv, dashboard PNG
04_business_dashboard.ipynb → generates README assets, impact CSV
```

---

## Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data | pandas, numpy |
| ML Models | scikit-learn, LightGBM |
| Explainability | SHAP |
| Class Imbalance | imbalanced-learn (SMOTE) |
| Statistical Tests | scipy (DeLong, bootstrap) |
| Visualization | matplotlib, seaborn |
| Environment | Jupyter Notebook, VS Code |

---

## Key Skills Demonstrated

- **A/B testing design** — control/treatment assignment, metric definition, guardrail metrics
- **Statistical rigor** — DeLong test, bootstrap CI, Cohen's d, p-value interpretation
- **Credit risk domain knowledge** — Gini coefficient, KS statistic, scorecard methodology, SOX awareness
- **ML engineering** — feature engineering, SMOTE, early stopping, SHAP explainability
- **Business translation** — threshold optimization, cost-sensitive evaluation, annualized impact modeling
- **Production thinking** — model calibration, fairness audit recommendation, monitoring plan

---

## Resume Bullet

> *Developed end-to-end credit risk A/B testing framework comparing Logistic Regression (AUC 0.699, Gini 0.399) vs LightGBM (AUC 0.738, Gini 0.476) on 300K+ loan applicants; validated +5.5% AUC lift via DeLong test (Z=18.40, p<0.001, Cohen's d=12.85) and threshold optimization, projecting ~$11.7M annual reduction in credit losses; applied SHAP explainability to meet model interpretability requirements aligned with SOX compliance standards.*

---
