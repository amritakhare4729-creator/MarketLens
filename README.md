# MarketLens

## Indian Stock Market Analytics & Prediction Dashboard

MarketLens is an end-to-end stock market analytics project focused on analyzing historical Indian stock market data, engineering technical indicators, evaluating machine-learning models, and presenting the results through an interactive Django dashboard.

The project analyzes 19 major Indian companies across 8 sectors using approximately five years of historical market data.

---

## Project Overview

MarketLens combines:

- Historical stock market data collection
- Data cleaning and verification
- Exploratory Data Analysis (EDA)
- Technical feature engineering
- Advanced technical indicators
- Stock and sector performance analysis
- Machine-learning experiments
- UP/DOWN movement classification
- Next-day return prediction
- Technical trading signals
- Interactive Django dashboard

The project is designed as a **data analytics and decision-support system**, rather than a guaranteed stock-price prediction system.

---

## Objectives

The main objectives of MarketLens are to:

1. Analyze historical stock market performance.
2. Compare companies across different sectors.
3. Identify trends and volatility patterns.
4. Engineer meaningful technical indicators.
5. Evaluate machine-learning approaches for short-term market prediction.
6. Compare ML models against simple baselines.
7. Present analytical insights through an interactive web dashboard.

---

## Dataset

The project contains historical data for 19 Indian companies.

### Companies

- Axis Bank
- Bharti Airtel
- Dr Reddy's
- HCLTech
- HDFC Bank
- Hindustan Unilever
- ICICI Bank
- Infosys
- ITC
- M&M
- Maruti Suzuki
- NTPC
- ONGC
- Reliance
- SBI
- Sun Pharma
- Tata Steel
- TCS
- Wipro

### Sectors

| Sector | Companies |
|---|---|
| Automobile | M&M, Maruti Suzuki |
| Banking | Axis Bank, HDFC Bank, ICICI Bank, SBI |
| Energy | NTPC, ONGC, Reliance |
| FMCG | Hindustan Unilever, ITC |
| IT | HCLTech, Infosys, TCS, Wipro |
| Metals | Tata Steel |
| Pharma | Dr Reddy's, Sun Pharma |
| Telecom | Bharti Airtel |

### Data Coverage

- Approximately 5 years of historical data
- Date range: September 2021 – September 2026
- 23,560 rows in the primary feature dataset
- 19 companies
- 8 sectors

---

## Data Processing & Feature Engineering

The project includes multiple stages of feature engineering.

### Basic Features

- Daily Return
- Price Change
- 20-Day Moving Average
- 50-Day Moving Average
- 20-Day Volatility
- High-Low Range
- High-Low Range Percentage
- Cumulative Return

### Advanced Technical Features

- 1-Day Return
- 3-Day Return
- 5-Day Return
- 10-Day Return
- 5-Day Momentum
- 10-Day Momentum
- Price vs MA20
- Price vs MA50
- Volume Change
- 20-Day Volume Average
- Volume Ratio
- RSI 14
- MACD
- MACD Signal
- MACD Histogram
- 5-Day Volatility
- 10-Day Volatility
- Open-Close Change
- High-Low Percentage
- Next-Day Return

---

## Exploratory Data Analysis

MarketLens performs company-level and sector-level analysis.

The analysis includes:

- Stock performance comparison
- Historical price trends
- Sector performance
- Volatility comparison
- Trading volume analysis
- Moving-average analysis
- Correlation analysis

### Notable Historical Performance

Based on the analyzed period:

- **M&M** recorded the strongest start-to-end price performance at approximately **+320%**.
- **Wipro** recorded the weakest performance at approximately **-46%**.
- Bharti Airtel and NTPC also showed strong positive performance.
- Several IT companies experienced negative start-to-end price performance during the period.

These figures describe historical performance and should not be interpreted as future return expectations.

---

## Machine Learning

MarketLens evaluates multiple machine-learning approaches.

### 1. Random Forest Regression

The first regression model attempted to predict the next trading day's closing price.

Performance:

| Metric | Random Forest | Baseline |
|---|---:|---:|
| MAE | 36.7940 | 21.3812 |
| RMSE | 120.1711 | 53.2567 |
| R² | 0.9985 | 0.9997 |

The simple baseline performed better than the Random Forest model.

This demonstrates why a high R² alone can be misleading for highly autocorrelated stock-price data.

---

### 2. Next-Day Return Prediction

A Random Forest model was then used to predict next-day percentage returns.

Performance:

| Metric | Random Forest | Baseline |
|---|---:|---:|
| MAE | 1.0998% | 1.0647% |
| RMSE | 1.5423% | 1.4977% |
| R² | -0.0610 | -0.0005 |

The model did not outperform the baseline.

---

### 3. Improved Random Forest

Additional technical indicators were introduced to improve the return prediction model.

Performance:

| Metric | Improved Random Forest | Baseline |
|---|---:|---:|
| MAE | 1.0788% | 1.0566% |
| RMSE | 1.5115% | 1.4862% |
| R² | -0.0348 | -0.0004 |
| Directional Accuracy | 49.70% | — |

The additional features improved the Random Forest slightly compared with the earlier return model, but it still did not outperform the baseline.

---

### 4. UP/DOWN Classification

The project also treats next-day movement as a binary classification problem:

- `1` = UP
- `0` = DOWN

#### Random Forest Classifier

| Metric | Result |
|---|---:|
| Accuracy | 48.77% |
| Precision | 46.68% |
| Recall | 46.79% |
| F1 Score | 46.73% |
| ROC-AUC | 0.4900 |

Baseline accuracy was approximately 48.03%.

---

### 5. Gradient Boosting

| Metric | Result |
|---|---:|
| Accuracy | 49.23% |
| Precision | 47.93% |
| Recall | 65.88% |
| F1 Score | 55.49% |
| ROC-AUC | 0.4922 |

---

### 6. Logistic Regression

| Metric | Result |
|---|---:|
| Accuracy | 49.23% |
| Precision | 47.90% |
| Recall | 64.82% |
| F1 Score | 55.09% |
| ROC-AUC | 0.4997 |

---

## Key Machine-Learning Finding

The experiments show that the evaluated models do **not** provide a strong or reliable next-day predictive advantage over simple baselines.

This is an important result rather than a failure.

Financial markets are noisy, non-stationary, and difficult to predict using historical technical indicators alone.

Therefore, MarketLens is positioned as an:

> **Analytics, trend-analysis, and decision-support dashboard rather than a guaranteed stock prediction system.**

---

## Technical Market Signals

MarketLens generates rule-based technical signals using indicators such as:

- Price vs 20-Day Moving Average
- Price vs 50-Day Moving Average
- RSI
- MACD
- 5-Day Return
- Daily Return

Signals are categorized as:

- BUY / POSITIVE
- HOLD / NEUTRAL
- SELL / NEGATIVE

Signal strength is categorized as:

- Weak
- Moderate
- Strong

The dashboard also identifies:

- Bullish trends
- Bearish trends
- Mixed trends
- Low, Moderate, and High volatility

These signals are analytical indicators and are not financial advice.

---

## Dashboard

MarketLens includes a Django-based web dashboard with four main sections:

### Dashboard

Provides an overview of:

- Market signals
- Company trends
- Signal distribution
- Market statistics

### Company Analysis

Allows users to inspect individual companies and view:

- Current technical signal
- Trend
- Signal strength
- Historical price information
- Technical indicators

### Sector Analysis

Provides sector-level comparison across:

- Automobile
- Banking
- Energy
- FMCG
- IT
- Metals
- Pharma
- Telecom

### Prediction & Models

Displays:

- Regression model results
- Classification results
- Model comparison
- Baseline comparison
- Model evaluation metrics

---

## Project Structure

```text
MarketLens/
│
├── Dashboard/
│   ├── migrations/
│   ├── templates/
│   │   └── dashboard/
│   │       ├── index.html
│   │       ├── company.html
│   │       ├── sectors.html
│   │       └── prediction.html
│   ├── admin.py
│   ├── apps.py
│   ├── models.py
│   ├── tests.py
│   └── views.py
│
├── data/
│   ├── model_results/
│   ├── model_visualizations/
│   ├── visualizations/
│   ├── advanced_features.csv
│   ├── master_stock_data.csv
│   └── stock_features.csv
│
├── marketlens_web/
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── src/
│   ├── data_collection.py
│   ├── data_cleaning.py
│   ├── data_verification.py
│   ├── eda.py
│   ├── feature_engineering.py
│   ├── advanced_features.py
│   ├── visualization.py
│   ├── model_training.py
│   ├── model_evaluation.py
│   ├── return_model.py
│   ├── improved_return_model.py
│   ├── classification_model.py
│   ├── gradient_boosting_model.py
│   ├── logistic_regression_model.py
│   ├── company_model_analysis.py
│   ├── final_model_comparison.py
│   └── prediction_signal.py
│
├── manage.py
├── requirements.txt
├── .gitignore
└── README.md