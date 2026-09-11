# MarketLens

## Indian Stock Market Analytics & Prediction Dashboard

🌐 **Live Demo:** https://marketlens-krm1.onrender.com/

🐙 **GitHub Repository:** https://github.com/amritakhare4729-creator/MarketLens

MarketLens is an end-to-end stock market analytics project that analyzes historical Indian stock market data, engineers technical indicators, evaluates machine-learning models, and presents the results through an interactive Django dashboard.

The project analyzes **19 major Indian companies across 8 sectors** using approximately five years of historical market data.

> **Note:** MarketLens is designed as an analytics and decision-support system. Its machine-learning experiments do not provide reliable guaranteed next-day predictions.

---

## 📊 Project Highlights

- 📈 Historical stock market analysis
- 🏢 Analysis of 19 Indian companies
- 🏦 Coverage across 8 market sectors
- 🔍 Exploratory Data Analysis (EDA)
- 🧮 Technical indicator engineering
- 📊 Company and sector performance analysis
- 🤖 Multiple machine-learning experiments
- 📉 Next-day return prediction
- ⬆️⬇️ UP/DOWN movement classification
- 💡 Rule-based technical market signals
- 🌐 Interactive Django web dashboard
- 🚀 Deployed online using Render

---

## 🌐 Live Dashboard

### [🚀 Open MarketLens Live](https://marketlens-krm1.onrender.com/)

The deployed dashboard contains four major sections:

### 1. Dashboard

Provides an overview of:

- Market signals
- Company trends
- Signal distribution
- Market statistics
- Latest company prices

### 2. Company Analysis

Allows users to explore individual companies and view:

- Technical signal
- Trend
- Signal strength
- Historical price information
- Technical indicators

### 3. Sector Analysis

Provides sector-level analysis across:

- Automobile
- Banking
- Energy
- FMCG
- IT
- Metals
- Pharma
- Telecom

### 4. Prediction & Models

Displays:

- Regression model results
- Classification results
- Model comparison
- Baseline comparison
- Evaluation metrics

---

## 🎯 Objectives

The main objectives of MarketLens are to:

1. Analyze historical stock market performance.
2. Compare companies across different sectors.
3. Identify trends and volatility patterns.
4. Engineer meaningful technical indicators.
5. Evaluate machine-learning approaches for short-term market analysis.
6. Compare machine-learning models against simple baselines.
7. Generate technical market signals.
8. Present analytical insights through an interactive web dashboard.

---

## 📁 Dataset

MarketLens analyzes historical data for **19 Indian companies**.

### Companies

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

## 🧮 Feature Engineering

MarketLens contains multiple stages of feature engineering.

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

## 📊 Exploratory Data Analysis

MarketLens performs company-level and sector-level analysis.

The analysis includes:

- Stock performance comparison
- Historical price trends
- Sector performance
- Volatility comparison
- Trading volume analysis
- Moving-average analysis
- Correlation analysis

### Historical Performance Highlights

During the analyzed period:

- **M&M:** approximately **+320%**
- **Bharti Airtel:** approximately **+189%**
- **NTPC:** approximately **+183%**
- **Sun Pharma:** approximately **+143%**
- **Wipro:** approximately **-46%**
- **TCS:** approximately **-40%**
- **Infosys:** approximately **-34%**

M&M recorded the strongest start-to-end price performance, while Wipro recorded the weakest.

These figures represent historical performance and should not be interpreted as future return expectations.

---

# 🤖 Machine Learning

MarketLens evaluates multiple machine-learning approaches for short-term market analysis.

## 1. Random Forest — Next-Day Close Price

The initial regression model attempted to predict the next trading day's closing price.

| Metric | Random Forest | Baseline |
|---|---:|---:|
| MAE | 36.7940 | **21.3812** |
| RMSE | 120.1711 | **53.2567** |
| R² | 0.9985 | **0.9997** |

The simple baseline performed better than the Random Forest model.

This demonstrates why a high R² alone can be misleading when predicting highly autocorrelated stock prices.

---

## 2. Random Forest — Next-Day Return

The next model attempted to predict the next trading day's percentage return.

| Metric | Random Forest | Baseline |
|---|---:|---:|
| MAE | 1.0998% | **1.0647%** |
| RMSE | 1.5423% | **1.4977%** |
| R² | -0.0610 | **-0.0005** |

The Random Forest model did not outperform the baseline.

---

## 3. Improved Random Forest

Additional technical indicators were introduced to improve the return prediction model.

| Metric | Improved Random Forest | Baseline |
|---|---:|---:|
| MAE | 1.0788% | **1.0566%** |
| RMSE | 1.5115% | **1.4862%** |
| R² | -0.0348 | **-0.0004** |
| Directional Accuracy | 49.70% | — |

The additional features slightly improved the Random Forest compared with the earlier return model, but the model still did not outperform the baseline.

---

## 4. Random Forest Classification

The project also treats next-day movement as a binary classification problem:

- `1` = UP
- `0` = DOWN

| Metric | Result |
|---|---:|
| Accuracy | 48.77% |
| Precision | 46.68% |
| Recall | 46.79% |
| F1 Score | 46.73% |
| ROC-AUC | 0.4900 |

Baseline accuracy was approximately **48.03%**.

---

## 5. Gradient Boosting

| Metric | Result |
|---|---:|
| Accuracy | 49.23% |
| Precision | 47.93% |
| Recall | 65.88% |
| F1 Score | 55.49% |
| ROC-AUC | 0.4922 |

---

## 6. Logistic Regression

| Metric | Result |
|---|---:|
| Accuracy | 49.23% |
| Precision | 47.90% |
| Recall | 64.82% |
| F1 Score | 55.09% |
| ROC-AUC | 0.4997 |

---

## 🔎 Key Machine-Learning Finding

The experiments show that the evaluated models **do not provide a strong or reliable next-day predictive advantage over simple baselines**.

This is an important analytical finding rather than a failure.

Financial markets are noisy, non-stationary, and difficult to predict using historical technical indicators alone.

Therefore, MarketLens is positioned as an:

> **Analytics, trend-analysis, and decision-support dashboard rather than a guaranteed stock prediction system.**

---

# 💡 Technical Market Signals

MarketLens generates rule-based technical signals using:

- Price vs 20-Day Moving Average
- Price vs 50-Day Moving Average
- RSI
- MACD
- 5-Day Return
- Daily Return

Signals are categorized as:

- 🟢 BUY / POSITIVE
- 🟡 HOLD / NEUTRAL
- 🔴 SELL / NEGATIVE

Signal strength is categorized as:

- Weak
- Moderate
- Strong

The dashboard also identifies:

- Bullish trends
- Bearish trends
- Mixed trends
- Low volatility
- Moderate volatility
- High volatility

These signals are analytical indicators and **are not financial advice**.

---

# 🛠️ Tech Stack

### Programming & Analysis

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- yfinance

### Web Development

- Django
- HTML
- CSS

### Database

- SQLite

### Deployment

- Git
- GitHub
- Render
- Gunicorn
- WhiteNoise

---

# 📂 Project Structure

```text
MarketLens/
│
├── Dashboard/
│   ├── migrations/
│   ├── static/
│   │   └── dashboard/
│   │       ├── marketlens-favicon.svg
│   │       ├── marketlens-icon.svg
│   │       └── marketlens-logo.svg
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
├── build.sh
├── .gitignore
└── README.md