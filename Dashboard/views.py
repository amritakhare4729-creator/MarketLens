import os
import json
import pandas as pd

from django.conf import settings
from django.shortcuts import render


BASE_DIR = settings.BASE_DIR


def load_csv(relative_path):
    """
    Load a CSV file from the MarketLens project directory.
    """
    file_path = os.path.join(BASE_DIR, relative_path)

    if not os.path.exists(file_path):
        raise FileNotFoundError(
            f"CSV file not found: {file_path}"
        )

    return pd.read_csv(file_path)


# ============================================================
# DASHBOARD HOME
# ============================================================

def dashboard_home(request):

    try:
        df = load_csv(
            "data/model_results/market_signals.csv"
        )

        # ----------------------------------------------------
        # Prepare dates
        # ----------------------------------------------------

        if "Date" in df.columns:
            df["Date"] = pd.to_datetime(
                df["Date"],
                errors="coerce"
            )

        # ----------------------------------------------------
        # Make sure required signal columns exist
        # ----------------------------------------------------

        if "Technical_Signal" not in df.columns:

            if "Signal" in df.columns:
                df["Technical_Signal"] = df["Signal"]

            else:
                df["Technical_Signal"] = "HOLD / NEUTRAL"

        if "Trend" not in df.columns:
            df["Trend"] = "Mixed"

        if "Signal_Strength" not in df.columns:
            df["Signal_Strength"] = "Weak"

        # ----------------------------------------------------
        # Get latest row for every company
        # ----------------------------------------------------

        if "Date" in df.columns:

            latest_signals = (
                df.sort_values("Date")
                .groupby("Company", as_index=False)
                .tail(1)
                .copy()
            )

        else:

            latest_signals = df.copy()

        # ----------------------------------------------------
        # NORMALIZE SIGNAL TEXT
        #
        # Actual values are:
        # BUY / POSITIVE
        # HOLD / NEUTRAL
        # SELL / NEGATIVE
        # ----------------------------------------------------

        signal_text = (
            latest_signals["Technical_Signal"]
            .fillna("")
            .astype(str)
            .str.upper()
            .str.strip()
        )

        # ----------------------------------------------------
        # Count BUY / HOLD / SELL
        # ----------------------------------------------------

        buy_count = int(
            signal_text.str.contains(
                "BUY",
                regex=False
            ).sum()
        )

        hold_count = int(
            signal_text.str.contains(
                "HOLD",
                regex=False
            ).sum()
        )

        sell_count = int(
            signal_text.str.contains(
                "SELL",
                regex=False
            ).sum()
        )

        # ----------------------------------------------------
        # Total companies
        # ----------------------------------------------------

        total_companies = int(
            latest_signals["Company"].nunique()
        )

        # ----------------------------------------------------
        # Trend counts
        # ----------------------------------------------------

        trend_text = (
            latest_signals["Trend"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        bullish_count = int(
            (trend_text == "Bullish").sum()
        )

        mixed_count = int(
            (trend_text == "Mixed").sum()
        )

        bearish_count = int(
            (trend_text == "Bearish").sum()
        )

        # ----------------------------------------------------
        # Create records for dashboard table
        # ----------------------------------------------------

        signal_records = []

        for _, row in latest_signals.iterrows():

            signal_records.append({

                "Company": row.get(
                    "Company",
                    ""
                ),

                "Ticker": row.get(
                    "Ticker",
                    ""
                ),

                "Close": float(
                    row.get(
                        "Close",
                        0
                    )
                ),

                "Technical_Signal": row.get(
                    "Technical_Signal",
                    "HOLD / NEUTRAL"
                ),

                "Signal": row.get(
                    "Technical_Signal",
                    "HOLD / NEUTRAL"
                ),

                "Signal_Strength": row.get(
                    "Signal_Strength",
                    "Weak"
                ),

                "Trend": row.get(
                    "Trend",
                    "Mixed"
                ),

                "Volatility_Level": row.get(
                    "Volatility_Level",
                    "Moderate"
                ),

                "RSI_14": row.get(
                    "RSI_14",
                    None
                ),

                "Signal_Score": row.get(
                    "Signal_Score",
                    0
                ),
            })

        # ----------------------------------------------------
        # Chart data
        # ----------------------------------------------------

        chart_labels = [
            "BUY",
            "HOLD",
            "SELL"
        ]

        chart_values = [
            buy_count,
            hold_count,
            sell_count
        ]

        # ----------------------------------------------------
        # Dashboard context
        # ----------------------------------------------------

        context = {

            "signals": signal_records,

            "signal_records": signal_records,

            "total_companies": total_companies,

            "buy_count": buy_count,

            "hold_count": hold_count,

            "sell_count": sell_count,

            "bullish_count": bullish_count,

            "mixed_count": mixed_count,

            "bearish_count": bearish_count,

            "chart_labels": json.dumps(
                chart_labels
            ),

            "chart_values": json.dumps(
                chart_values
            ),
        }

        return render(
            request,
            "dashboard/index.html",
            context
        )

    except Exception as e:

        return render(
            request,
            "dashboard/index.html",
            {

                "error": str(e),

                "signals": [],

                "signal_records": [],

                "total_companies": 0,

                "buy_count": 0,

                "hold_count": 0,

                "sell_count": 0,

                "bullish_count": 0,

                "mixed_count": 0,

                "bearish_count": 0,

                "chart_labels": json.dumps(
                    [
                        "BUY",
                        "HOLD",
                        "SELL"
                    ]
                ),

                "chart_values": json.dumps(
                    [
                        0,
                        0,
                        0
                    ]
                ),
            }
        )
# ============================================================
# COMPANY DETAIL
# ============================================================

def company_detail(request, ticker):

    try:

        # ----------------------------------------------------
        # Load historical data
        # ----------------------------------------------------

        df = load_csv(
            "data/advanced_features.csv"
        )

        # ----------------------------------------------------
        # Load market signals
        # ----------------------------------------------------

        signals_df = load_csv(
            "data/model_results/market_signals.csv"
        )

        # ----------------------------------------------------
        # Filter company
        # ----------------------------------------------------

        company_df = df[
            df["Ticker"].astype(str).str.upper()
            == ticker.upper()
        ].copy()

        if company_df.empty:
            return render(
                request,
                "dashboard/company.html",
                {
                    "error": (
                        f"No historical data found "
                        f"for {ticker}"
                    )
                }
            )

        # ----------------------------------------------------
        # Sort historical data
        # ----------------------------------------------------

        company_df["Date"] = pd.to_datetime(
            company_df["Date"],
            errors="coerce"
        )

        company_df = company_df.sort_values(
            "Date"
        )

        # ----------------------------------------------------
        # Latest historical row
        # ----------------------------------------------------

        latest = company_df.iloc[-1]

        # ----------------------------------------------------
        # Find latest market signal
        # ----------------------------------------------------

        signal_df = signals_df[
            signals_df["Ticker"].astype(str).str.upper()
            == ticker.upper()
        ].copy()

        if not signal_df.empty:

            signal_df["Date"] = pd.to_datetime(
                signal_df["Date"],
                errors="coerce"
            )

            signal_df = signal_df.sort_values(
                "Date"
            )

            signal = signal_df.iloc[-1]

            technical_signal = signal.get(
                "Technical_Signal",
                signal.get("Signal", "HOLD")
            )

            trend = signal.get(
                "Trend",
                "Mixed"
            )

            signal_strength = signal.get(
                "Signal_Strength",
                "Weak"
            )

            signal_score = signal.get(
                "Signal_Score",
                0
            )

            volatility_level = signal.get(
                "Volatility_Level",
                "Moderate"
            )

        else:

            technical_signal = "HOLD"
            trend = "Mixed"
            signal_strength = "Weak"
            signal_score = 0
            volatility_level = "Moderate"

        # ----------------------------------------------------
        # Historical chart data
        # ----------------------------------------------------

        chart_df = company_df[
            [
                "Date",
                "Close",
                "MA_20",
                "MA_50"
            ]
        ].copy()

        chart_df = chart_df.dropna(
            subset=["Date"]
        )

        chart_dates = [
            date.strftime("%Y-%m-%d")
            for date in chart_df["Date"]
        ]

        chart_close = [
            None if pd.isna(value)
            else float(value)
            for value in chart_df["Close"]
        ]

        chart_ma20 = [
            None if pd.isna(value)
            else float(value)
            for value in chart_df["MA_20"]
        ]

        chart_ma50 = [
            None if pd.isna(value)
            else float(value)
            for value in chart_df["MA_50"]
        ]

        # ----------------------------------------------------
        # Technical indicators
        # ----------------------------------------------------

        rsi = latest.get("RSI_14", None)
        macd = latest.get("MACD", None)
        macd_signal = latest.get(
            "MACD_Signal",
            None
        )
        volatility = latest.get(
            "Volatility_20",
            None
        )

        # ----------------------------------------------------
        # Company name
        # ----------------------------------------------------

        company_name = latest.get(
            "Company",
            ticker
        )

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        context = {

            # Company information
            "company": company_name,
            "company_name": company_name,
            "ticker": ticker,

            # Price
            "current_price": float(
                latest["Close"]
            ),

            "Close": float(
                latest["Close"]
            ),

            # Market signal
            "Technical_Signal": technical_signal,
            "technical_signal": technical_signal,

            # Compatibility with templates
            "Signal": technical_signal,
            "signal": technical_signal,

            "Trend": trend,
            "trend": trend,

            "Signal_Strength": signal_strength,
            "signal_strength": signal_strength,

            "Signal_Score": signal_score,
            "signal_score": signal_score,

            "Volatility_Level": volatility_level,
            "volatility_level": volatility_level,

            # Indicators
            "rsi": rsi,
            "macd": macd,
            "macd_signal": macd_signal,
            "volatility": volatility,

            # Chart
            "chart_dates": json.dumps(
                chart_dates
            ),

            "chart_close": json.dumps(
                chart_close
            ),

            "chart_ma20": json.dumps(
                chart_ma20
            ),

            "chart_ma50": json.dumps(
                chart_ma50
            ),

            # Useful for debugging
            "history_rows": len(
                company_df
            ),
        }

        return render(
            request,
            "dashboard/company.html",
            context
        )

    except Exception as e:

        return render(
            request,
            "dashboard/company.html",
            {
                "error": str(e),
                "ticker": ticker,
            }
        )


# ============================================================
# SECTOR ANALYSIS
# ============================================================

def sector_analysis(request):

    try:

        df = load_csv(
            "data/advanced_features.csv"
        )

        # ----------------------------------------------------
        # Company -> sector mapping
        # ----------------------------------------------------

        sector_mapping = {

            "M&M": "Automobile",
            "Maruti Suzuki": "Automobile",

            "SBI": "Banking",
            "ICICI Bank": "Banking",
            "Axis Bank": "Banking",
            "HDFC Bank": "Banking",

            "NTPC": "Energy",
            "ONGC": "Energy",
            "Reliance": "Energy",

            "Hindustan Unilever": "FMCG",
            "ITC": "FMCG",

            "HCLTech": "IT",
            "Infosys": "IT",
            "TCS": "IT",
            "Wipro": "IT",

            "Tata Steel": "Metals",

            "Dr Reddy's": "Pharma",
            "Sun Pharma": "Pharma",

            "Bharti Airtel": "Telecom",
        }

        df["Sector"] = df["Company"].map(
            sector_mapping
        )

        # ----------------------------------------------------
        # Sector summary
        # ----------------------------------------------------

        sector_summary = (
            df.groupby("Sector")
            .agg(
                Companies=("Company", "nunique"),
                Average_Daily_Return=(
                    "Daily_Return",
                    "mean"
                ),
                Average_Volatility=(
                    "Volatility_20",
                    "mean"
                ),
                Average_Trading_Volume=(
                    "Volume",
                    "mean"
                ),
            )
            .reset_index()
        )

        sector_summary = sector_summary.dropna(
            subset=["Sector"]
        )

        sector_summary = sector_summary.sort_values(
            "Average_Daily_Return",
            ascending=False
        )

        # ----------------------------------------------------
        # Chart data
        # ----------------------------------------------------

        sector_labels = (
            sector_summary["Sector"]
            .tolist()
        )

        sector_returns = [
            float(x)
            if pd.notna(x)
            else 0
            for x in sector_summary[
                "Average_Daily_Return"
            ]
        ]

        sector_volatility = [
            float(x)
            if pd.notna(x)
            else 0
            for x in sector_summary[
                "Average_Volatility"
            ]
        ]

        context = {

            "sector_table":
                sector_summary.to_dict(
                    "records"
                ),

            "sector_labels":
                json.dumps(
                    sector_labels
                ),

            "sector_returns":
                json.dumps(
                    sector_returns
                ),

            "sector_volatility":
                json.dumps(
                    sector_volatility
                ),
        }

        return render(
            request,
            "dashboard/sectors.html",
            context
        )

    except Exception as e:

        return render(
            request,
            "dashboard/sectors.html",
            {
                "error": str(e),
                "sector_table": [],
                "sector_labels": json.dumps([]),
                "sector_returns": json.dumps([]),
                "sector_volatility": json.dumps([]),
            }
        )


# ============================================================
# PREDICTION ANALYSIS
# ============================================================

def prediction_analysis(request):

    try:

        # ----------------------------------------------------
        # Load final model comparison
        # ----------------------------------------------------

        final_df = load_csv(
            "data/model_results/final_model_comparison.csv"
        )

        # ----------------------------------------------------
        # Separate classification models
        # ----------------------------------------------------

        classification_df = final_df[
            final_df["Task"].astype(str).str.contains(
                "Direction",
                case=False,
                na=False
            )
        ].copy()

        # ----------------------------------------------------
        # Separate regression models
        # ----------------------------------------------------

        regression_df = final_df[
            final_df["Task"].astype(str).str.contains(
                "Return",
                case=False,
                na=False
            )
        ].copy()

        # ----------------------------------------------------
        # Classification metrics
        # ----------------------------------------------------

        if not classification_df.empty:

            classification_df = classification_df[
                [
                    "Model",
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1",
                    "ROC_AUC",
                    "Baseline",
                    "Verdict"
                ]
            ].copy()

            classification_df = classification_df.dropna(
                subset=["Accuracy"]
            )

        # ----------------------------------------------------
        # Regression metrics
        # ----------------------------------------------------

        if not regression_df.empty:

            regression_df = regression_df[
                [
                    "Model",
                    "MAE",
                    "RMSE",
                    "R2",
                    "Baseline",
                    "Verdict"
                ]
            ].copy()

            regression_df = regression_df.dropna(
                subset=["MAE"]
            )

        # ----------------------------------------------------
        # Best classification model
        # ----------------------------------------------------

        best_classification = None

        if not classification_df.empty:

            best_classification = (
                classification_df
                .sort_values(
                    "Accuracy",
                    ascending=False
                )
                .iloc[0]
                .to_dict()
            )

        # ----------------------------------------------------
        # Best regression model
        # ----------------------------------------------------

        best_regression = None

        if not regression_df.empty:

            best_regression = (
                regression_df
                .sort_values(
                    "MAE",
                    ascending=True
                )
                .iloc[0]
                .to_dict()
            )

        # ----------------------------------------------------
        # Classification chart data
        # ----------------------------------------------------

        chart_labels = (
            classification_df["Model"]
            .astype(str)
            .tolist()
        )

        accuracy_values = [
            float(x)
            for x in classification_df["Accuracy"]
        ]

        f1_values = [
            float(x)
            for x in classification_df["F1"]
        ]

        # ----------------------------------------------------
        # Summary values
        # ----------------------------------------------------

        total_classification_models = len(
            classification_df
        )

        total_regression_models = len(
            regression_df
        )

        # ----------------------------------------------------
        # Context
        # ----------------------------------------------------

        context = {

            # Classification
            "classification_table":
                classification_df.to_dict(
                    "records"
                ),

            "best_classification":
                best_classification,

            "total_classification_models":
                total_classification_models,

            # Regression
            "regression_table":
                regression_df.to_dict(
                    "records"
                ),

            "return_table":
                regression_df.to_dict(
                    "records"
                ),

            "best_regression":
                best_regression,

            "best_return":
                best_regression,

            "total_regression_models":
                total_regression_models,

            # Chart
            "chart_labels":
                json.dumps(
                    chart_labels
                ),

            "accuracy_values":
                json.dumps(
                    accuracy_values
                ),

            "f1_values":
                json.dumps(
                    f1_values
                ),

            # General
            "total_models":
                total_classification_models,

        }

        return render(
            request,
            "dashboard/prediction.html",
            context
        )

    except Exception as e:

        return render(
            request,
            "dashboard/prediction.html",
            {
                "error": str(e),

                "classification_table": [],
                "regression_table": [],
                "return_table": [],

                "best_classification": None,
                "best_regression": None,
                "best_return": None,

                "total_classification_models": 0,
                "total_regression_models": 0,
                "total_models": 0,

                "chart_labels":
                    json.dumps([]),

                "accuracy_values":
                    json.dumps([]),

                "f1_values":
                    json.dumps([]),
            }
        )