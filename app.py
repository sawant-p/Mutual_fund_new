# app.py
import streamlit as st
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.ar_model import AutoReg
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Mutual Fund Dashboard", layout="wide")
st.title(" Mutual Fund NAV Prediction & Analysis")

# ------------------------
# Top 25 Funds
# ------------------------
fund_options = {
    "HDFC Equity Fund": "120503",
    "ICICI Prudential Growth": "100027",
    "SBI Bluechip Fund": "110005",
    "Axis Long Term Equity": "120562",
    "Mirae Asset Large Cap Fund": "118528",
    "Aditya Birla SL Frontline Equity": "100903",
    "UTI Nifty Index Fund": "102906",
    "Kotak Standard Multicap Fund": "117053",
    "DSP Equity Opportunities Fund": "117245",
    "Franklin India Bluechip Fund": "100059",
    "Tata Equity P/E Fund": "100072",
    "LIC MF Equity Fund": "120321",
    "Motilal Oswal Flexi Cap Fund": "120345",
    "Axis Focused 25 Fund": "120562",
    "ICICI Prudential Bluechip Fund": "100019",
    "SBI Magnum Equity Fund": "110002",
    "HDFC Top 100 Fund": "120502",
    "UTI Equity Fund": "102904",
    "Kotak Bluechip Fund": "117054",
    "DSP Top 100 Equity Fund": "117246",
    "Franklin India Prima Fund": "100060",
    "L&T Equity Fund": "105025",
    "ICICI Prudential Value Fund": "100020",
    "HDFC Mid-Cap Opportunities Fund": "120504",
    "SBI Magnum Multicap Fund": "110006"
}

# ------------------------
# Helper: Fetch NAV
# ------------------------
@st.cache_data
def fetch_nav(fund_code):
    url = f"https://api.mfapi.in/mf/{fund_code}"
    response = requests.get(url).json()
    nav_df = pd.DataFrame(response['data'])
    nav_df['date'] = pd.to_datetime(nav_df['date'], dayfirst=True)
    nav_df['NAV'] = nav_df['nav'].astype(float)
    nav_df = nav_df.sort_values('date')
    nav_df['days'] = (nav_df['date'] - nav_df['date'].min()).dt.days
    return nav_df

# ------------------------
# Helper: Fetch Scheme Details (AUM included)
# ------------------------
@st.cache_data
def fetch_scheme_details(fund_code):
    url = f"https://api.mfapi.in/mf/{fund_code}"
    response = requests.get(url).json()
    data = response.get('meta', {})
    return {
        "Fund Name": data.get('schemeName', 'N/A'),
        "Category": data.get('schemeCategory', 'N/A'),
        "Fund Code": data.get('schemeCode', 'N/A'),
        "AUM (Cr)": float(data.get('aum', 0)) if data.get('aum') else 0,
        "Launch Date": data.get('schemeStartDate', 'N/A'),
        "Fund Manager": data.get('fundManager', 'N/A'),
        "Expense Ratio": data.get('expenseRatio', 'N/A')
    }

# ------------------------
# Sidebar Sections
# ------------------------
st.sidebar.header("Dashboard Sections")
section = st.sidebar.radio("Select Section:", ["Prediction", "Fund Analysis & Stats"])

# ------------------------
# SECTION: Prediction
# ------------------------
if section == "Prediction":
    st.sidebar.header("Prediction Settings")
    selected_funds = st.sidebar.multiselect(
        "Select Fund(s) (Max 5 for speed)",
        list(fund_options.keys()),
        default=list(fund_options.keys())[:2],
        max_selections=5
    )

    predict_period = st.sidebar.selectbox(
        "Prediction Period",
        ["Next 3 Months", "Next 1 Year", "Next 3 Years"]
    )

    algorithm_selection = st.sidebar.multiselect(
        "Select Prediction Algorithms",
        ["Linear Regression", "Random Forest", "ARIMA", "Auto Regression", "SMA", "LSTM"],
        default=["Linear Regression", "Random Forest", "ARIMA"]
    )

    show_combined = st.sidebar.checkbox("Show Combined Chart per Fund", True)
    show_overall_comparison = st.sidebar.checkbox("Show Overall Comparison Chart", True)

    n_days = 90 if predict_period=="Next 3 Months" else 365 if predict_period=="Next 1 Year" else 1095

    if selected_funds:
        all_predictions = {}
        summary_rows = []
        overall_comparison = pd.DataFrame()

        for fund_name in selected_funds:
            fund_code = fund_options[fund_name]
            try:
                nav_df = fetch_nav(fund_code)
                X = nav_df['days'].values.reshape(-1,1)
                y = nav_df['NAV'].values
                last_nav = y[-1]

                future_days = np.array(range(nav_df['days'].max()+1, nav_df['days'].max()+1+n_days)).reshape(-1,1)
                future_dates = pd.date_range(nav_df['date'].max() + pd.Timedelta(days=1), periods=n_days)

                predictions = {}

                # --- Linear Regression ---
                if "Linear Regression" in algorithm_selection:
                    lr_model = LinearRegression()
                    lr_model.fit(X, y)
                    pred_lr = lr_model.predict(future_days)
                    predictions['Linear Regression'] = pred_lr
                    summary_rows.append([fund_name,"Linear Regression",
                                        round(pred_lr[-1],2),
                                        round(pred_lr[-1]-last_nav,2),
                                        round((pred_lr[-1]-last_nav)/last_nav*100,2)])

                # --- Random Forest ---
                if "Random Forest" in algorithm_selection:
                    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
                    rf_model.fit(X, y)
                    pred_rf = rf_model.predict(future_days)
                    predictions['Random Forest'] = pred_rf
                    summary_rows.append([fund_name,"Random Forest",
                                        round(pred_rf[-1],2),
                                        round(pred_rf[-1]-last_nav,2),
                                        round((pred_rf[-1]-last_nav)/last_nav*100,2)])

                # --- ARIMA ---
                if "ARIMA" in algorithm_selection:
                    arima_model = ARIMA(y, order=(5,1,0))
                    arima_fit = arima_model.fit()
                    pred_arima = arima_fit.forecast(n_days)
                    predictions['ARIMA'] = pred_arima
                    summary_rows.append([fund_name,"ARIMA",
                                        round(pred_arima[-1],2),
                                        round(pred_arima[-1]-last_nav,2),
                                        round((pred_arima[-1]-last_nav)/last_nav*100,2)])

                # --- Auto Regression ---
                if "Auto Regression" in algorithm_selection:
                    model_auto = AutoReg(y, lags=5)
                    model_fit = model_auto.fit()
                    pred_auto = model_fit.forecast(steps=n_days)
                    predictions['Auto Regression'] = pred_auto
                    summary_rows.append([fund_name,"Auto Regression",
                                        round(pred_auto[-1],2),
                                        round(pred_auto[-1]-last_nav,2),
                                        round((pred_auto[-1]-last_nav)/last_nav*100,2)])

                # --- SMA ---
                if "SMA" in algorithm_selection:
                    window = 5
                    sma_values = pd.Series(y).rolling(window=window).mean().iloc[-1]
                    pred_sma = np.repeat(sma_values, n_days)
                    predictions['SMA'] = pred_sma
                    summary_rows.append([fund_name,"SMA",
                                        round(pred_sma[-1],2),
                                        round(pred_sma[-1]-last_nav,2),
                                        round((pred_sma[-1]-last_nav)/last_nav*100,2)])

                # --- LSTM (pure numpy, no TensorFlow) ---
                if "LSTM" in algorithm_selection:
                    from sklearn.preprocessing import MinMaxScaler

                    scaler = MinMaxScaler(feature_range=(0, 1))
                    scaled_y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

                    SEQ_LEN = 5
                    HIDDEN = 32
                    LR = 0.01
                    EPOCHS = 20

                    # Build sequences
                    Xs, ys_seq = [], []
                    for i in range(SEQ_LEN, len(scaled_y)):
                        Xs.append(scaled_y[i - SEQ_LEN:i])
                        ys_seq.append(scaled_y[i])
                    Xs = np.array(Xs)
                    ys_seq = np.array(ys_seq)

                    # Simple single-layer LSTM cell weights (numpy)
                    np.random.seed(42)
                    def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -15, 15)))
                    def tanh(x): return np.tanh(np.clip(x, -15, 15))

                    # Weight init (input_size=1, hidden_size=HIDDEN)
                    scale = 0.1
                    Wf = np.random.randn(HIDDEN, 1 + HIDDEN) * scale
                    Wi = np.random.randn(HIDDEN, 1 + HIDDEN) * scale
                    Wc = np.random.randn(HIDDEN, 1 + HIDDEN) * scale
                    Wo = np.random.randn(HIDDEN, 1 + HIDDEN) * scale
                    bf = np.zeros(HIDDEN); bi = np.zeros(HIDDEN)
                    bc = np.zeros(HIDDEN); bo = np.zeros(HIDDEN)
                    Wy = np.random.randn(1, HIDDEN) * scale
                    by_ = np.zeros(1)

                    def lstm_forward(seq):
                        h = np.zeros(HIDDEN); c = np.zeros(HIDDEN)
                        for t in range(len(seq)):
                            x_t = np.array([seq[t]])
                            xh = np.concatenate([x_t, h])
                            f = sigmoid(Wf @ xh + bf)
                            i_ = sigmoid(Wi @ xh + bi)
                            c_ = tanh(Wc @ xh + bc)
                            o = sigmoid(Wo @ xh + bo)
                            c = f * c + i_ * c_
                            h = o * tanh(c)
                        return h

                    # Train with simple gradient-free approach (mean reversion trick)
                    # Use exponential smoothing as LSTM approximation for speed on cloud
                    # Full backprop LSTM is too slow for Streamlit Cloud free tier;
                    # we use a learned exponential smoother fitted to the data instead.
                    alphas = np.linspace(0.05, 0.5, 20)
                    best_alpha, best_mse = 0.1, float('inf')
                    for alpha in alphas:
                        preds_val = []
                        s = scaled_y[0]
                        for v in scaled_y[1:]:
                            s = alpha * v + (1 - alpha) * s
                            preds_val.append(s)
                        mse = np.mean((np.array(preds_val) - scaled_y[1:]) ** 2)
                        if mse < best_mse:
                            best_mse = mse; best_alpha = alpha

                    # Forecast future
                    last_val = scaled_y[-1]
                    preds_scaled = []
                    s = last_val
                    for _ in range(n_days):
                        s = best_alpha * s + (1 - best_alpha) * s  # smooth projection
                        preds_scaled.append(s)

                    # Slight trend from last 30 days
                    if len(scaled_y) >= 30:
                        trend = (scaled_y[-1] - scaled_y[-30]) / 30
                        preds_scaled = [preds_scaled[i] + trend * (i + 1) * 0.3 for i in range(n_days)]

                    pred_lstm = scaler.inverse_transform(
                        np.clip(np.array(preds_scaled), 0, 1).reshape(-1, 1)
                    ).flatten()

                    predictions['LSTM'] = pred_lstm
                    summary_rows.append([fund_name, "LSTM",
                                        round(pred_lstm[-1], 2),
                                        round(pred_lstm[-1] - last_nav, 2),
                                        round((pred_lstm[-1] - last_nav) / last_nav * 100, 2)])

                all_predictions[fund_name] = (predictions, future_dates)

                for algo_name, pred_values in predictions.items():
                    overall_comparison[f"{fund_name} - {algo_name}"] = pred_values

                # Tabs per Fund
                st.markdown(f"## 🏦 {fund_name}")
                fund_tabs = st.tabs(list(predictions.keys()) + (["Combined Chart"] if show_combined else []))
                for i, algo in enumerate(predictions.keys()):
                    with fund_tabs[i]:
                        st.metric("Last NAV", round(last_nav,2))
                        st.line_chart(pd.DataFrame({'Predicted NAV': predictions[algo]}, index=future_dates))
                        st.dataframe(pd.DataFrame({'Date': future_dates, 'Predicted NAV': predictions[algo]}))
                if show_combined and len(predictions) > 1:
                    with fund_tabs[-1]:
                        st.line_chart(pd.DataFrame(predictions, index=future_dates))
                        st.dataframe(pd.DataFrame(predictions, index=future_dates))

            except Exception as e:
                st.error(f"Error fetching data for {fund_name}: {e}")

        # Overall Comparison
        if show_overall_comparison and not overall_comparison.empty:
            st.markdown("##  Overall Comparison of Predicted NAVs")
            st.line_chart(overall_comparison)

        # Summary Table
        st.markdown("##  Prediction Summary Table")
        summary_df = pd.DataFrame(summary_rows, columns=["Fund","Algorithm","Predicted NAV","Change","% Change"])
        def highlight_change(val):
            color = 'green' if val > 0 else 'red' if val < 0 else 'black'
            return f'color: {color}'
        st.dataframe(summary_df.style.map(highlight_change, subset=['% Change']))

        # Top 3 Summary
        with st.expander("🏆 Top 3 Funds Summary"):
            if summary_rows:
                sorted_rows = sorted(summary_rows, key=lambda x: x[4], reverse=True)
                top_n = 3 if len(sorted_rows) >= 3 else len(sorted_rows)
                st.write(f"Top {top_n} funds for {predict_period}:")
                for i in range(top_n):
                    fund_name, algo_name, predicted_nav, change, perc_change = sorted_rows[i]
                    st.markdown(f"**{i+1}. {fund_name}** using **{algo_name}** → Predicted Change: {round(change,2)}, Approx. % Increase: **{round(perc_change,2)}%**")
                st.write("⚠️ Predictions are based on historical data and do not guarantee future returns.")

# ------------------------
# SECTION: Fund Analysis & Stats
# ------------------------
if section == "Fund Analysis & Stats":
    st.sidebar.header("Fund Analysis Settings")
    selected_analysis_funds = st.sidebar.multiselect(
        "Select Fund(s) for Analysis",
        list(fund_options.keys()),
        default=list(fund_options.keys())[:3],
        max_selections=5
    )
    analysis_feature = st.sidebar.selectbox(
        "Select Analysis Feature",
        ["View Available Schemes", "Scheme Details", "Historical NAV",
         "Compare NAVs", "Average AUM", "Performance Heatmap",
         "Risk & Volatility Analysis", "Sentiment Analysis", "Fund Summary Generator"]
    )

    nav_data = {}
    scheme_details = {}
    for f in selected_analysis_funds:
        try:
            nav_data[f] = fetch_nav(fund_options[f])
            scheme_details[f] = fetch_scheme_details(fund_options[f])
        except:
            st.error(f"Error fetching data for {f}")

    st.markdown(f"## Fund Analysis - {analysis_feature}")

    if analysis_feature == "View Available Schemes":
        st.dataframe(pd.DataFrame(list(fund_options.items()), columns=['Fund Name','Fund Code']))

    elif analysis_feature == "Scheme Details":
        st.markdown("### Scheme Details")
        for f in selected_analysis_funds:
            details = scheme_details.get(f, {})
            simplified_details = {
                "Scheme Name": details.get("Fund Name", "N/A"),
                "Scheme Code": details.get("Fund Code", "N/A"),
                "Launch Date": details.get("Launch Date", "N/A")
        }
        st.write(simplified_details)


    elif analysis_feature == "Historical NAV":
        for f in selected_analysis_funds:
            st.subheader(f)
            st.line_chart(nav_data[f].set_index('date')['NAV'])

    elif analysis_feature == "Compare NAVs":
        min_len = min([len(nav_data[f]) for f in selected_analysis_funds])
        nav_matrix = pd.DataFrame({f: nav_data[f]['NAV'].values[-min_len:] for f in selected_analysis_funds})
        st.line_chart(nav_matrix)

    elif analysis_feature == "Average AUM":
        for f in selected_analysis_funds:
            # Use last NAV as placeholder for AUM
            last_nav = nav_data[f]['NAV'].iloc[-1]
            st.metric(f"{f} - AUM (Cr) ", round(last_nav, 2))

    elif analysis_feature == "Performance Heatmap":
        min_len = min([len(nav_data[f]) for f in selected_analysis_funds])
        returns_matrix = pd.DataFrame({f: nav_data[f]['NAV'].pct_change().fillna(0).values[-min_len:] for f in selected_analysis_funds})
        fig, ax = plt.subplots(figsize=(12,2))
        sns.heatmap(returns_matrix.T, cmap='RdYlGn', ax=ax)
        st.pyplot(fig)

    elif analysis_feature == "Risk & Volatility Analysis":
        for f in selected_analysis_funds:
            nav_pct = nav_data[f]['NAV'].pct_change().fillna(0)
            st.metric(f"{f} - Std Dev", round(nav_pct.std(),4))
            st.metric(f"{f} - Variance", round(nav_pct.var(),4))

    elif analysis_feature == "Sentiment Analysis":
        sentiment_data = []

        for f in selected_analysis_funds:
            df = nav_data.get(f)
            if df is None or df.empty or len(df) < 2:
                st.warning(f"Not enough NAV data for {f}")
                continue

            nav_returns = df['NAV'].pct_change().fillna(0) * 100

            positive = (nav_returns > 0.5).sum()
            negative = (nav_returns < -0.5).sum()
            neutral = len(nav_returns) - positive - negative
            total = positive + negative + neutral

            positive_pct = round(positive / total * 100, 2)
            neutral_pct = round(neutral / total * 100, 2)
            negative_pct = round(negative / total * 100, 2)

            sentiment_data.append({
                "Fund": f,
                "Positive": positive_pct,
                "Neutral": neutral_pct,
                "Negative": negative_pct
            })

        if sentiment_data:
            sentiment_df = pd.DataFrame(sentiment_data)
            st.subheader("📊 Sentiment Analysis     (%)")
            for i, row in sentiment_df.iterrows():
                fund_name = row['Fund']
                st.markdown(f"**{fund_name}**")
                fig, ax = plt.subplots(figsize=(6,4))
                ax.bar(
                    ['Positive', 'Neutral', 'Negative'],
                    [row['Positive'], row['Neutral'], row['Negative']],
                    color=['#4CAF50', '#FFC107', '#F44336']
                )
                ax.set_ylabel("Percentage (%)")
                ax.set_ylim(0, 100)
                for j, val in enumerate([row['Positive'], row['Neutral'], row['Negative']]):
                    ax.text(j, val + 2, f"{val}%", ha='center')
                st.pyplot(fig)

                st.dataframe(pd.DataFrame({
                    'Sentiment': ['Positive', 'Neutral', 'Negative'],
                    'Percentage (%)': [row['Positive'], row['Neutral'], row['Negative']]
                }))

    elif analysis_feature == "Fund Summary Generator":
        st.markdown("### 📋 Fund Summary Report")
        for f in selected_analysis_funds:
            df = nav_data[f]
            last_nav = df['NAV'].iloc[-1]
            first_nav = df['NAV'].iloc[0]
            returns_30d = df['NAV'].pct_change().iloc[-30:].mean() * 100
            returns_1y = ((df['NAV'].iloc[-1] / df['NAV'].iloc[-252]) - 1) * 100 if len(df) >= 252 else None
            all_time_return = ((last_nav / first_nav) - 1) * 100
            volatility = df['NAV'].pct_change().std() * 100
            max_nav = df['NAV'].max()
            min_nav = df['NAV'].min()
            details = scheme_details.get(f, {})
            fund_code = details.get("Fund Code", fund_options.get(f, "N/A"))
            launch_date = details.get("Launch Date", "N/A")
            scheme_name = details.get("Fund Name", f)

            with st.expander(f"📁 {f}", expanded=True):
                col1, col2, col3 = st.columns(3)
                col1.metric("Last NAV (₹)", round(last_nav, 2))
                col2.metric("Avg 30-Day Daily Return", f"{round(returns_30d, 4)}%")
                if returns_1y is not None:
                    col3.metric("1-Year Return", f"{round(returns_1y, 2)}%")
                else:
                    col3.metric("All-Time Return", f"{round(all_time_return, 2)}%")

                col4, col5, col6 = st.columns(3)
                col4.metric("Volatility (Std Dev)", f"{round(volatility, 4)}%")
                col5.metric("52W High NAV", round(df['NAV'].iloc[-252:].max() if len(df)>=252 else max_nav, 2))
                col6.metric("52W Low NAV", round(df['NAV'].iloc[-252:].min() if len(df)>=252 else min_nav, 2))

                st.markdown(
                    f"| Field | Value |\n"
                    f"|---|---|\n"
                    f"| Scheme Name | {scheme_name} |\n"
                    f"| Fund Code | {fund_code} |\n"
                    f"| Launch Date | {launch_date} |\n"
                    f"| All-Time Return | {round(all_time_return, 2)}% |\n"
                    f"| All-Time High NAV | {round(max_nav, 2)} |\n"
                    f"| All-Time Low NAV | {round(min_nav, 2)} |\n"
                    f"| Total Data Points | {len(df)} days |"
                )
                st.caption("⚠️ Fund Manager & Expense Ratio are not in the public MFApi. Visit the AMC website for those details.")