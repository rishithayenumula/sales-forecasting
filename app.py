"""
Sales Forecasting & Demand Intelligence Dashboard
Run locally:  streamlit run app.py
Deploy: push this folder to GitHub, then deploy on https://share.streamlit.io
         pointing at app.py (train.csv must sit next to it in the repo).
"""
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error

st.set_page_config(page_title="Sales Forecasting & Demand Intelligence", layout="wide")

# Make the sidebar navigation menu bigger and easier to read
st.markdown("""
<style>
/* Sidebar title */
section[data-testid="stSidebar"] h1 {
    font-size: 30px !important;
}
/* "Navigate" label above the radio menu */
section[data-testid="stSidebar"] .stRadio > label {
    font-size: 20px !important;
    font-weight: 700 !important;
}
/* Each menu option text */
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label p {
    font-size: 20px !important;
    padding: 6px 0px !important;
}
/* Radio button circle size */
section[data-testid="stSidebar"] .stRadio [role="radiogroup"] label div:first-child {
    transform: scale(1.3);
    margin-right: 8px;
}
</style>
""", unsafe_allow_html=True)

SARIMA_ORDER = (1, 1, 1)
SARIMA_SEASONAL_ORDER = (1, 1, 1, 12)
HORIZON_MAX = 3


# ----------------------------------------------------------------------------
# Data loading & caching
# ----------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], format="%d/%m/%Y")
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], format="%d/%m/%Y")
    df = df.drop_duplicates().reset_index(drop=True)
    df["Year"] = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Quarter"] = df["Order Date"].dt.quarter
    return df


@st.cache_data
def monthly_series(df, category=None, region=None):
    sub = df.copy()
    if category:
        sub = sub[sub["Category"] == category]
    if region:
        sub = sub[sub["Region"] == region]
    s = sub.set_index("Order Date").resample("MS")["Sales"].sum().asfreq("MS").fillna(0)
    return s


@st.cache_resource
def fit_sarima_and_forecast(series_key, series, horizon):
    train = series.iloc[:-horizon] if len(series) > horizon + 12 else series
    model = SARIMAX(series, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER,
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    fc = fit.get_forecast(steps=horizon)
    pred = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)
    return pred, ci


@st.cache_data
def backtest_metrics(series_key, series, horizon=3):
    """Hold out the last `horizon` months to report honest MAE/RMSE for the dashboard."""
    if len(series) <= horizon + 12:
        return None, None
    train = series.iloc[:-horizon]
    test = series.iloc[-horizon:]
    model = SARIMAX(train, order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER,
                     enforce_stationarity=False, enforce_invertibility=False)
    fit = model.fit(disp=False)
    pred = fit.get_forecast(steps=horizon).predicted_mean
    mae = mean_absolute_error(test.values, pred.values)
    rmse = np.sqrt(mean_squared_error(test.values, pred.values))
    return mae, rmse


@st.cache_data
def weekly_anomaly_table(df):
    weekly = df.set_index("Order Date").resample("W")["Sales"].sum()
    feat = pd.DataFrame({"sales": weekly.values}, index=weekly.index)
    feat["rolling_mean_4"] = feat["sales"].rolling(4, min_periods=1, center=True).mean()
    feat["deviation"] = feat["sales"] - feat["rolling_mean_4"]

    iso = IsolationForest(contamination=0.06, random_state=42)
    feat["iso_anomaly"] = iso.fit_predict(feat[["sales", "deviation"]]) == -1

    feat["roll_mean_prior"] = feat["sales"].shift(1).rolling(4, min_periods=4).mean()
    feat["roll_std_prior"] = feat["sales"].shift(1).rolling(4, min_periods=4).std()
    feat["z_score"] = (feat["sales"] - feat["roll_mean_prior"]) / feat["roll_std_prior"]
    feat["z_anomaly"] = feat["z_score"].abs() > 2

    feat["is_anomaly"] = feat["iso_anomaly"] | feat["z_anomaly"]
    return feat


@st.cache_data
def cluster_subcategories(df):
    monthly_sub = df.groupby(["Sub-Category", pd.Grouper(key="Order Date", freq="MS")])["Sales"].sum().reset_index()
    agg = df.groupby("Sub-Category").agg(
        total_sales_volume=("Sales", "sum"),
        avg_order_value=("Sales", "mean"),
    )
    stats = monthly_sub.groupby("Sub-Category")["Sales"].agg(["std", "mean"])
    agg["sales_volatility"] = stats["std"] / stats["mean"]

    year_totals = df.groupby(["Sub-Category", "Year"])["Sales"].sum().reset_index()
    def yoy(g):
        g = g.sort_values("Year")
        if len(g) < 2 or g["Sales"].iloc[0] == 0:
            return np.nan
        return (g["Sales"].iloc[-1] - g["Sales"].iloc[0]) / g["Sales"].iloc[0] / (len(g) - 1) * 100
    agg["avg_yoy_growth_pct"] = year_totals.groupby("Sub-Category").apply(yoy, include_groups=False)
    agg = agg.dropna()

    X = agg[["total_sales_volume", "avg_yoy_growth_pct", "sales_volatility", "avg_order_value"]]
    X_scaled = StandardScaler().fit_transform(X)
    km = KMeans(n_clusters=4, n_init=10, random_state=42)
    agg["cluster"] = km.fit_predict(X_scaled)

    profile = agg.groupby("cluster")[["total_sales_volume", "avg_yoy_growth_pct", "sales_volatility"]].mean()
    vol_med, volat_med = profile["total_sales_volume"].median(), profile["sales_volatility"].median()
    growth_rank = profile["avg_yoy_growth_pct"].sort_values(ascending=False)
    fastest, slowest = growth_rank.index[0], growth_rank.index[-1]
    labels = {}
    for cid, row in profile.iterrows():
        if cid == fastest and row["avg_yoy_growth_pct"] > 0:
            labels[cid] = "Growing Demand"
        elif cid == slowest and row["avg_yoy_growth_pct"] < profile["avg_yoy_growth_pct"].mean():
            labels[cid] = "Declining / Slow-Growth Demand"
        elif row["total_sales_volume"] >= vol_med and row["sales_volatility"] <= volat_med:
            labels[cid] = "High Volume, Stable Demand"
        else:
            labels[cid] = "Low Volume, High Volatility"
    agg["cluster_label"] = agg["cluster"].map(labels)

    pca = PCA(n_components=2, random_state=42)
    pcs = pca.fit_transform(X_scaled)
    agg["pc1"], agg["pc2"] = pcs[:, 0], pcs[:, 1]
    return agg


STOCKING_STRATEGY = {
    "High Volume, Stable Demand": "Maintain steady safety stock; use simple reorder-point replenishment; low forecast risk.",
    "Growing Demand": "Increase stock buffers ahead of forecasted growth; revisit reorder points monthly; watch for stockouts.",
    "Declining / Slow-Growth Demand": "Trim inventory gradually; avoid large bulk orders; consider promotions to clear stock.",
    "Low Volume, High Volatility": "Keep lean, frequent replenishment; avoid large upfront commitments; monitor demand spikes closely.",
}

df = load_data()

st.sidebar.title("📦 Sales Intelligence")
page = st.sidebar.radio("Navigate", ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"])

# ----------------------------------------------------------------------------
# PAGE 1: Sales Overview
# ----------------------------------------------------------------------------
if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2 = st.columns(2)
    with col1:
        yearly = df.groupby("Year")["Sales"].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(yearly.index.astype(str), yearly.values, color="#2b6cb0")
        ax.set_title("Total Sales by Year")
        ax.set_ylabel("Sales ($)")
        st.pyplot(fig)

    with col2:
        monthly = df.set_index("Order Date").resample("MS")["Sales"].sum()
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(monthly.index, monthly.values, color="#dd6b20", marker="o", markersize=3)
        ax.set_title("Monthly Sales Trend")
        ax.set_ylabel("Sales ($)")
        st.pyplot(fig)

    st.subheader("Sales by Region & Category")
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        region_filter = st.multiselect("Region", sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
    with fcol2:
        cat_filter = st.multiselect("Category", sorted(df["Category"].unique()), default=sorted(df["Category"].unique()))

    filtered = df[df["Region"].isin(region_filter) & df["Category"].isin(cat_filter)]
    pivot = filtered.groupby(["Region", "Category"])["Sales"].sum().unstack(fill_value=0)
    st.bar_chart(pivot)
    st.dataframe(pivot.style.format("${:,.0f}"))


# ----------------------------------------------------------------------------
# PAGE 2: Forecast Explorer
# ----------------------------------------------------------------------------
elif page == "Forecast Explorer":
    st.title("Forecast Explorer")
    st.caption("Best-performing model from notebook Task 3 comparison: **SARIMA** (lowest RMSE on 3-month backtest).")

    dim = st.selectbox("Select dimension", ["Category", "Region"])
    if dim == "Category":
        options = sorted(df["Category"].unique())
    else:
        options = sorted(df["Region"].unique())
    choice = st.selectbox(f"Select {dim}", options)
    horizon = st.slider("Forecast horizon (months ahead)", 1, 3, 3)

    if dim == "Category":
        series = monthly_series(df, category=choice)
    else:
        series = monthly_series(df, region=choice)

    key = f"{dim}-{choice}"
    pred, ci = fit_sarima_and_forecast(key, series, HORIZON_MAX)
    pred = pred.iloc[:horizon]
    ci = ci.iloc[:horizon]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(series.index[-18:], series.values[-18:], label="Actual", color="#2b6cb0")
    ax.plot(pred.index, pred.values, label="Forecast", color="#dd6b20", marker="o")
    ax.fill_between(ci.index, ci.iloc[:, 0], ci.iloc[:, 1], color="#dd6b20", alpha=0.2, label="95% CI")
    ax.legend()
    ax.set_title(f"{horizon}-Month SARIMA Forecast — {choice} ({dim})")
    st.pyplot(fig)

    mae, rmse = backtest_metrics(key, series, horizon=3)
    m1, m2 = st.columns(2)
    if mae is not None:
        m1.metric("Backtest MAE (last 3 actual months)", f"${mae:,.0f}")
        m2.metric("Backtest RMSE (last 3 actual months)", f"${rmse:,.0f}")
    else:
        st.info("Not enough history in this slice to compute a reliable backtest.")

    st.subheader("Forecast values")
    st.dataframe(pred.rename("Forecasted Sales ($)").to_frame().style.format("${:,.2f}"))


# ----------------------------------------------------------------------------
# PAGE 3: Anomaly Report
# ----------------------------------------------------------------------------
elif page == "Anomaly Report":
    st.title("Anomaly Report")
    feat = weekly_anomaly_table(df)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(feat.index, feat["sales"], color="#2b6cb0", label="Weekly Sales", zorder=1)
    ax.scatter(feat.index[feat["iso_anomaly"]], feat["sales"][feat["iso_anomaly"]],
               color="#e53e3e", label="Isolation Forest", zorder=3, s=50)
    ax.scatter(feat.index[feat["z_anomaly"]], feat["sales"][feat["z_anomaly"]],
               facecolors="none", edgecolors="#dd6b20", label="Z-Score", zorder=2, s=120, linewidths=2, marker="D")
    ax.legend()
    ax.set_title("Weekly Sales with Detected Anomalies")
    st.pyplot(fig)

    st.subheader("Detected anomaly weeks")
    anomalies = feat[feat["is_anomaly"]][["sales", "iso_anomaly", "z_anomaly"]].copy()
    anomalies.columns = ["Sales ($)", "Flagged by Isolation Forest", "Flagged by Z-Score"]
    st.dataframe(anomalies.style.format({"Sales ($)": "${:,.2f}"}))


# ----------------------------------------------------------------------------
# PAGE 4: Product Demand Segments
# ----------------------------------------------------------------------------
elif page == "Product Demand Segments":
    st.title("Product Demand Segments")
    clusters = cluster_subcategories(df)

    fig, ax = plt.subplots(figsize=(8, 6))
    palette = {"High Volume, Stable Demand": "#2b6cb0", "Growing Demand": "#38a169",
               "Declining / Slow-Growth Demand": "#e53e3e", "Low Volume, High Volatility": "#dd6b20"}
    for label, group in clusters.groupby("cluster_label"):
        ax.scatter(group["pc1"], group["pc2"], label=label, s=100,
                   color=palette.get(label, "#805ad5"), edgecolor="white")
    for name, row in clusters.iterrows():
        ax.annotate(name, (row["pc1"], row["pc2"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.legend()
    ax.set_title("Sub-Category Demand Clusters (PCA Projection)")
    st.pyplot(fig)

    st.subheader("Sub-categories by demand cluster")
    table = clusters[["cluster_label", "total_sales_volume", "avg_yoy_growth_pct", "sales_volatility"]].copy()
    table.columns = ["Cluster", "Total Sales ($)", "Avg YoY Growth (%)", "Volatility (CV)"]
    st.dataframe(table.sort_values("Cluster").style.format(
        {"Total Sales ($)": "${:,.0f}", "Avg YoY Growth (%)": "{:.1f}%", "Volatility (CV)": "{:.2f}"}))

    st.subheader("Recommended stocking strategy per cluster")
    for label, strategy in STOCKING_STRATEGY.items():
        if label in table["Cluster"].values:
            st.markdown(f"**{label}** — {strategy}")
