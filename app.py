"""Streamlit Web Application: Ames House Price Prediction & Real Estate Valuation Tool."""

import os
import sys
import joblib
import numpy as np
import pandas as pd
import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Ames House Price Predictor",
    page_icon="🏡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for modern styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.3rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1E40AF 0%, #3B82F6 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    }
    .metric-val {
        font-size: 2.4rem;
        font-weight: 800;
    }
    .metric-sub {
        font-size: 0.95rem;
        opacity: 0.9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 18px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_trained_pipeline():
    """Load the trained scikit-learn + XGBoost end-to-end pipeline."""
    possible_paths = [
        "models/house_price_pipeline.joblib",
        "e:/aiml/AIml/models/house_price_pipeline.joblib",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            return joblib.load(p)
    return None


@st.cache_data
def load_sample_dataset():
    """Load reference dataset for feature defaults and distribution plots."""
    possible_paths = [
        "data/AmesHousing.csv",
        "AmesHousing.csv",
        "e:/aiml/AIml/data/AmesHousing.csv",
    ]
    for p in possible_paths:
        if os.path.exists(p):
            df = pd.read_csv(p)
            for col in ["Order", "PID"]:
                if col in df.columns:
                    df = df.drop(columns=[col])
            return df
    return None


pipeline = load_trained_pipeline()
df_sample = load_sample_dataset()

# Header
st.markdown('<div class="main-title">🏡 Ames Real Estate Price Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">An end-to-end Machine Learning pipeline trained on 80+ structural property features '
    'using hyperparameter-tuned <b>XGBoost Regressor (R² = 0.918)</b>.</div>',
    unsafe_allow_html=True,
)

if pipeline is None:
    st.error("⚠️ Model artifact not found! Please execute `python src/train.py` first to generate the trained pipeline.")
    st.stop()

# ----------------- SIDEBAR INPUTS -----------------
st.sidebar.header("📐 Property Parameters")
st.sidebar.caption("Adjust key property attributes to estimate market valuation.")

# Extract default template from dataset medians and modes
if df_sample is not None:
    feature_defaults = {}
    for col in df_sample.columns:
        if col == "SalePrice":
            continue
        if df_sample[col].dtype in ["int64", "float64"]:
            feature_defaults[col] = float(df_sample[col].median())
        else:
            mode_val = df_sample[col].mode()
            feature_defaults[col] = str(mode_val[0]) if len(mode_val) > 0 else "Normal"
else:
    feature_defaults = {}

# Major impactful inputs
overall_qual = st.sidebar.slider("Overall Material & Finish Quality (1-10)", min_value=1, max_value=10, value=7,
                                 help="Rates the overall material and finish of the house (1=Very Poor, 10=Very Excellent)")
gr_liv_area = st.sidebar.slider("Above-Grade Living Area (sq ft)", min_value=400, max_value=5000, value=1750, step=25)
total_bsmt_sf = st.sidebar.slider("Total Basement Area (sq ft)", min_value=0, max_value=3500, value=1000, step=25)
garage_cars = st.sidebar.selectbox("Garage Car Capacity", options=[0, 1, 2, 3, 4], index=2)
year_built = st.sidebar.slider("Year Built", min_value=1870, max_value=2020, value=2003)
year_remod = st.sidebar.slider("Year Remodeled / Added", min_value=1950, max_value=2020, value=2005)

st.sidebar.markdown("---")
st.sidebar.subheader("📍 Location & Architecture")

neighborhood_options = [
    "North Ames (NAmes)", "College Creek (CollgCr)", "Old Town (OldTown)",
    "Edwards (Edwards)", "Somerset (Somerst)", "Northridge Heights (NridgHt)",
    "Gilbert (Gilbert)", "Sawyer (Sawyer)", "Northwest Ames (NWAmes)",
    "Sawyer West (SawyerW)", "Mitchel (Mitchel)", "Crawford (Crawfor)",
    "Stone Brook (StoneBr)", "Timberland (Timber)", "Clear Creek (ClearCr)",
]
neighborhood_mapping = {
    "North Ames (NAmes)": "NAmes",
    "College Creek (CollgCr)": "CollgCr",
    "Old Town (OldTown)": "OldTown",
    "Edwards (Edwards)": "Edwards",
    "Somerset (Somerst)": "Somerst",
    "Northridge Heights (NridgHt)": "NridgHt",
    "Gilbert (Gilbert)": "Gilbert",
    "Sawyer (Sawyer)": "Sawyer",
    "Northwest Ames (NWAmes)": "NWAmes",
    "Sawyer West (SawyerW)": "SawyerW",
    "Mitchel (Mitchel)": "Mitchel",
    "Crawford (Crawfor)": "Crawfor",
    "Stone Brook (StoneBr)": "StoneBr",
    "Timberland (Timber)": "Timber",
    "Clear Creek (ClearCr)": "ClearCr",
}
selected_neigh_label = st.sidebar.selectbox("Neighborhood", neighborhood_options, index=1)
selected_neigh = neighborhood_mapping.get(selected_neigh_label, "CollgCr")

house_style = st.sidebar.selectbox("House Style", ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"], index=1)
bsmt_qual = st.sidebar.selectbox("Basement Quality", ["Ex (Excellent)", "Gd (Good)", "TA (Typical)", "Fa (Fair)", "Po (Poor)", "None"], index=1)
kitchen_qual = st.sidebar.selectbox("Kitchen Quality", ["Ex (Excellent)", "Gd (Good)", "TA (Typical)", "Fa (Fair)"], index=1)
full_bath = st.sidebar.selectbox("Full Bathrooms", [1, 2, 3, 4], index=1)

# Build full feature dictionary for prediction
input_record = feature_defaults.copy()
input_record.update({
    "Overall Qual": overall_qual,
    "Gr Liv Area": gr_liv_area,
    "Total Bsmt SF": total_bsmt_sf,
    "Garage Cars": garage_cars,
    "Year Built": year_built,
    "Year Remod/Add": year_remod,
    "Neighborhood": selected_neigh,
    "House Style": house_style,
    "Bsmt Qual": bsmt_qual.split()[0],
    "Kitchen Qual": kitchen_qual.split()[0],
    "Full Bath": full_bath,
    "1st Flr SF": total_bsmt_sf if total_bsmt_sf > 0 else 900,
    "Garage Area": garage_cars * 240,
})

input_df = pd.DataFrame([input_record])

# Run Model Inference
predicted_price = float(pipeline.predict(input_df)[0])
mae_margin = 14812.0  # From tuned XGBoost benchmark

# ----------------- MAIN TABS -----------------
tab1, tab2, tab3 = st.tabs(["📊 Price Valuation", "🔬 Model Benchmark & Insights", "📁 Dataset Insights"])

with tab1:
    col1, col2 = st.columns([1.2, 1])

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-sub">ESTIMATED MARKET VALUATION</div>
            <div class="metric-val">${predicted_price:,.0f}</div>
            <div class="metric-sub">Estimated Range: <b>${max(0, predicted_price - mae_margin):,.0f}</b> – <b>${predicted_price + mae_margin:,.0f}</b> (± MAE: ${mae_margin:,.0f})</div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.subheader("Key Input Summary")
        summary_cols = st.columns(3)
        summary_cols[0].metric("Living Space", f"{gr_liv_area:,} sq ft")
        summary_cols[1].metric("Overall Quality", f"{overall_qual} / 10")
        summary_cols[2].metric("Garage", f"{garage_cars} Cars")

        summary_cols2 = st.columns(3)
        summary_cols2[0].metric("Basement Area", f"{total_bsmt_sf:,} sq ft")
        summary_cols2[1].metric("Year Built", f"{year_built}")
        summary_cols2[2].metric("Neighborhood", f"{selected_neigh}")

    with col2:
        st.subheader("📈 Market Price Comparison")
        if df_sample is not None and "SalePrice" in df_sample.columns:
            median_market = df_sample["SalePrice"].median()
            p75_market = df_sample["SalePrice"].quantile(0.75)
            percentile = (df_sample["SalePrice"] < predicted_price).mean() * 100

            st.write(f"This property is valued higher than **{percentile:.1f}%** of properties in the historical Ames benchmark dataset.")
            st.progress(min(max(float(percentile) / 100.0, 0.0), 1.0))

            compare_df = pd.DataFrame({
                "Category": ["Historical Median", "Historical 75th Percentile", "Your Property Estimate"],
                "Valuation ($)": [median_market, p75_market, predicted_price],
            })
            st.bar_chart(compare_df.set_index("Category"))

with tab2:
    st.subheader("🏆 Model Benchmarking & Architecture")
    st.markdown("""
    The modeling workflow utilizes a scikit-learn **`ColumnTransformer`** to guarantee **zero data leakage**:
    - **Numeric Pipeline**: Missing values imputed with column medians + normalized via `StandardScaler`.
    - **Categorical Pipeline**: Missing attributes flagged as `'Missing'` + encoded into 317 binary indicator columns with `OneHotEncoder(handle_unknown='ignore')`.
    """)

    benchmark_data = {
        "Model Architecture": [
            "Multiple Linear Regression",
            "Random Forest Regressor",
            "XGBoost Regressor (Baseline)",
            "XGBoost Regressor (GridSearchCV Tuned)",
        ],
        "R² Score": [0.8941, 0.9104, 0.9163, 0.9176],
        "MAE ($)": ["$15,710.45", "$15,838.43", "$14,985.63", "$14,812.20"],
        "RMSE ($)": ["$29,132.32", "$26,801.92", "$25,911.63", "$25,699.96"],
        "Cross-Val R² (3-Fold)": ["0.8840", "0.8970", "0.9080", "0.9109"],
    }
    st.dataframe(pd.DataFrame(benchmark_data), use_container_width=True)

    col_fig1, col_fig2 = st.columns(2)
    with col_fig1:
        if os.path.exists("reports/actual_vs_predicted.png"):
            st.image("reports/actual_vs_predicted.png", caption="Actual vs. Predicted Sale Prices (R² = 0.918)")
        elif os.path.exists("e:/aiml/AIml/reports/actual_vs_predicted.png"):
            st.image("e:/aiml/AIml/reports/actual_vs_predicted.png", caption="Actual vs. Predicted Sale Prices")
    with col_fig2:
        if os.path.exists("reports/feature_importance.png"):
            st.image("reports/feature_importance.png", caption="Top 10 Influential Feature Importances (XGBoost)")
        elif os.path.exists("e:/aiml/AIml/reports/feature_importance.png"):
            st.image("e:/aiml/AIml/reports/feature_importance.png", caption="Top 10 Influential Feature Importances")

with tab3:
    st.subheader("📊 Ames Housing Dataset Overview")
    if df_sample is not None:
        st.write(f"The Ames Housing dataset contains **{len(df_sample):,}** residential transactions across 80+ structural and environmental features.")
        st.dataframe(df_sample.head(10), use_container_width=True)

        col_eda1, col_eda2 = st.columns(2)
        with col_eda1:
            if os.path.exists("reports/target_distribution.png"):
                st.image("reports/target_distribution.png", caption="Sale Price Target Distribution")
            elif os.path.exists("e:/aiml/AIml/reports/target_distribution.png"):
                st.image("e:/aiml/AIml/reports/target_distribution.png", caption="Sale Price Target Distribution")
        with col_eda2:
            if os.path.exists("reports/correlation_heatmap.png"):
                st.image("reports/correlation_heatmap.png", caption="Top Numeric Feature Correlations")
            elif os.path.exists("e:/aiml/AIml/reports/correlation_heatmap.png"):
                st.image("e:/aiml/AIml/reports/correlation_heatmap.png", caption="Top Numeric Feature Correlations")
