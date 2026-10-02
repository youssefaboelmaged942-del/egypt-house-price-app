import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from catboost import CatBoostRegressor

# 1. إعداد الصفحة بشكل واسع (Wide Layout)
st.set_page_config(page_title="Egypt Real Estate Price Predictor", page_icon="🏠", layout="wide")

# 2. حقن CSS لتصميم حديث
st.markdown("""
<style>
    /* خلفية البطاقات وتحسين الظلال */
    div[data-testid="stForm"] {
        background-color: #1e222d;
        border: 1px solid #2e3545;
        border-radius: 12px;
        padding: 20px;
    }
    /* تحسين شكل بطاقات النشر والنتائج */
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
        color: #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

BASE = Path(__file__).parent


def find_file(name: str) -> Path:
    for folder in (BASE, BASE / "models"):
        p = folder / name
        if p.exists():
            return p
    st.error(f"File not found: {name}. Put it next to app.py (or inside a 'models' folder).")
    st.stop()


@st.cache_resource
def load_artifacts():
    model = CatBoostRegressor()
    model.load_model(str(find_file("catboost_price_model.cbm")))
    meta = json.loads(find_file("model_meta.json").read_text(encoding="utf-8"))
    return model, meta


model, meta = load_artifacts()
opt = meta["options"]
FEATURES = meta["features"]


def build_features(inp: dict) -> pd.DataFrame:
    """Same feature engineering used in training."""
    d = pd.DataFrame([inp])
    beds = d["Bedrooms"].replace(0, np.nan)
    d["Level_num"] = pd.to_numeric(d["Level"].replace({"Ground": 0, "Highest": 12, "10+": 11}), errors="coerce")
    d["Area_per_bedroom"] = d["Area"] / beds
    d["Bath_per_bedroom"] = d["Bathrooms"] / beds
    d["Total_rooms"] = d["Bedrooms"] + d["Bathrooms"]
    d["Has_Compound"] = (~d["Compound"].isin(["Unknown", "Not In Compound"])).astype(int)
    d["Compound_cnt"] = d["Compound"].map(meta["compound_counts"]).fillna(0)
    d["City_cnt"] = d["City"].map(meta["city_counts"]).fillna(0)
    return d[FEATURES]


def egp(x: float) -> str:
    return f"{x:,.0f} EGP"


# ---------------- Header ----------------
st.title("🏠 Egypt Real Estate")
st.caption("AI-powered property valuation and market positioning analytics (CatBoost Regressor).")
st.divider()

# ---------------- Dashboard Layout (2 Columns) ----------------
col_input, col_results = st.columns([1, 1.2], gap="large")

with col_input:
    st.subheader("📋 Property Details")
    with st.form("property_form"):
        c1, c2 = st.columns(2)
        with c1:
            city = st.selectbox("City", opt["City"])
            compound = st.selectbox("Compound", opt["Compound"], help="Choose 'Unknown' if you don't know it, or 'Not In Compound'.")
            area = st.number_input("Area (m²)", min_value=float(meta["area_limits"][0]),
                                   max_value=float(meta["area_limits"][1]), value=150.0, step=5.0)
            delivery = st.selectbox("Delivery term", opt["Delivery_Term"])
        with c2:
            bedrooms = st.number_input("Bedrooms", min_value=0, max_value=15, value=3, step=1)
            bathrooms = st.number_input("Bathrooms", min_value=0, max_value=15, value=2, step=1)
            level = st.selectbox("Level", opt["Level"])
            furnished = st.selectbox("Furnished", opt["Furnished"])
            
        submitted = st.form_submit_button("⚡ Predict & Analyze Price", type="primary", use_container_width=True)

with col_results:
    if submitted:
        inp = {"Bedrooms": int(bedrooms), "Bathrooms": int(bathrooms), "Area": float(area), "Furnished": furnished,
               "Level": level, "Compound": compound, "Delivery_Term": delivery, "City": city}
        X = build_features(inp)
        ppm_pred = float(np.exp(model.predict(X))[0])
        price = ppm_pred * area
        low, high = price * meta["range_ratio"]["p10"], price * meta["range_ratio"]["p90"]

        st.subheader("📊 Price Valuation Results")
        
        # Display Key Metrics
        m1, m2 = st.columns(2)
        m1.metric("Estimated Market Price", egp(price))
        m2.metric("Predicted Price / m²", egp(ppm_pred))

        st.info(f"💡 **Likely Price Range (80% Confidence):** {egp(low)} – {egp(high)}")

        city_ppm = meta["city_median_ppm"].get(city, meta["global_median_ppm"])
        diff_pct = (ppm_pred / city_ppm - 1) * 100
        
        st.write(f"**City Benchmark:** Median in **{city}** is `{egp(city_ppm)}` / m² ({diff_pct:+.1f}% vs city median).")

        if compound == "Unknown" or level == "Unknown" or furnished == "Unknown":
            st.warning("⚠️ Some fields are set to 'Unknown'. Providing exact details sharpens the accuracy.")

        # Plotly Interactive Chart: City Comparison
        st.subheader("🏙️ City Market Benchmark (Price/m²)")
        
        # Prepare comparison data
        top_cities = dict(list(meta["city_median_ppm"].items())[:6])
        top_cities[f"Your Property ({city})"] = ppm_pred
        
        df_chart = pd.DataFrame(list(top_cities.items()), columns=["Location", "Price_per_m2"]).sort_values("Price_per_m2")
        
        fig = px.bar(
            df_chart, 
            x="Price_per_m2", 
            y="Location", 
            orientation="h",
            color="Location",
            title="Comparison with Top Cities Median Price/m²",
            text_auto=".0f"
        )
        fig.update_layout(showlegend=False, height=300, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.subheader("📊 Real-Time Analytics Dashboard")
        st.info("👈 Fill in the property details on the left and click **Predict & Analyze Price** to view the interactive AI report.")

st.divider()

# Footer / Model Info
with st.expander("ℹ️ About the Valuation Model & Data"):
    t = meta["test_metrics"]
    st.write(
        f"- **Algorithm:** CatBoost Regressor trained on Egyptian property market listings.\n"
        f"- **Performance Metrics:** Evaluated on {t['Rows']:,} listings — R² = {t['R2']}, "
        f"Median Absolute Error ≈ {t['MedAPE']}%, and {t['Within30']}% of predictions fall within 30% of market listing prices.\n"
        f"- **Disclaimer:** Estimates reflect market asking prices, not official legal appraisals."
    )
