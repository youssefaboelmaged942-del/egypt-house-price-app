import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from catboost import CatBoostRegressor

# 1. إعداد الصفحة
st.set_page_config(page_title="Egypt Real Estate Price Predictor", page_icon="🏠", layout="wide")

# 2. حقن CSS للتوسيط الرأسي وتحسين وضوح النصوص والخلفيات
st.markdown(
    """
<style>
    /* توسيط المحتوى رأسياً في منتصف الشاشة */
    .main .block-container {
        max-width: 95% !important;
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
    }

    /* الخلفية الرئيسية */
    .stApp {
        background: linear-gradient(rgba(0, 0, 0, 0.6), rgba(0, 0, 0, 0.6)), 
                    url("https://images.unsplash.com/photo-1512917774080-9991f1c4c750?q=80&w=1920&auto=format&fit=crop") !important;
        background-size: cover !important;
        background-position: center !important;
        background-attachment: fixed !important;
    }

    /* تحويل كل النصوص للون الأبيض الناصع */
    h1, h2, h3, h4, h5, h6, p, span, label, div, .stCaption {
        color: #ffffff !important;
    }

    /* خلفية داكنة واضحة للـ Form */
    div[data-testid="stForm"] {
        background-color: rgba(15, 23, 42, 0.92) !important;
        border: 1px solid rgba(255, 255, 255, 0.25) !important;
        border-radius: 12px;
        padding: 16px 20px !important;
        box-shadow: 0 10px 25px rgba(0,0,0,0.5);
    }

    /* خلفية داكنة لكروت النتائج */
    div[data-testid="stMetric"] {
        background-color: rgba(15, 23, 42, 0.92) !important;
        border-radius: 10px;
        padding: 10px 14px !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        color: #38bdf8 !important; /* لون أزرق فاتح مميز للأرقام */
    }

    /* خلفية صندوق الـ Info والتوضيحات */
    .stAlert {
        background-color: rgba(15, 23, 42, 0.9) !important;
        color: #ffffff !important;
        border: 1px solid rgba(56, 189, 248, 0.4) !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

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
st.title("🏠  Egypt Real Estate Price Predictor ")

# ---------------- Dashboard Layout ----------------
col_input, col_results = st.columns([1, 1.15], gap="large")

with col_input:
    st.markdown("### 📋 Property Details")
    with st.form("property_form"):
        c1, c2 = st.columns(2)
        with c1:
            city = st.selectbox("City", opt["City"])
            compound = st.selectbox("Compound", opt["Compound"])
            area = st.number_input("Area (m²)", min_value=float(meta["area_limits"][0]),
                                   max_value=float(meta["area_limits"][1]), value=205.0, step=5.0)
            delivery = st.selectbox("Delivery term", opt["Delivery_Term"])
        with c2:
            bedrooms = st.number_input("Bedrooms", min_value=0, max_value=15, value=3, step=1)
            bathrooms = st.number_input("Bathrooms", min_value=0, max_value=15, value=2, step=1)
            level = st.selectbox("Level", opt["Level"])
            furnished = st.selectbox("Furnished", opt["Furnished"])
            
        submitted = st.form_submit_button("⚡ Predict & Analyze Price", type="primary", use_container_width=True)

    with st.expander("ℹ️ About Model"):
        t = meta["test_metrics"]
        st.write(f"CatBoost Regressor | R² = {t['R2']} | MedAPE ≈ {t['MedAPE']}%")

with col_results:
    st.markdown("### 📊 Price Valuation Results")
    if submitted:
        inp = {"Bedrooms": int(bedrooms), "Bathrooms": int(bathrooms), "Area": float(area), "Furnished": furnished,
               "Level": level, "Compound": compound, "Delivery_Term": delivery, "City": city}
        X = build_features(inp)
        ppm_pred = float(np.exp(model.predict(X))[0])
        price = ppm_pred * area
        low, high = price * meta["range_ratio"]["p10"], price * meta["range_ratio"]["p90"]

        # Display Key Metrics
        m1, m2 = st.columns(2)
        m1.metric("Estimated Market Price", egp(price))
        m2.metric("Predicted Price / m²", egp(ppm_pred))

        st.info(f"💡 **Likely Range:** {egp(low)} – {egp(high)}")

        city_ppm = meta["city_median_ppm"].get(city, meta["global_median_ppm"])
        diff_pct = (ppm_pred / city_ppm - 1) * 100
        
        st.markdown(f"📍 **City Benchmark:** Median in **{city}** is `{egp(city_ppm)}` / m² ({diff_pct:+.1f}% vs city median).")

        # Plotly Chart مع إعطائه خلفية داكنة وتحسين خطوطه
        top_cities = dict(list(meta["city_median_ppm"].items())[:5])
        top_cities[f"Your Property ({city})"] = ppm_pred
        
        df_chart = pd.DataFrame(list(top_cities.items()), columns=["Location", "Price_per_m2"]).sort_values("Price_per_m2")
        
        fig = px.bar(
            df_chart, 
            x="Price_per_m2", 
            y="Location", 
            orientation="h",
            color="Location",
            title="Price/m² Comparison",
            text_auto=".0f"
        )
        
        fig.update_layout(
            showlegend=False, 
            height=260, 
            margin=dict(l=10, r=20, t=35, b=10), 
            paper_bgcolor="rgba(15, 23, 42, 0.92)", 
            plot_bgcolor="rgba(15, 23, 42, 0.92)",
            font=dict(color="#ffffff", size=12),
            title_font=dict(color="#ffffff", size=14)
        )
        fig.update_xaxes(title="", tickfont=dict(color="#ffffff"), gridcolor="rgba(255,255,255,0.1)")
        fig.update_yaxes(title="", tickfont=dict(color="#ffffff"))
        
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.info("👈 Fill in the property details on the left and click **Predict & Analyze Price** to view the interactive AI report.")
