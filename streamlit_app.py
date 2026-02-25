import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="HDB Resale Price Calculator", page_icon="🏠", layout="centered")

@st.cache_resource
def load_model():
    # Put the joblib in the same folder as app.py, OR change this path
    return joblib.load("final_resale_price_model_7f.joblib")

model = load_model()

st.title("🏠 HDB Resale Price Calculator")
st.write("Enter the 7 features below to get a predicted resale price.")

with st.form("inputs"):
    floor_area_sqm = st.number_input("floor_area_sqm", min_value=10.0, max_value=300.0, value=90.0, step=1.0)
    Hawker_Within_2km = st.number_input("Hawker_Within_2km", min_value=0.0, max_value=200.0, value=8.0, step=1.0)
    mrt_nearest_distance = st.number_input("mrt_nearest_distance (meters)", min_value=0.0, max_value=5000.0, value=350.0, step=10.0)
    Trac_year = st.number_input("Trac_year (transaction year)", min_value=1990, max_value=2035, value=2020, step=1)
    year_completed = st.number_input("year_completed", min_value=1900, max_value=2035, value=2001, step=1)
    mid_storey = st.number_input("mid_storey", min_value=1, max_value=60, value=10, step=1)

    # Adjust options to match your training data categories if needed
    flat_type = st.selectbox(
        "flat_type",
        options=["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"],
        index=3
    )

    submitted = st.form_submit_button("Predict")

if submitted:
    # Build a single-row DataFrame with EXACT 7 inputs
    # Note: model pipeline maps Trac_year -> Tranc_Year internally (from your training script)
    X = pd.DataFrame([{
        "floor_area_sqm": float(floor_area_sqm),
        "Hawker_Within_2km": float(Hawker_Within_2km),
        "mrt_nearest_distance": float(mrt_nearest_distance),
        "Trac_year": int(Trac_year),
        "year_completed": int(year_completed),
        "mid_storey": int(mid_storey),
        "flat_type": str(flat_type),
    }])

    pred = float(model.predict(X)[0])

    st.success(f"Estimated resale price: **${pred:,.0f}**")
    st.caption("This is a model estimate; actual market prices may vary.")
