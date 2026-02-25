import os
import joblib
import requests
import pandas as pd
import streamlit as st

st.set_page_config(page_title="HDB Resale Price Calculator", page_icon="🏠")

MODEL_URL = "https://github.com/emmabiee/HDB_resale_price_predictor/releases/download/v1.1/final_resale_price_model_7f.joblib"
MODEL_PATH = "final_resale_price_model_7f.joblib"


def download_model():
    if os.path.exists(MODEL_PATH):
        return
    
    with st.spinner("Downloading model (first run only)..."):
        response = requests.get(MODEL_URL, stream=True)
        response.raise_for_status()
        
        with open(MODEL_PATH, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


@st.cache_resource
def load_model():
    download_model()
    return joblib.load(MODEL_PATH)


model = load_model()

st.title("🏠 HDB Resale Price Calculator")
st.write("Input the 7 features to predict resale price")

floor_area_sqm = st.number_input("floor_area_sqm", 10.0, 300.0, 90.0)
Hawker_Within_2km = st.number_input("Hawker_Within_2km", 0, 50, 5)
mrt_nearest_distance = st.number_input("mrt_nearest_distance", 0.0, 20000.0, 500.0)
Trac_year = st.number_input("Trac_year", 1990, 2035, 2020)
year_completed = st.number_input("year_completed", 1900, 2035, 2000)
mid_storey = st.number_input("mid_storey", 1, 60, 10)

flat_type = st.selectbox(
    "flat_type",
    ["1 ROOM", "2 ROOM", "3 ROOM", "4 ROOM", "5 ROOM", "EXECUTIVE", "MULTI-GENERATION"]
)

if st.button("Predict Resale Price"):
    input_df = pd.DataFrame([{
        "floor_area_sqm": floor_area_sqm,
        "Hawker_Within_2km": Hawker_Within_2km,
        "mrt_nearest_distance": mrt_nearest_distance,
        "Trac_year": Trac_year,
        "year_completed": year_completed,
        "mid_storey": mid_storey,
        "flat_type": flat_type
    }])

    prediction = model.predict(input_df)[0]
    st.success(f"Estimated Resale Price: ${prediction:,.0f}")
