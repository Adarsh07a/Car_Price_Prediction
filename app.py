import streamlit as st
import numpy as np
import pandas as pd
import pickle
from joblib import dump
from joblib import load


# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="🚗 Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# -------------------- LOAD MODEL --------------------
model = load("ridge_model.pkl")
scaler = load("scaler.pkl")

# -------------------- CUSTOM CSS --------------------
st.markdown("""
    <style>
    .main {
        background: linear-gradient(135deg, #1f4037, #99f2c8);
    }
    .title {
        font-size: 45px;
        font-weight: bold;
        color: white;
        text-align: center;
    }
    .subtitle {
        font-size: 18px;
        color: #f0f0f0;
        text-align: center;
        margin-bottom: 30px;
    }
    .stButton>button {
        background-color: #ff4b2b;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px 24px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------- TITLE --------------------
st.markdown('<p class="title">🚗 Car Price Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Predict the price of a car using Machine Learning</p>', unsafe_allow_html=True)

# -------------------- SIDEBAR INPUTS --------------------
st.sidebar.header("🔧 Enter Car Details")

symboling = st.sidebar.slider("Symboling", -3, 3, 0)

fueltype = st.sidebar.selectbox("Fuel Type", ["gas", "diesel"])
aspiration = st.sidebar.selectbox("Aspiration", ["std", "turbo"])
doornumber = st.sidebar.selectbox("Doors", ["two", "four"])
carbody = st.sidebar.selectbox("Car Body", ["sedan", "hatchback", "wagon", "hardtop", "convertible"])
drivewheel = st.sidebar.selectbox("Drive Wheel", ["fwd", "rwd", "4wd"])
enginelocation = st.sidebar.selectbox("Engine Location", ["front", "rear"])

wheelbase = st.sidebar.slider("Wheelbase", 80.0, 120.0, 90.0)
carlength = st.sidebar.slider("Car Length", 140.0, 210.0, 170.0)
carwidth = st.sidebar.slider("Car Width", 60.0, 80.0, 65.0)
carheight = st.sidebar.slider("Car Height", 45.0, 60.0, 50.0)
curbweight = st.sidebar.slider("Curb Weight", 1500, 4000, 2500)

enginetype = st.sidebar.selectbox("Engine Type", ["ohc", "ohcf", "ohcv", "dohc", "l", "rotor"])
cylindernumber = st.sidebar.selectbox("Cylinders", ["two", "three", "four", "five", "six", "eight", "twelve"])
enginesize = st.sidebar.slider("Engine Size", 60, 300, 120)

# -------------------- DATA PREPROCESS --------------------
input_dict = {
    "symboling": symboling,
    "wheelbase": wheelbase,
    "carlength": carlength,
    "carwidth": carwidth,
    "carheight": carheight,
    "curbweight": curbweight,
    "enginesize": enginesize,
}

df = pd.DataFrame([input_dict])

# -------------------- ONE HOT ENCODING --------------------
categorical_cols = {
    "fueltype": fueltype,
    "aspiration": aspiration,
    "doornumber": doornumber,
    "carbody": carbody,
    "drivewheel": drivewheel,
    "enginelocation": enginelocation,
    "enginetype": enginetype,
    "cylindernumber": cylindernumber
}

for col, val in categorical_cols.items():
    df[col + "_" + val] = 1

# -------------------- FIX FEATURE MISMATCH --------------------
# IMPORTANT: must match training features
expected_features = scaler.feature_names_in_

for col in expected_features:
    if col not in df.columns:
        df[col] = 0

df = df[expected_features]

# -------------------- SCALE --------------------
scaled_input = scaler.transform(df)

# -------------------- PREDICT --------------------
if st.button("🚀 Predict Price"):
    prediction = model.predict(scaled_input)[0]

    st.success(f"💰 Estimated Car Price: ₹ {round(prediction, 2)}")

    st.markdown("### 📊 Prediction Insights")
    st.write("This prediction is based on engine size, weight, and other specifications you provided.")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("Built with ❤️ using Streamlit | ML Project")