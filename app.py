import streamlit as st
import pandas as pd
import joblib

# ================= PAGE CONFIG =================
st.set_page_config(
    page_title="AI-Based Water Monitoring System",
    layout="wide"
)

st.title("💧 AI-Based Water Quality & Leak Detection Dashboard")

# ================= LOAD MODELS =================
leak_model = joblib.load("leak_model.pkl")
water_model = joblib.load("water_quality_model.pkl")

# ================= SIDEBAR INPUTS =================
st.sidebar.header("🔧 Sensor Inputs")

ph = st.sidebar.slider("pH", 0.0, 14.0, 7.0)
hardness = st.sidebar.slider("Hardness", 0.0, 400.0, 150.0)
solids = st.sidebar.slider("Solids", 0.0, 50000.0, 10000.0)
conductivity = st.sidebar.slider("Conductivity", 0.0, 1000.0, 400.0)
turbidity = st.sidebar.slider("Turbidity", 0.0, 10.0, 3.0)

pressure = st.sidebar.slider("Pressure (bar)", 0.0, 100.0, 55.0)
flow = st.sidebar.slider("Flow Rate (L/s)", 0.0, 100.0, 30.0)
temp = st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0)

pressure_flow_ratio = pressure / (flow + 0.01)

features = [
    'ph','Hardness','Solids','Conductivity','Turbidity',
    'Pressure_(bar)','Flow_Rate_(L/s)','Temperature_(°C)',
    'Pressure_Flow_Ratio'
]

input_data = pd.DataFrame([[ph, hardness, solids, conductivity, turbidity,
                            pressure, flow, temp, pressure_flow_ratio]],
                          columns=features)

# ================= PREDICTIONS =================
leak_pred = leak_model.predict(input_data)[0]
water_pred = water_model.predict(input_data)[0]

# ================= OUTPUT =================
col1, col2 = st.columns(2)

with col1:
    st.subheader("🚰 Leak Detection Status")
    if leak_pred == 1:
        st.error("🚨 Leak Detected")
    else:
        st.success("✅ No Leak Detected")
        
safe_ranges = {
    "pH": (6.5, 8.5),
    "Hardness": (0, 400),
    "Solids": (0, 50000),
    "Conductivity": (0, 1000),
    "Turbidity": (0, 5),
    "Pressure_(bar)":(40 ,80)
}
input_dict = {
    "pH": ph,
    "Hardness": hardness,
    "Solids": solids,
    "Conductivity": conductivity,
    "Turbidity": turbidity,
    "Pressure_(bar)": pressure
}
reasons = []

for feature, value in input_dict.items():
    low, high = safe_ranges[feature]
    if value < low:
        reasons.append(f"{feature} is too LOW ({value})")
    elif value > high:
        reasons.append(f"{feature} is too HIGH ({value})")



with col2:
    st.subheader("💧 Water Quality Status")
    if water_pred == 0 and 6.5 <= ph <= 8.5:
        st.warning("⚠️ pH is safe, but other parameters are outside safe limits")
        if reasons:
            st.subheader("Reason(s) why water is unsafe:")
        for r in reasons:
            st.write(f"- {r}")
    elif water_pred == 1:
        st.success("✅ Water is safe for drinking")
    else:
        st.error("❌ Water is unsafe")

# ================= INSIGHTS =================
st.markdown("---")
st.subheader("📊 Smart Insights")

if ph < 6.5 or ph > 8.5:
    st.warning("⚠️ pH level outside safe drinking range")

st.info("🤖 Predictions are generated using Machine Learning models trained on historical sensor data.")

st.write("Input values used for prediction:")
st.dataframe(input_data)

st.subheader("🤖 Why this prediction?")

water_importance = water_model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": water_model.feature_importances_
}).sort_values(by="Importance", ascending=False)

st.table(importance_df)


st.markdown("---")
st.subheader("🔍 Feature Importance (Leak Detection Model)")

importance = leak_model.feature_importances_

importance_df = pd.DataFrame({
    "Feature": features,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.bar_chart(importance_df.set_index("Feature"))

st.subheader("🔍 Feature Importance (Water Quality Model)")

water_importance = water_model.feature_importances_

water_df = pd.DataFrame({
    "Feature": features,
    "Importance": water_importance
}).sort_values(by="Importance", ascending=False)

st.bar_chart(water_df.set_index("Feature"))

safe_ranges = {
    "pH": (6.5, 8.5),
    "Hardness": (0, 400),
    "Solids": (0, 50000),
    "Conductivity": (0, 1000),
    "Turbidity": (0, 5),
    "Pressure_(bar)":(40 ,80)
}
input_dict = {
    "pH": ph,
    "Hardness": hardness,
    "Solids": solids,
    "Conductivity": conductivity,
    "Turbidity": turbidity,
    "Pressure_(bar)": pressure
}
reasons = []

for feature, value in input_dict.items():
    low, high = safe_ranges[feature]
    if value < low:
        reasons.append(f"{feature} is too LOW ({value})")
    elif value > high:
        reasons.append(f"{feature} is too HIGH ({value})")

