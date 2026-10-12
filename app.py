
import streamlit as st
import pandas as pd
import pickle

# Page configuration
st.set_page_config(
    page_title="Diamond Price Prediction",
    page_icon="💎",
    layout="centered"
)

# Load the trained pipeline
@st.cache_resource
def load_model():
    with open("knn_pipeline.pkl", "rb") as file:
        return pickle.load(file)

model = load_model()

# Application title
st.title("💎 Diamond Price Prediction")
st.write(
    "Enter the diamond details to estimate its price "
    "using a KNN Regression model."
)

# User inputs
carat = st.number_input(
    "Carat", min_value=0.1, max_value=10.0,
    value=1.0, step=0.1
)

cut = st.selectbox(
    "Cut",
    ["Fair", "Good", "Very Good", "Premium", "Ideal"]
)

color = st.selectbox(
    "Color",
    ["D", "E", "F", "G", "H", "I", "J"]
)

clarity = st.selectbox(
    "Clarity",
    ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]
)

depth = st.number_input(
    "Depth (%)", min_value=40.0, max_value=80.0,
    value=61.5, step=0.1
)

table = st.number_input(
    "Table (%)", min_value=40.0, max_value=100.0,
    value=57.0, step=0.1
)

x = st.number_input(
    "Length (x, mm)", min_value=0.0, max_value=15.0,
    value=6.0, step=0.1
)

y = st.number_input(
    "Width (y, mm)", min_value=0.0, max_value=15.0,
    value=6.0, step=0.1
)

z = st.number_input(
    "Depth (z, mm)", min_value=0.0, max_value=15.0,
    value=3.7, step=0.1
)

# Predict button
if st.button("Predict Diamond Price"):
    input_data = pd.DataFrame([{
        "carat": carat,
        "cut": cut,
        "color": color,
        "clarity": clarity,
        "depth": depth,
        "table": table,
        "x": x,
        "y": y,
        "z": z
    }])

    prediction = model.predict(input_data)[0]

    st.success(
        f"Estimated Diamond Price: ${prediction:,.2f}"
    )
