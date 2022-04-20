import streamlit as st
from multiapp import MultiApp
from apps import about, prediction_raw, prediction_enhanced, credits
from PIL import Image

st.markdown(
    "<h1 style='text-align: center; color: green;'>CHEST XRAY Pneumonia Identication</h1>",
    unsafe_allow_html=True,
)
app = MultiApp()
st.sidebar.title("Pneumonia Prediction")
st.sidebar.success("Predict Pneumonia given a Chest X-Ray Image")
app.add_app("About", about.app)
app.add_app("Predict", prediction_raw.app)
app.add_app("Credits", credits.app)
app.run()
