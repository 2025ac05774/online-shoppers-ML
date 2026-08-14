import streamlit as st;

st.set_page_config(
    page_title="Online Shopper Conversion Classifier",
    page_icon="🛒",
    layout="wide",
)

st.sidebar.title("🛒 Controls")


st.title("Online Shopper Conversion Classifier")
st.markdown(
    "Predicting whether an e-commerce browsing session ends in a purchase. "
    "Five classifiers trained on the UCI *Online Shoppers Purchasing Intention* dataset."
)