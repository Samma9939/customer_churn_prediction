import streamlit as st
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/model.pkl")

# Page config
st.set_page_config(page_title="Churn Predictor", layout="wide")

# 🎨 CLEAN UI
st.markdown("""
<style>
.stApp {
    background-color: #f5f7fb;
}
.main-card {
    background-color: white;
    padding: 25px;
    border-radius: 15px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.title {
    font-size: 40px;
    font-weight: bold;
    color: #1f2937;
}
.subtitle {
    font-size: 16px;
    color: #6b7280;
}
.result-box {
    text-align: center;
    padding: 20px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# 🎯 HEADER SECTION WITH IMAGE
col_title, col_img = st.columns([2,1])

with col_title:
    st.markdown('<div class="title">📊 Customer Churn Prediction</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Analyze customer behavior and predict churn risk</div>', unsafe_allow_html=True)

with col_img:
    st.image("assets/dashboard.png", width=250)

st.markdown("<br>", unsafe_allow_html=True)

# Layout
col1, col2 = st.columns(2)

# 🎯 INPUT SECTION
with col1:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("🧾 Customer Details")

    tenure = st.slider("Tenure (months)", 0, 60, 10)
    distance = st.slider("Distance from Warehouse", 0, 50, 20)
    devices = st.slider("Devices Registered", 1, 10, 3)
    satisfaction = st.slider("Satisfaction Score", 1, 5, 3)
    days = st.slider("Days Since Last Order", 0, 30, 5)

    category = st.selectbox("Preferred Category", ["Electronics", "Fashion", "Grocery", "Others"])
    marital = st.selectbox("Marital Status", ["Single", "Married"])
    address = st.slider("Number of Addresses", 1, 10, 2)
    complain = st.selectbox("Complaint", ["No", "Yes"])
    cashback = st.slider("Cashback Amount", 0, 500, 100)

    predict_btn = st.button("🔍 Predict Churn Risk")

    st.markdown('</div>', unsafe_allow_html=True)

# Encoding
category_map = {"Electronics":0, "Fashion":1, "Grocery":2, "Others":3}
marital_map = {"Single":0, "Married":1}
complain_map = {"No":0, "Yes":1}

input_data = np.array([[ 
    tenure, distance, devices,
    category_map[category], satisfaction,
    marital_map[marital], address,
    complain_map[complain], days, cashback
]])

# 🎯 OUTPUT SECTION
with col2:
    st.markdown('<div class="main-card">', unsafe_allow_html=True)
    st.subheader("📈 Prediction Result")

    if predict_btn:
        with st.spinner("Predicting..."):
            result = model.predict(input_data)
            prob = model.predict_proba(input_data)[0]

            churn_prob = prob[1]
            stay_prob = prob[0]

        # RESULT BOX
        if result[0] == 1:
            st.markdown(f"""
            <div class="result-box" style="background-color:#fee2e2;">
                <h3 style="color:#b91c1c;">⚠️ Customer Likely to CHURN</h3>
                <h2 style="color:#b91c1c;">{churn_prob*100:.1f}% Risk</h2>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-box" style="background-color:#dcfce7;">
                <h3 style="color:#166534;">✅ Customer Likely to STAY</h3>
                <h2 style="color:#166534;">{(1-churn_prob)*100:.1f}% Safe</h2>
            </div>
            """, unsafe_allow_html=True)

        st.subheader("Churn Probability")
        st.progress(float(churn_prob))

        # Chart
        df = pd.DataFrame({
            "Outcome": ["Stay", "Churn"],
            "Probability": [stay_prob, churn_prob]
        })

        fig, ax = plt.subplots()
        ax.bar(df["Outcome"], df["Probability"])
        ax.set_ylabel("Probability")
        ax.set_title("Prediction Distribution")

        st.pyplot(fig)

        # Insight
        st.subheader("💡 Insight")
        if churn_prob > 0.6:
            st.warning("High churn risk. Improve engagement.")
        else:
            st.success("Customer likely to stay.")

    else:
        st.info("👈 Enter details and click Predict")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("---")
st.caption("Customer Churn Prediction Dashboard")