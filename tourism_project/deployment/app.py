# Importing packages
import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib
import io

# App title and description
st.set_page_config(
    page_title="Wellness Tourism Prediction",
    page_icon="🏖️",
    layout="wide"
)

st.title("Wellness Tourism Prediction App")
st.markdown("""
This application predicts whether a customer is likely to purchase a wellness tourism package
based on their demographic, behavioral, and engagement data.
""")

# Sidebar 
with st.sidebar:
    st.header("About This Model")
    st.markdown("""
    **Model Details:**
    - Algorithm: XGBoost Classifier (pipeline with preprocessing)
    - Trained on: Wellness Tourism Dataset
    - Target: Product Taken (1 = Purchased, 0 = Not Purchased)
    **Key Features:**
    - Handles class imbalance with scale_pos_weight
    - Uses preprocessing pipeline (scaling + encoding)
    - Optimized for ROC-AUC score
    """)

    st.subheader("Model Performance")
    st.metric("ROC AUC", "0.9683")
    st.metric("Precision (Class 1)", "0.867")
    st.metric("Recall (Class 1)", "0.818")

# Load Model 
MODEL_REPO_ID = "simnid/wellness-tourism-model"
MODEL_FILENAME = "best_wellness_tourism_model.joblib"

@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILENAME,
            repo_type="model"
        )
        return joblib.load(model_path)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.warning("Model could not be loaded.")
    st.stop()

# --- Customer Input ---
st.header("Customer Information")
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    Age = st.number_input("Age", 18, 80, 35, 1)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", 0, 5, 0)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

with col2:
    st.subheader("Travel Preferences")
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
    Passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    OwnCar = st.selectbox("Owns Car", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    NumberOfTrips = st.number_input("Number of Previous Trips", 0, 20, 2)

with col3:
    st.subheader("Engagement Details")
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", 0.0, 60.0, 15.0, 0.5)
    NumberOfPersonVisiting = st.number_input("Number of People Visiting", 1, 10, 2)
    NumberOfFollowups = st.number_input("Number of Follow-ups", 0, 10, 3)
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 0.0, 5.0, 3.0, 0.1)

# Financial Information 
st.subheader("Financial Information")
col4, col5 = st.columns(2)
with col4:
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    MonthlyIncome = st.number_input("Monthly Income ($)", 1000, 1000000, 15000, 500)

with col5:
    PitchEfficiency = DurationOfPitch * PitchSatisfactionScore
    st.metric("Calculated Pitch Efficiency", f"{PitchEfficiency:.2f}")

# Assemble Input 
input_data = pd.DataFrame([{
    'Age': Age,
    'TypeofContact': TypeofContact,
    'CityTier': CityTier,
    'DurationOfPitch': DurationOfPitch,
    'Occupation': Occupation,
    'Gender': Gender,
    'NumberOfPersonVisiting': NumberOfPersonVisiting,
    'NumberOfFollowups': NumberOfFollowups,
    'ProductPitched': ProductPitched,
    'PreferredPropertyStar': PreferredPropertyStar,
    'MaritalStatus': MaritalStatus,
    'NumberOfTrips': NumberOfTrips,
    'Passport': Passport,
    'PitchSatisfactionScore': PitchSatisfactionScore,
    'OwnCar': OwnCar,
    'NumberOfChildrenVisiting': NumberOfChildrenVisiting,
    'Designation': Designation,
    'MonthlyIncome': MonthlyIncome,
    'PitchEfficiency': PitchEfficiency
}])

with st.expander("View Input Data"):
    st.dataframe(input_data)
    csv = input_data.to_csv(index=False).encode('utf-8')
    st.download_button("Download Input Data", csv, "input_data.csv", "text/csv")

# Prediction 
st.header("Prediction")
if st.button("Predict Purchase Probability", type="primary", use_container_width=True):
    with st.spinner("Making prediction..."):
        try:
            prediction_proba = model.predict_proba(input_data)[0]
            prediction_class = model.predict(input_data)[0]

            col_result1, col_result2 = st.columns(2)
            with col_result1:
                st.subheader("Prediction Result")
                if prediction_class == 1:
                    st.success("Customer is LIKELY to purchase")
                    st.balloons()
                else:
                    st.info("Customer is UNLIKELY to purchase")
            with col_result2:
                st.subheader("Probability Scores")
                st.metric("Probability of Purchase", f"{prediction_proba[1]*100:.1f}%")
                st.metric("Probability of No Purchase", f"{prediction_proba[0]*100:.1f}%")
                st.progress(int(prediction_proba[1]*100))

        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Bulk CSV Prediction 
st.header("Bulk CSV Prediction")
BULK_TEST_FILENAME = "bulk_test_sample.csv"

@st.cache_resource
def load_bulk_sample():
    try:
        path = hf_hub_download(
            repo_id="simnid/wellness-tourism-dataset",
            filename=BULK_TEST_FILENAME,
            repo_type="dataset"
        )
        return pd.read_csv(path)
    except Exception as e:
        st.warning(f"Could not load bulk CSV: {e}")
        return None

bulk_sample = load_bulk_sample()
uploaded_file = st.file_uploader("Upload your CSV for bulk prediction", type=["csv"])
if uploaded_file:
    bulk_sample = pd.read_csv(uploaded_file)

if bulk_sample is not None:
    st.write("Bulk data preview:")
    st.dataframe(bulk_sample.head())
    if st.button("Predict Bulk Probabilities"):
        with st.spinner("Predicting..."):
            try:
                preds_proba = model.predict_proba(bulk_sample)
                preds_class = model.predict(bulk_sample)
                bulk_sample['Probability_Purchase'] = preds_proba[:,1]
                bulk_sample['Prediction'] = preds_class
                st.dataframe(bulk_sample)
                csv_bulk = bulk_sample.to_csv(index=False).encode('utf-8')
                st.download_button("Download Bulk Predictions", csv_bulk, "bulk_predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error predicting bulk data: {e}")

# Footer 
st.markdown("---")
st.caption("Wellness Tourism Prediction Model | Built with XGBoost & Streamlit")
