import streamlit as st
import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download
import joblib

# App title and description
st.set_page_config(
    page_title="Wellness Tourism Prediction",
    page_icon="🏖️",
    layout="wide"
)

st.title("🏖️ Wellness Tourism Prediction App")
st.markdown("""
This application predicts whether a customer is likely to purchase a wellness tourism package
based on their demographic, behavioral, and engagement data.
""")

# Sidebar for information
with st.sidebar:
    st.header("About This Model")
    st.markdown("""
    **Model Details:**
    - Algorithm: XGBoost Classifier
    - Trained on: Wellness Tourism Dataset
    - Target: Product Taken (1 = Purchased, 0 = Not Purchased)

    **Key Features:**
    - Handles class imbalance with scale_pos_weight
    - Uses preprocessing pipeline (scaling + encoding)
    - Optimized for ROC-AUC score
    """)

    # Display model metrics from your training
    st.subheader("Model Performance")
    st.metric("ROC AUC", "0.9414")
    st.metric("Precision (Class 1)", "0.69")
    st.metric("Recall (Class 1)", "0.79")

# Function to download and load model
@st.cache_resource
def load_model():
    """Load the trained model from Hugging Face Hub"""
    try:
        model_path = hf_hub_download(
            repo_id="simnid/wellness-tourism-model",
            filename="best_wellness_tourism_model.joblib"
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

# Load the model
model = load_model()

if model is None:
    st.warning("Model could not be loaded. Please check your connection.")
    st.stop()

# User input section
st.header("📋 Customer Information")

# Create columns for better layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographic Information")
    Age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0, step=1)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

with col2:
    st.subheader("Travel Preferences")
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
    Passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    OwnCar = st.selectbox("Owns Car", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    NumberOfTrips = st.number_input("Number of Previous Trips", min_value=0, max_value=20, value=2, step=1)

with col3:
    st.subheader("Engagement Details")
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=15.0, step=0.5)
    NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=2, step=1)
    NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3, step=1)
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

# Additional inputs
st.subheader("Financial Information")
col4, col5 = st.columns(2)

with col4:
    Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
    MonthlyIncome = st.number_input("Monthly Income ($)", min_value=1000, max_value=50000, value=15000, step=500)

with col5:
    # Calculate Pitch Efficiency (feature from your preprocessing)
    PitchEfficiency = DurationOfPitch * PitchSatisfactionScore
    st.metric("Calculated Pitch Efficiency", f"{PitchEfficiency:.2f}")

# Assemble input into DataFrame
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

# Display the input data
with st.expander("View Input Data"):
    st.dataframe(input_data)

# Prediction section
st.header("🎯 Prediction")

if st.button("Predict Purchase Probability", type="primary", use_container_width=True):
    with st.spinner("Making prediction..."):
        try:
            # Make prediction
            prediction_proba = model.predict_proba(input_data)[0]
            prediction_class = model.predict(input_data)[0]

            # Display results
            col_result1, col_result2 = st.columns(2)

            with col_result1:
                st.subheader("Prediction Result")
                if prediction_class == 1:
                    st.success("✅ **Customer is LIKELY to purchase**")
                    st.balloons()
                else:
                    st.info("❌ **Customer is UNLIKELY to purchase**")

            with col_result2:
                st.subheader("Probability Scores")
                # Create gauge-like visualization
                prob_purchase = prediction_proba[1] * 100
                prob_no_purchase = prediction_proba[0] * 100

                st.metric("Probability of Purchase", f"{prob_purchase:.1f}%")
                st.metric("Probability of No Purchase", f"{prob_no_purchase:.1f}%")

                # Visual progress bar
                st.progress(int(prob_purchase))
                st.caption(f"Confidence: {prob_purchase:.1f}%")

            # Business insights
            st.subheader("📊 Business Insights")

            if prediction_class == 1:
                if prob_purchase > 80:
                    st.success("**High Confidence Lead** - Consider offering premium packages")
                elif prob_purchase > 60:
                    st.warning("**Medium Confidence Lead** - Standard follow-up recommended")
                else:
                    st.info("**Low Confidence Lead** - May require additional engagement")

                st.markdown("""
                **Recommended Actions:**
                - Schedule follow-up call within 48 hours
                - Offer personalized package options
                - Highlight wellness benefits specific to customer profile
                """)
            else:
                st.markdown("""
                **Recommended Actions:**
                - Consider re-engagement in 3-6 months
                - Collect feedback on pitch satisfaction
                - Update marketing materials for similar profiles
                """)

        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Model information
with st.expander("ℹ️ Model Information"):
    st.markdown("""
    **Model Architecture:**
    - Preprocessing: StandardScaler for numeric features + OneHotEncoder for categorical features
    - Algorithm: XGBoost Classifier
    - Hyperparameters from grid search:
        - n_estimators: 200
        - max_depth: 7
        - learning_rate: 0.1
        - colsample_bytree: 0.6
        - reg_lambda: 0.5

    **Training Performance:**
    - ROC AUC: 0.9414
    - PR AUC: 0.8344
    - Test Accuracy: 0.8898
    - Precision (Class 1): 0.69
    - Recall (Class 1): 0.79

    **Note:** Class 1 represents customers who purchased the wellness tourism package.
    """)

# Footer
st.markdown("---")
st.caption("Wellness Tourism Prediction Model | Built with XGBoost & Streamlit")
