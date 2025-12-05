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
    - Algorithm: XGBoost Classifier (pipeline with preprocessing)
    - Trained on: Wellness Tourism Dataset
    - Target: Product Taken (1 = Purchased, 0 = Not Purchased)
    """)
    st.subheader("Model Performance (example)")
    st.metric("ROC AUC", "0.94")
    st.metric("Precision (Class 1)", "0.69")
    st.metric("Recall (Class 1)", "0.79")

# Function to download and load model (pipeline)
@st.cache_resource
def load_model():
    """Load the trained pipeline from Hugging Face Hub"""
    try:
        model_path = hf_hub_download(
            repo_id="simnid/wellness-tourism-model",
            filename="best_wellness_tourism_model.joblib",
            repo_type="model"
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_model()
if model is None:
    st.warning("Model could not be loaded. Please check your connection.")
    st.stop()

# Attempt to infer expected input columns from the trained pipeline
def get_expected_input_columns(model):
    try:
        # If ColumnTransformer was used as first step in pipeline with name 'preprocessor'
        if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
            pre = model.named_steps["preprocessor"]
            cols = []
            for transformer in pre.transformers_:
                name, trans, cols_list = transformer
                # cols_list may be a slice or list
                if isinstance(cols_list, (list, tuple)):
                    cols.extend(list(cols_list))
                else:
                    try:
                        cols.extend(list(cols_list))
                    except Exception:
                        pass
            return cols
    except Exception:
        pass
    # Fallback: define expected columns explicitly
    return [
        'Age','TypeofContact','CityTier','DurationOfPitch','Occupation','Gender',
        'NumberOfPersonVisiting','NumberOfFollowups','ProductPitched','PreferredPropertyStar',
        'MaritalStatus','NumberOfTrips','Passport','PitchSatisfactionScore','OwnCar',
        'NumberOfChildrenVisiting','Designation','MonthlyIncome','PitchEfficiency'
    ]

expected_cols = get_expected_input_columns(model)

# User input section
st.header("Customer Information")
col1, col2, col3 = st.columns(3)

with col1:
    Age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
    Gender = st.selectbox("Gender", ["Male", "Female"])
    MaritalStatus = st.selectbox("Marital Status", ["Single", "Married", "Divorced", "Unmarried"])
    NumberOfChildrenVisiting = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0, step=1)
    Designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])

with col2:
    CityTier = st.selectbox("City Tier", [1, 2, 3])
    PreferredPropertyStar = st.selectbox("Preferred Property Star Rating", [3, 4, 5])
    Passport = st.selectbox("Has Passport", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    OwnCar = st.selectbox("Owns Car", [0, 1], format_func=lambda x: "No" if x == 0 else "Yes")
    NumberOfTrips = st.number_input("Number of Previous Trips", min_value=0, max_value=20, value=2, step=1)

with col3:
    TypeofContact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    DurationOfPitch = st.number_input("Duration of Pitch (minutes)", min_value=0.0, max_value=60.0, value=15.0, step=0.5)
    NumberOfPersonVisiting = st.number_input("Number of People Visiting", min_value=1, max_value=10, value=2, step=1)
    NumberOfFollowups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3, step=1)
    ProductPitched = st.selectbox("Product Pitched", ["Basic", "Deluxe", "Standard", "Super Deluxe", "King"])
    PitchSatisfactionScore = st.slider("Pitch Satisfaction Score", 1, 5, 3)

# Financial info & derived feature
PitchEfficiency = DurationOfPitch * PitchSatisfactionScore
st.metric("Calculated Pitch Efficiency", f"{PitchEfficiency:.2f}")

Occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
MonthlyIncome = st.number_input("Monthly Income ($)", min_value=1000, max_value=50000, value=15000, step=500)

# Assemble input as raw (strings preserved)
input_row = {
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
}

input_data = pd.DataFrame([input_row])

try:
    cols_to_use = [c for c in expected_cols if c in input_data.columns]
    input_data = input_data[cols_to_use]
except Exception:
    pass

with st.expander("View Input Data"):
    st.dataframe(input_data)

# Prediction
st.header("Prediction")
if st.button("Predict Purchase Probability", type="primary", use_container_width=True):
    with st.spinner("Making prediction..."):
        try:
            # model is a pipeline that includes preprocessing
            prediction_proba = model.predict_proba(input_data)[0]
            prediction_class = int(model.predict(input_data)[0])

            prob_purchase = float(prediction_proba[1] * 100)
            prob_no_purchase = float(prediction_proba[0] * 100)

            col_result1, col_result2 = st.columns(2)
            with col_result1:
                st.subheader("Prediction Result")
                if prediction_class == 1:
                    st.success("**Customer is LIKELY to purchase**")
                    st.balloons()
                else:
                    st.info("**Customer is UNLIKELY to purchase**")

            with col_result2:
                st.subheader("Probability Scores")
                st.metric("Probability of Purchase", f"{prob_purchase:.1f}%")
                st.metric("Probability of No Purchase", f"{prob_no_purchase:.1f}%")
                st.progress(int(min(max(prob_purchase, 0), 100)))
                st.caption(f"Confidence: {prob_purchase:.1f}%")

            # Business insights...
        except Exception as e:
            st.error(f"Error making prediction: {e}")

# Footer
st.markdown("---")
st.caption("Wellness Tourism Prediction Model | Built with XGBoost & Streamlit")
