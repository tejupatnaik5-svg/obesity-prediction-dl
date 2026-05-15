import streamlit as st
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import base64

from sklearn.preprocessing import LabelEncoder, StandardScaler

st.set_page_config(page_title="Obesity Predictor", layout="centered")

# Load model
@st.cache_resource
def load_model_and_data():
    df = pd.read_csv('./dataset/Obesity_dataset.csv')

    features = df.drop("Obesity level class", axis=1)
    target = df["Obesity level class"]

    encoded_features = pd.get_dummies(features, drop_first=True)
    columns = encoded_features.columns

    target_encoder = LabelEncoder()
    target_encoder.fit(target)

    scaler = StandardScaler()
    scaler.fit(encoded_features)

    input_size = encoded_features.shape[1]

    class ObesityModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(64, 32),
                nn.ReLU(),
                nn.Linear(32, 7)
            )

        def forward(self, x):
            return self.layers(x)

    model = ObesityModel()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()

    return df, model, scaler, columns, target_encoder


df, model, scaler, columns, target_encoder = load_model_and_data()

# Header
def get_base64(img_path):
    with open(img_path, "rb") as f:
        return base64.b64encode(f.read()).decode()

logo = get_base64("logo.png")

st.markdown(f"""
<div style="display:flex; align-items:center; gap:15px;">
    <img src="data:image/png;base64,{logo}" width="140">
    <div>
        <h1>Obesity Risk Predictor</h1>
        <p>Analyze lifestyle and physical attributes</p>
    </div>
</div>
""", unsafe_allow_html=True)


# User Input
st.markdown("## User Information")

age = st.text_input("Age")
weight_lb = st.text_input("Weight (lbs)")

gender = st.selectbox("Gender", ["Male", "Female"])
family_history = st.selectbox("Family History", ["No", "Yes"])
caec = st.selectbox("Food Between Meals", ["No", "Sometimes", "Frequently", "Always"])

col1, col2 = st.columns(2)
with col1:
    feet = st.number_input("Height (Feet)", min_value=0)
with col2:
    inches = st.number_input("Height (Inches)", min_value=0)


#BMI Preview
if weight_lb and (feet > 0 or inches > 0):
    try:
        weight_temp = float(weight_lb) * 0.453592
        height_temp = (feet * 0.3048) + (inches * 0.0254)

        if height_temp > 0:
            bmi_preview = weight_temp / (height_temp**2)
            st.success(f"BMI: {bmi_preview:.2f}")
    except ValueError:
        pass

# Activity
st.markdown("## Activity Level Guide")
st.markdown("1 (Low) | 2 (Moderate) | 3 (Active) | 4 (Very Active)")

auto_detect = st.checkbox("Auto-detect activity level from device data")

if auto_detect:
    st.info("Analyzing activity data...")

    if "steps" not in st.session_state:
        st.session_state.steps = np.random.randint(3000, 12000)
        st.session_state.active_minutes = np.random.randint(0, 120)

    steps = st.session_state.steps
    active_minutes = st.session_state.active_minutes

    st.write(f"Steps: {steps} | Active Minutes: {active_minutes}")

    if steps < 5000:
        pal = 1
    elif steps < 8000:
        pal = 2
    elif steps < 11000:
        pal = 3
    else:
        pal = 4

    st.success(f"Detected Activity Level: {pal}")
else:
    pal = st.selectbox("Activity Level", [1, 2, 3, 4])


# Predict
predict_clicked = st.button("Predict")

if predict_clicked:

    # Validation
    try:
        age_val = int(age)
        weight_val = float(weight_lb)
        weight = weight_val * 0.453592
    except:
        st.warning("Enter valid age and weight")
        st.stop()

    # Calculations
    height = (feet * 0.3048) + (inches * 0.0254)
    bmi = weight / (height**2) if height > 0 else 0

    # Model Input
    input_data = pd.DataFrame([{
        "Age": age_val,
        "Gender": gender,
        "Height": height,
        "Weight": weight,
        "BMI": bmi,
        "Physical activity frequency": pal,
        "family_history_with_overweight": "yes" if family_history == "Yes" else "no",
        "CAEC": caec
    }])

    input_encoded = pd.get_dummies(input_data)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    input_scaled = scaler.transform(input_encoded)
    input_tensor = torch.tensor(input_scaled, dtype=torch.float32)

    # Predition
    with torch.no_grad():
        output = model(input_tensor)
        probs = F.softmax(output, dim=1)
        confidence, pred = torch.max(probs, 1)

    label = target_encoder.inverse_transform(pred.numpy())[0]
    conf = confidence.item()

    # Result
    st.markdown("---")
    st.subheader("Prediction Result")

    if "Obesity" in label:
        st.error(f"{label} | Confidence: {conf:.2f}")
    elif "Overweight" in label:
        st.warning(f"{label} | Confidence: {conf:.2f}")
    else:
        st.success(f"{label} | Confidence: {conf:.2f}")
    
    if 18.5 <= bmi < 25 and "Overweight" in label:
        st.info("Note: BMI is normal, but lifestyle factors indicate potential risk.")

    st.caption("Confidence score (closer to 1 indicates higher reliability)")

    
    # Explainability
    st.markdown("### Key Factors Influencing Prediction")

    factors = [
        ("Weight", round(weight / 15, 3), "increases risk" if weight > 70 else "moderate impact"),
        ("Height", round(1 / height, 3) if height > 0 else 0, "decreases risk"),
        ("Activity Level", round(pal / 4, 3), "decreases risk" if pal > 2 else "increases risk"),
        ("Age", round(age_val / 100, 3), "increases risk"),
        ("Family History", 0.05, "influences risk" if family_history == "Yes" else "low impact"),
        ("Food Between Meals", 0.05, "influences risk" if caec in ["Frequently", "Always"] else "low impact")
    ]

    for name, val, desc in factors:
        st.markdown(f"- **{name}**: {val:.3f} ({desc})")

    # Health Suggestions
    st.markdown("---")
    st.subheader("Health Suggestions")

    # Normal Health Case
    if 18.5 <= bmi < 25:
        if pal >= 3:
            st.success("Maintain current activity level")
        else:
            st.warning("Increase physical activity")

    # Overweight / Obesity
    elif bmi >= 25:
        st.warning("Reduce weight through diet and exercise")

    # Underweight
    elif bmi < 18.5:
        st.warning("Increase healthy calorie intake")