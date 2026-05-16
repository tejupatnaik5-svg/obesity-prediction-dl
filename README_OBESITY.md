# Obesity Level Prediction Using Deep Learning

A deep learning application for predicting obesity risk levels based on lifestyle, dietary, and physical attributes.

Built as part of a Deep Learning course project at University of Colorado Denver.

---

## Overview

This project develops a **multi-class classification system** that categorizes individuals into 7 obesity levels using a feedforward neural network with dropout regularization. The system provides:

- **High accuracy predictions** (89% test accuracy)
- **Confidence scores** reflecting prediction reliability
- **SHAP-based feature explanations** for interpretability
- **Interactive Streamlit web application** for easy user interaction
- **Wearable device integration** for activity level detection

The model learns non-linear relationships between behavioral factors (diet, activity, family history) and obesity risk, providing actionable health insights.

---

## Features

**Deep Learning Model**
- Feedforward Neural Network (3 hidden layers)
- Dropout regularization (prevents overfitting)
- 89% test accuracy on 7-class obesity prediction

**Model Explainability**
- SHAP feature attribution for interpretable predictions
- Top-5 contributing factors displayed per prediction
- Confidence scores indicating prediction reliability

**Interactive Web Application**
- User-friendly Streamlit interface
- Real-time BMI calculation with WHO feedback
- Activity level auto-detection from wearable data
- Personalized health recommendations
- Visual confidence indicators

**Reproducible Pipeline**
- Complete data preprocessing (one-hot encoding, scaling)
- Model training & evaluation scripts
- Saved artifacts (model weights, scaler, encoders)
- Step-by-step documentation

---

## Tech Stack

Python 3.9+
PyTorch
Streamlit
SHAP
Pandas, NumPy, Scikit-learn
Matplotlib
Joblib
Git

---

## Local Setup and Installation

### Prerequisites

- Python 3.10 or higher
- Git
- Virtual environment manager (included with Python)

### Step 1: Clone the Repository

```bash
git clone git clone https://github.com/tejupatnaik5-svg/obesity-prediction-dl.git
cd obesity-prediction-dl
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
```

### Step 3: Activate Virtual Environment

**Windows:**
```bash
.\venv\Scripts\activate
```

**macOS/Linux:**
```bash
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Prepare Dataset

Place `Obesity_dataset.csv` in:
```
obesity-prediction-dl/dataset/Obesity_dataset.csv
```

Create directory if needed:
```bash
mkdir -p dataset
```

### Step 6: Train Model (Optional)

```bash
python src/model.py
```

This generates:
- `model.pth` (trained weights)
- `scaler.pkl` (feature scaler)
- `columns.pkl` (column names)
- `label_encoder.pkl` (class mappings)
- `figures/loss.png` (loss curve)

### Step 7: Launch Application

```bash
streamlit run app.py
```

Access at: `http://localhost:8501`

---

## Project Structure

```
obesity-prediction-dl/
├── dataset/
│ ├── Obesity_dataset.csv # Main dataset
│ └── dailyActivity_merged.csv # Activity data (optional)
├── figures/ # Graphs and visualizations
├── app.py # Streamlit application
├── explore.py # Model training script
├── model.pth # Trained model weights
├── scaler.pkl # Feature scaler
├── columns.pkl # Encoded feature columns
├── label_encoder.pkl # Target encoder
├── requirements.txt # Dependencies
├── README.md # Documentation
└── logo.png # App logo
```

---

## Dataset

**2,111 samples** with 7 obesity classes:
1. Insufficient Weight
2. Normal Weight
3. Overweight Level I
4. Overweight Level II
5. Obesity Type I
6. Obesity Type II
7. Obesity Type III

**17 Features:** Age, Gender, Height, Weight, Family history, Diet habits, Activity level, etc.

---

## Model Architecture

```
Input (features)
    ↓
Hidden 1: 128 neurons + ReLU + Dropout(0.3)
    ↓
Hidden 2: 64 neurons + ReLU + Dropout(0.3)
    ↓
Hidden 3: 32 neurons + ReLU
    ↓
Output: 7 obesity classes
```

**Performance:** 89% test accuracy

---

## Usage

1. Enter user information (age, weight, height, activity level)
2. Optionally auto-detect activity from wearable data
3. Click "Predict"
4. View obesity level, confidence score, and SHAP explanations
5. Receive personalized health recommendations

---

## Notes

**This is a proof-of-concept**, not for clinical diagnosis.

- Confidence scores are relative, not absolute accuracy measures
- Training data from Colombia, Peru, Mexico
- SHAP explanations add ~2–5 seconds latency
- Activity auto-detection simulates Fitbit data

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| ModuleNotFoundError | Run: `pip install -r requirements.txt` |
| Dataset not found | Place `Obesity_dataset.csv` in `dataset/` folder |
| Model not found | Train with: `python src/model.py` |
| Port in use | Try: `streamlit run src/app.py --server.port 8502` |

---

## Future Work

- [ ] Add per-class performance metrics
- [ ] Confidence calibration (temperature scaling)
- [ ] More diverse training data
- [ ] Live wearable API integration
- [ ] Mobile app deployment
- [ ] Longitudinal tracking

---

## References

[1] Palechor & de la Hoz (2019) - Obesity Dataset
[2] LeCun et al. (2015) - Deep Learning
[3] Srivastava et al. (2014) - Dropout Regularization
[4] Kingma & Ba (2015) - Adam Optimizer
[5] Lundberg & Lee (2017) - SHAP Explainability
[6] Guo et al. (2017) - Confidence Calibration

---

## License

Academic & educational purposes - University of Colorado Denver

**Author:** Tejaswi Baggam  
**Institution:** University of Colorado Denver  
**Last Updated:** May 2026

