import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report
)

# Page Configuration
st.set_page_config(
    page_title="NASA NEO Asteroid Hazard Predictor",
    page_icon="☄️",
    layout="wide"
)

# Title & Description
st.title("☄️ NASA Near-Earth Objects (NEO) Hazard Predictor")
st.markdown("##### Machine Learning Assignment 2 - Classification & Streamlit Cloud Deployment")

# Sidebar Setup
st.sidebar.header("⚙️ App Configuration")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Support Vector Machine": "support_vector_machine.pkl"
}

# 1. Dataset Upload Option (CSV)
uploaded_file = st.sidebar.file_uploader("Upload Test Dataset (test_data.csv)", type=["csv"])

# 2. Model Selection Dropdown
selected_model_name = st.sidebar.selectbox("Select Classifier Model", list(MODEL_FILES.keys()))

st.sidebar.markdown("---")
st.sidebar.caption("Ensure `test_data.csv` contains feature variables and target column `is_potentially_hazardous_asteroid`.")

if uploaded_file is not None:
    try:
        data = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded test data: **{data.shape} rows, {data.shape[1]} columns**")
        
        # Detect target column
        target_col = 'is_potentially_hazardous_asteroid'
        if target_col not in data.columns:
            target_col = data.columns[-1]
            
        X_test = data.drop(columns=[target_col])
        y_test = data[target_col].astype(int)
        
        model_path = os.path.join("model", MODEL_FILES[selected_model_name])
        scaler_path = os.path.join("model", "scaler.pkl")
        
        if not os.path.exists(model_path):
            st.error(f"Model file `{model_path}` not found in repository. Please ensure `.pkl` files exist in `model/` directory.")
        else:
            model = joblib.load(model_path)
            
            # Apply feature scaling for linear/distance-based models
            if selected_model_name in ["Logistic Regression", "KNN", "Naive Bayes", "Support Vector Machine"]:
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    X_eval = scaler.transform(X_test)
                else:
                    X_eval = X_test
            else:
                X_eval = X_test
            
            # Predict (without retraining)
            y_pred = model.predict(X_eval)
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_eval)[:, 1]
            else:
                y_proba = y_pred
            
            # Metric Calculations
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba)
            prec = precision_score(y_test, y_pred, zero_division=0)
            rec = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            mcc = matthews_corrcoef(y_test, y_pred)
            
            # Display Evaluation Metrics Display
            st.subheader(f"📊 Evaluation Metrics ({selected_model_name})")
            col1, col2, col3, col4, col5, col6 = st.columns(6)
            
            col1.metric("Accuracy", f"{acc:.4f}")
            col2.metric("AUC Score", f"{auc:.4f}")
            col3.metric("Precision", f"{prec:.4f}")
            col4.metric("Recall", f"{rec:.4f}")
            col5.metric("F1 Score", f"{f1:.4f}")
            col6.metric("MCC Score", f"{mcc:.4f}")
            
            st.markdown("---")
            
            # Visualizations
            c1, c2 = st.columns(2)
            
            with c1:
                st.subheader("📌 Confusion Matrix")
                cm = confusion_matrix(y_test, y_pred)
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', cbar=False,
                            xticklabels=['Non-Hazardous (0)', 'Hazardous (1)'],
                            yticklabels=['Non-Hazardous (0)', 'Hazardous (1)'], ax=ax)
                plt.xlabel("Predicted Label")
                plt.ylabel("True Label")
                plt.title(f"Confusion Matrix - {selected_model_name}")
                st.pyplot(fig)
                
            with c2:
                st.subheader("📋 Classification Report")
                report_dict = classification_report(y_test, y_pred, output_dict=True)
                report_df = pd.DataFrame(report_dict).transpose()
                st.dataframe(report_df.style.format("{:.4f}"))
                
            with st.expander("🔍 View Raw Test Dataset Sample"):
                st.dataframe(data.head(10))

    except Exception as e:
        st.error(f"Error evaluating model: {str(e)}")
else:
    st.info("👈 Please upload `test_data.csv` using the sidebar menu to view model evaluation metrics.")