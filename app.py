import streamlit as st
import pandas as pd
import joblib
import os
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

st.set_page_config(page_title="Bank Marketing Classifier", layout="wide")
st.title("Bank Marketing – Term Deposit Prediction")
st.write("Upload test data and select a model to see evaluation metrics.")

# Available models
MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/support_vector_machine.joblib"
}

# Sidebar
st.sidebar.header("Options")
uploaded_file = st.sidebar.file_uploader("Upload test CSV", type=["csv"])
selected_model = st.sidebar.selectbox("Select Model", list(MODEL_FILES.keys()))

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    # Expect target column named 'y'
    if 'y' not in df.columns:
        st.error("The uploaded CSV must contain a column named 'y' (target).")
    else:
        X = df.drop(columns=['y'])
        y = df['y']
        
        # Load model
        model_path = MODEL_FILES[selected_model]
        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
        else:
            model = joblib.load(model_path)
            
            # Predict
            y_pred = model.predict(X)
            y_proba = model.predict_proba(X)[:, 1]
            
            # Metrics
            acc = accuracy_score(y, y_pred)
            auc = roc_auc_score(y, y_proba)
            prec = precision_score(y, y_pred)
            rec = recall_score(y, y_pred)
            f1 = f1_score(y, y_pred)
            mcc = matthews_corrcoef(y, y_pred)
            
            # Display metrics
            st.subheader(f"Metrics for {selected_model}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{acc:.4f}")
            col1.metric("AUC", f"{auc:.4f}")
            col2.metric("Precision", f"{prec:.4f}")
            col2.metric("Recall", f"{rec:.4f}")
            col3.metric("F1 Score", f"{f1:.4f}")
            col3.metric("MCC", f"{mcc:.4f}")
            
            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y, y_pred)
            fig, ax = plt.subplots(figsize=(5, 4))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No", "Yes"])
            disp.plot(ax=ax, cmap="Blues")
            st.pyplot(fig)
            
            # Optional: show a few predictions
            with st.expander("Show sample predictions"):
                sample = X.head(10).copy()
                sample["Actual"] = y.head(10).values
                sample["Predicted"] = y_pred[:10]
                st.dataframe(sample)
else:
    st.info("Please upload a test CSV file using the sidebar to begin.")