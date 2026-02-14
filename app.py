"""
NexGen ML Classifier - Modern Streamlit UI
Fixed: Arrow Serialization & Width Warnings
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import time
import warnings
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, classification_report, roc_curve, auc,
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

warnings.filterwarnings('ignore')

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NexGen ML Classifier",
    page_icon="🧬",
    layout="wide",
)

# --- ENHANCED CUSTOM CSS ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #f8fafc, #e2e8f0);
    }

    /* Glassmorphism card effect */
    .metric-card {
        background: rgba(255, 255, 255, 0.8);
        border-radius: 16px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.3);
        text-align: center;
        transition: all 0.3s ease;
        margin-bottom: 20px;
    }

    .metric-card:hover {
        transform: translateY(-4px);
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.1);
    }

    .metric-label {
        font-size: 0.85rem;
        color: #64748b;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }

    .metric-value {
        font-size: 2rem;
        color: #0f172a;
        font-weight: 800;
    }

    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: rgba(255, 255, 255, 0.5);
        backdrop-filter: blur(10px);
    }

    /* Custom Button */
    .stButton>button {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: white;
        border: none;
        border-radius: 12px;
        font-weight: 600;
        height: 3em;
        transition: all 0.2s ease;
    }

    .stButton>button:hover {
        transform: scale(1.02);
        border: none;
        color: white;
    }
    
    .main-title {
        background: linear-gradient(to right, #1e293b, #4f46e5);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 3rem;
        margin-bottom: 0;
    }
</style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
def custom_metric(label, value, icon="⚡"):
    """Styled metric card"""
    st.markdown(f"""
    <div class="metric-card">
        <div style="font-size: 28px; margin-bottom: 10px;">{icon}</div>
        <div class="metric-label">{label}</div>
        <div class="metric-value">{value}</div>
    </div>
    """, unsafe_allow_html=True)

def load_models():
    models = {}
    model_dir = 'models'
    model_files = {
        'Logistic Regression': 'logistic_regression.pkl',
        'Decision Tree': 'decision_tree.pkl',
        'K-Nearest Neighbor': 'knn.pkl',
        'Naive Bayes': 'naive_bayes.pkl',
        'Random Forest': 'random_forest.pkl',
        'XGBoost': 'xgboost.pkl'
    }
    for model_name, file_name in model_files.items():
        model_path = os.path.join(model_dir, file_name)
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models[model_name] = pickle.load(f)
    return models

def load_scaler():
    scaler_path = 'models/scaler.pkl'
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            return pickle.load(f)
    return None

def load_evaluation_results():
    results_path = 'models/results.pkl'
    if os.path.exists(results_path):
        with open(results_path, 'rb') as f:
            return pickle.load(f)
    return None

# --- MAIN APP ---
def main():
    st.markdown('<h1 class="main-title">ML Classification Models</h1>', unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar
    st.sidebar.markdown("## 📋 Navigation")
    page = st.sidebar.radio("Select a page:", ["Model Comparison", "Model Prediction", "Dataset Analysis"])

    if page == "Model Comparison":
        st.header("Model Comparison & Evaluation")
        results = load_evaluation_results()

        if results:
            # FIX: Filter out the 'Confusion Matrix' array so Arrow can serialize the dataframe
            comparison_data = []
            for model_name, metrics in results.items():
                comparison_data.append({
                    'Model': model_name,
                    'Accuracy': metrics.get('Accuracy', 0),
                    'AUC': metrics.get('AUC', 0),
                    'Precision': metrics.get('Precision', 0),
                    'Recall': metrics.get('Recall', 0),
                    'F1 Score': metrics.get('F1 Score', 0),
                    'MCC': metrics.get('MCC', 0)
                })

            df_results = pd.DataFrame(comparison_data)

            st.subheader("📊 Evaluation Metrics Comparison")
            # Updated width to 'stretch' as per 2026 requirements
            st.dataframe(df_results.style.background_gradient(cmap='Blues', subset=['Accuracy', 'AUC']), width='stretch')

            # Download
            csv = df_results.to_csv(index=False)
            st.download_button("📥 Download Results as CSV", csv, "model_comparison.csv", "text/csv")

            # Charts
            st.subheader("📈 Metrics Visualization")
            col1, col2 = st.columns(2)
            with col1:
                fig_acc = px.bar(df_results, x='Model', y='Accuracy', color='Accuracy',
                                 title='Accuracy Comparison', color_continuous_scale='Viridis')
                st.plotly_chart(fig_acc, width='stretch')
            with col2:
                fig_auc = px.bar(df_results, x='Model', y='AUC', color='AUC',
                                 title='AUC Score Comparison', color_continuous_scale='Magma')
                st.plotly_chart(fig_auc, width='stretch')
        else:
            st.warning("⚠️ No evaluation results found. Please train models first.")

    elif page == "Model Prediction":
        st.header("Make Predictions")
        models = load_models()
        scaler = load_scaler()

        if models and scaler:
            uploaded_file = st.file_uploader("Upload Test CSV", type="csv")
            if uploaded_file:
                df_test = pd.read_csv(uploaded_file)
                st.success(f"Uploaded: {df_test.shape}")
                st.dataframe(df_test.head(), width='stretch')

                selected_model = st.selectbox("Select Model:", list(models.keys()))

                if st.button("🚀 Run Prediction"):
                    df_scaled = scaler.transform(df_test)
                    model = models[selected_model]

                    preds = model.predict(df_scaled)
                    probs = model.predict_proba(df_scaled)

                    st.subheader(f"Results: {selected_model}")
                    results_df = pd.DataFrame({
                        'Prediction': preds,
                        'Confidence': np.max(probs, axis=1)
                    })
                    st.dataframe(results_df, width='stretch')
        else:
            st.warning("⚠️ Models/Scaler not found in 'models/' directory.")

    elif page == "Dataset Analysis":
        st.header("Dataset Overview")
        data_path = 'data/breast_cancer.csv'
        if os.path.exists(data_path):
            df = pd.read_csv(data_path)

            # Metric Row
            c1, c2, c3, c4 = st.columns(4)
            with c1: custom_metric("Samples", df.shape[0], "📂")
            with c2: custom_metric("Features", df.shape[1]-1, "📊")
            with c3: custom_metric("Positives", (df['target']==1).sum(), "✅")
            with c4: custom_metric("Negatives", (df['target']==0).sum(), "❌")

            st.dataframe(df.describe().T, width='stretch')
        else:
            st.warning("⚠️ Dataset not found at 'data/breast_cancer.csv'.")

    # Footer
    st.markdown("---")
    st.markdown('<div style="text-align: center; color: gray;">ML Classification Models | Built with Streamlit 🎈</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()