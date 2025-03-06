import streamlit as st
import requests
import base64
import time
import os
import sys
import pandas as pd
import plotly.express as px
import numpy as np

# Add server directory to path
THIS_FILE_DIR = os.path.dirname(__file__)
PARENT_DIR = os.path.abspath(os.path.join(THIS_FILE_DIR, ".."))
SERVER_DIR = os.path.join(PARENT_DIR, "server")
sys.path.append(SERVER_DIR)

# Local Imports
from client_handler import FHEMedicalClient
from nlp import extract_valid_symptoms
from helper import parse_metrics_file

# Configuration
SERVER_URL = "http://127.0.0.1:5000"

# ------------------------------------------------------------------------------
# Streamlit layout
# ------------------------------------------------------------------------------
st.set_page_config(
    page_title="FHE Medical Diagnosis",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Dark theme CSS overrides
st.markdown(
    """
    <style>
    body {color: #fff; background-color: #0e1117;}
    .stApp {background-color: #0e1117;}
    .stTextArea textarea {background-color: #262730; color: #fff;}
    .stCheckbox label {color: #fff;}
    
    /* Make cards more compact */
    .card {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #262730;
        margin-bottom: 1rem;
    }
    
    /* Prediction display */
    .prediction {
        display: flex;
        justify-content: space-between;
        padding: 8px;
        margin-bottom: 4px;
        border-radius: 4px;
        background-color: rgba(255, 255, 255, 0.05);
    }
    .prediction-disease {
        font-weight: bold;
    }
    .prediction-prob {
        color: #00cfff;
    }
    
    /* Timing metrics */
    .time-metrics {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 10px;
    }
    .time-metric {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 8px;
        border-radius: 4px;
        text-align: center;
    }
    .time-label {
        font-size: 0.8em;
        color: #aaa;
    }
    .time-value {
        font-size: 1.2em;
        font-weight: bold;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Make sure session is initialized
if "session_id" not in st.session_state:
    with st.spinner("Initializing session..."):
        resp = requests.post(f"{SERVER_URL}/init_session")
        if resp.status_code == 200:
            st.session_state.session_id = resp.json()["session_id"]
        else:
            st.error("Failed to initialize session")
            st.stop()

session_id = st.session_state.session_id

# ------------------------------------------------------------------------------
# Create two tabs: 1) Workflow, 2) Performance Dashboard
# ------------------------------------------------------------------------------
tab1, tab2 = st.tabs(["Workflow", "Performance Dashboard"])

# ------------------------------------------------------------------------------
# TAB 1: Encrypted Diagnosis Workflow
# ------------------------------------------------------------------------------
with tab1:
    st.header("Encrypted Diagnosis Workflow")

    with st.sidebar:
        st.header("Input")
        clinical_text = st.text_area(
            "Enter clinical text",
            height=200,
            placeholder="e.g., I have a fever and depression..."
        )
        use_lr = st.checkbox("Logistic Regression (LR)", value=True)
        use_dp = st.checkbox("Differentially Private LR (DP)", value=True)
        process_button = st.button("Process")

    if process_button:
        if not clinical_text:
            st.error("Please enter some clinical text")
        elif not (use_lr or use_dp):
            st.error("Please select at least one model")
        else:
            # 1) Extract symptoms
            with st.spinner("Extracting symptoms..."):
                symptoms = extract_valid_symptoms(clinical_text)
            
            # Display results in a cleaner card
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Extracted Symptoms")
            if symptoms:
                for symptom in symptoms:
                    st.markdown(f"• {symptom}")
            else:
                st.warning("No symptoms extracted. Please provide more detailed text.")
            st.markdown('</div>', unsafe_allow_html=True)

            # 2) Which models?
            selected_models = []
            if use_lr:
                selected_models.append("LR")
            if use_dp:
                selected_models.append("DP")

            # Prepare for encryption
            enc_results = {}
            enc_times = {}
            clients = {}
            encrypted_data_dict = {}
            public_keys_dict = {}
            encryption_times = {}

            # 3) Encrypt data
            for model_type in selected_models:
                with st.spinner(f"Preparing encrypted data for {model_type}..."):
                    client = FHEMedicalClient(session_id, model_type)
                    clients[model_type] = client
                    pub_key = client.load_eval_keys()
                    ciphertext, enc_time = client.process_and_encrypt(symptoms)
                    encrypted_data_dict[model_type] = base64.b64encode(ciphertext).decode("utf-8")
                    public_keys_dict[model_type] = base64.b64encode(pub_key).decode("utf-8")
                    encryption_times[model_type] = enc_time

            # 4) Encrypted Inference
            if selected_models:
                enc_payload = {
                    "session_id": session_id,
                    "model_type": ", ".join(selected_models),
                }
                for model_type in selected_models:
                    enc_payload[f"encrypted_data_{model_type.lower()}"] = encrypted_data_dict[model_type]
                    enc_payload[f"public_key_{model_type.lower()}"] = public_keys_dict[model_type]

                with st.spinner("Processing encrypted inference..."):
                    resp_enc = requests.post(f"{SERVER_URL}/predict_encrypted", json=enc_payload)
                    if resp_enc.status_code == 200:
                        enc_data = resp_enc.json()["results"]
                        for model_type in selected_models:
                            if model_type in enc_data:
                                enc_result = base64.b64decode(enc_data[model_type]["encrypted_result"])
                                inference_time = enc_data[model_type]["inference_time"]
                                client = clients[model_type]
                                start_dec = time.time()
                                preds = client.decrypt_result(enc_result)
                                dec_time = time.time() - start_dec
                                enc_results[model_type] = preds
                                enc_times[model_type] = {
                                    "encryption": encryption_times[model_type],
                                    "inference": inference_time,
                                    "decryption": dec_time,
                                    "total": encryption_times[model_type] + inference_time + dec_time
                                }
                            else:
                                st.warning(f"No encrypted results returned for {model_type}")
                    else:
                        st.error(f"Encrypted inference failed: {resp_enc.text}")

            # 5) Plaintext Inference
            pt_results = {}
            pt_times = {}
            if selected_models:
                pt_payload = {
                    "clinical_text": clinical_text,
                    "model_type": ", ".join(selected_models)
                }
                with st.spinner("Processing plaintext inference..."):
                    resp_pt = requests.post(f"{SERVER_URL}/predict_plaintext", json=pt_payload)
                    if resp_pt.status_code == 200:
                        pt_data = resp_pt.json()["results"]
                        for model_type in selected_models:
                            if model_type in pt_data:
                                pt_results[model_type] = pt_data[model_type]["predictions"]
                                pt_times[model_type] = pt_data[model_type]["inference_time"]
                            else:
                                st.warning(f"No plaintext results returned for {model_type}")
                    else:
                        st.error(f"Plaintext inference failed: {resp_pt.text}")

            # 6) Display Results
            for model_type in selected_models:
                st.subheader(f"{model_type} Model Results", divider="rainbow")
                col1, col2 = st.columns(2)

                # Encrypted
                with col1:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### 🔒 Encrypted Inference")
                    ct_b64 = encrypted_data_dict.get(model_type, "")
                    if ct_b64:
                        ct_bytes = base64.b64decode(ct_b64)
                        ct_hex = ct_bytes.hex()
                        truncated_ct = ct_hex[:50] + "..." if len(ct_hex) > 50 else ct_hex
                        st.code(truncated_ct, language="text")
                    else:
                        st.write("No ciphertext available.")

                    if model_type in enc_results:
                        # Display predictions as simple text
                        st.markdown("**Top Predictions:**")
                        preds = enc_results[model_type][:3]
                        for disease, prob in preds:
                            st.markdown(
                                f"""<div class="prediction">
                                   <span class="prediction-disease">{disease}</span>
                                   <span class="prediction-prob">{prob*100:.1f}%</span>
                                   </div>""", 
                                unsafe_allow_html=True
                            )
                        
                        # Display timing information
                        st.markdown("<br>**Timing Information:**", unsafe_allow_html=True)
                        times = enc_times[model_type]
                        st.markdown(
                            f"""<div class="time-metrics">
                                <div class="time-metric">
                                    <div class="time-label">Encryption</div>
                                    <div class="time-value">{times['encryption']:.3f}s</div>
                                </div>
                                <div class="time-metric">
                                    <div class="time-label">Inference</div>
                                    <div class="time-value">{times['inference']:.3f}s</div>
                                </div>
                                <div class="time-metric">
                                    <div class="time-label">Decryption</div>
                                    <div class="time-value">{times['decryption']:.3f}s</div>
                                </div>
                                <div class="time-metric">
                                    <div class="time-label">Total</div>
                                    <div class="time-value">{times['total']:.3f}s</div>
                                </div>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    else:
                        st.write("No encrypted results available")
                    st.markdown('</div>', unsafe_allow_html=True)

                # Plaintext
                with col2:
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    st.markdown("#### 📄 Plaintext Inference")
                    truncated_text = clinical_text[:50] + "..." if len(clinical_text) > 50 else clinical_text
                    st.code(truncated_text, language="text")
                    
                    if model_type in pt_results:
                        # Display predictions as simple text
                        st.markdown("**Top Predictions:**")
                        preds = pt_results[model_type][:3]
                        for disease, prob in preds:
                            st.markdown(
                                f"""<div class="prediction">
                                   <span class="prediction-disease">{disease}</span>
                                   <span class="prediction-prob">{prob*100:.1f}%</span>
                                   </div>""", 
                                unsafe_allow_html=True
                            )
                        
                        # Display timing information
                        st.markdown("<br>**Timing Information:**", unsafe_allow_html=True)
                        st.markdown(
                            f"""<div class="time-metrics">
                                <div class="time-metric">
                                    <div class="time-label">Inference</div>
                                    <div class="time-value">{pt_times[model_type]:.3f}s</div>
                                </div>
                            </div>""", 
                            unsafe_allow_html=True
                        )
                    else:
                        st.write("No plaintext results available")
                    st.markdown('</div>', unsafe_allow_html=True)

            st.info("Note: Encrypted inference times are higher due to FHE's overhead, ensuring data privacy.")

# ------------------------------------------------------------------------------
# TAB 2: Performance Dashboard
# ------------------------------------------------------------------------------
with tab2:
    st.header("Performance Dashboard")

    # 1) Paths to your new .txt results
    model_paths = {
        "LR Baseline": "/home/isaacng33/individual_project/streamlit_app/artifacts/results/lr_baseline.txt",
        "LR DP (Plain)": "/home/isaacng33/individual_project/streamlit_app/artifacts/results/lr_dp_plain.txt",
        "LR DP (FHE)": "/home/isaacng33/individual_project/streamlit_app/artifacts/results/lr_dp_fhe.txt",
        "LR Encrypted": "/home/isaacng33/individual_project/streamlit_app/artifacts/results/lr_encrypted.txt",
    }

    # 2) Parse metrics for each model
    results_data = {}
    for model_name, path in model_paths.items():
        m = parse_metrics_file(path) or {}
        results_data[model_name] = m

    # 3) Construct a DataFrame
    rows = []
    for model_name, met in results_data.items():
        training_time = met.get("training_time", 0) or 0
        compile_time = met.get("compile_time", 0) or 0
        total_time = training_time + compile_time

        row = {
            "Model": model_name,
            "Training Time (s)": training_time,
            "Compile Time (s)": compile_time,
            "Total Time (s)": total_time,
            "Prediction Time (s)": met.get("prediction_time", 0),
            "Accuracy": met.get("accuracy", 0),
            "Precision": met.get("precision", 0),
            "Recall": met.get("recall", 0),
            "F1 Score": met.get("f1_score", 0),
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    # Make sure each model appears only once
    df = df.drop_duplicates(subset=["Model"])

    if df.empty:
        st.write("No metrics data found.")
    else:
        # Display metrics in a simple table
        st.subheader("Summary Metrics")
        # Round all numeric columns to 3 decimal places for cleaner display
        rounded_df = df.round(3)
        st.table(rounded_df)
        
        # Second row of visualizations
        cols = st.columns(2)
        
        # 1. Total Time Chart (Bar)
        with cols[0]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Total Build Time")
            
            fig = px.bar(
                df,
                x="Model",
                y="Total Time (s)",
                title="Training + Compile Time by Model",
                color="Total Time (s)",
                color_continuous_scale="Blues",
                height=300,
                text=df["Total Time (s)"].apply(lambda x: f"{x:.2f}s")
            )
            
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Time (seconds)",
                hovermode="x unified",
                hoverlabel=dict(bgcolor="white", font_size=14),
                margin=dict(l=40, r=20, t=40, b=20)
            )
            
            # Simplified hover labels
            fig.update_traces(
                hovertemplate="Model: %{x}<br>Time: %{y:.3f}s"
            )
            
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 2. Prediction Time (Log Bar)
        with cols[1]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Prediction Time (Log Scale)")
            
            fig = px.bar(
                df,
                x="Model",
                y="Prediction Time (s)",
                title="Inference Performance",
                color="Prediction Time (s)",
                color_continuous_scale="Reds",
                height=300,
                text=df["Prediction Time (s)"].apply(lambda x: f"{x:.3f}s")
            )
            
            fig.update_layout(
                xaxis_title="",
                yaxis_title="Time (seconds)",
                yaxis_type="log",  # Set log scale
                hovermode="x unified",
                hoverlabel=dict(bgcolor="white", font_size=14),
                margin=dict(l=40, r=20, t=40, b=20)
            )
            
            # Simplified hover labels
            fig.update_traces(
                hovertemplate="Model: %{x}<br>Time: %{y:.4f}s"
            )
            
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)
        
        # 3. Performance Metrics Detail
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Performance Metrics Breakdown")
        
        # Create a grouped bar chart with plotly
        perf_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
        df_perf = df[["Model"] + perf_cols].copy()
        
        # Melt the DataFrame to get it in the right format for Plotly
        df_perf_melt = pd.melt(
            df_perf,
            id_vars=["Model"],
            value_vars=perf_cols,
            var_name="Metric",
            value_name="Value"
        )
        
        fig = px.bar(
            df_perf_melt,
            x="Model",
            y="Value",
            color="Metric",
            barmode="group",
            title="Model Performance Metrics",
            height=350,
            text=df_perf_melt["Value"].apply(lambda x: f"{x:.2f}")
        )
        
        fig.update_layout(
            xaxis_title="",
            yaxis_title="Metric Value",
            yaxis_range=[0, 1.05],
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(l=40, r=20, t=60, b=20),
            hovermode="x unified"
        )
        
        # Simplified hover labels
        fig.update_traces(
            hovertemplate="Model: %{x}<br>Metric: %{marker.color}<br>Value: %{y:.3f}"
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Explanation
        st.info(
            "This dashboard shows performance metrics across models. Hover over chart elements to see "
            "detailed values. The bar charts show time and classification performance metrics. "
            "Note the log scale on prediction time for better visibility."
        )
