import streamlit as st
import requests
import base64
import time
import os
import sys
import pandas as pd
import plotly.express as px
import json
import re

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

# Dark theme CSS overrides (unchanged)
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

# Session initialization (unchanged)
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
# Create three tabs
# ------------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Workflow", "Performance Dashboard", "Privacy Insights"])

# ------------------------------------------------------------------------------
# TAB 1: Encrypted Diagnosis Workflow (unchanged)
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

    # Educational Expander (unchanged)
    with st.expander("How does this work?"):
        st.markdown("""
        This application uses **Fully Homomorphic Encryption (FHE)** to ensure data privacy during medical diagnosis. Here's the complete data flow and client-side architecture:

        - **Symptom Extraction:** The process begins on the client side, where Natural Language Processing (NLP) extracts symptoms from the clinical text you enter (e.g., "fever" or "depression").
        - **Client-Side Encryption:** The extracted symptoms are encrypted using FHE directly on your device. Encryption keys are generated and stored locally on the client side, meaning they never leave your machine. This ensures that only encrypted data is transmitted, keeping your sensitive information secure.
        - **Encrypted Data Transmission:** The encrypted symptoms (ciphertext) are sent to the server over a secure connection. Since the server only receives encrypted data and does not have access to the keys, it cannot decrypt or view the plaintext information.
        - **Server-Side Encrypted Inference:** On the server, a machine learning model (e.g., Logistic Regression or Differentially Private LR) performs predictions directly on the encrypted data using FHE. This computation happens without ever decrypting the data, preserving privacy throughout the process.
        - **Encrypted Results Return:** The server sends the encrypted predictions back to the client. These results remain encrypted during transmission.
        - **Client-Side Decryption:** Finally, the predictions are decrypted on your device using the locally stored encryption keys. The decrypted results match what would be obtained from plaintext inference, ensuring accuracy while maintaining privacy.

        **Key Points:**
        - **Client-Side Key Management:** Encryption keys are generated and stored exclusively on the client side, ensuring the server never sees them.
        - **Privacy Guarantee:** Only encrypted text is sent to the server, making it impossible for anyone intercepting the data or the server itself to access your raw medical information.
        - **End-to-End Security:** From symptom extraction to result decryption, your data remains private and secure.
        """)

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
# TAB 2: Performance Dashboard (unchanged)
# ------------------------------------------------------------------------------
with tab2:
    st.header("Performance Dashboard")

    # 1) Paths to your new .txt results (Update these paths based on your local setup)
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
        rounded_df = df.round(3)
        st.table(rounded_df)

        # Second row of visualizations
        cols = st.columns(2)

        # 1. Stacked Bar Chart for Total Build Time
        with cols[0]:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.subheader("Total Build Time")
            fig = px.bar(df, x="Model", y=["Training Time (s)", "Compile Time (s)"],
                         title="Training + Compile Time by Model", height=300,
                         text=df["Total Time (s)"].apply(lambda x: f"{x:.2f}s"))
            fig.update_layout(barmode="stack", yaxis_title="Time (seconds)", xaxis_title="")
            st.plotly_chart(fig, use_container_width=True)
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
                yaxis_type="log",
                hovermode="x unified",
                hoverlabel=dict(bgcolor="white", font_size=14),
                margin=dict(l=40, r=20, t=40, b=20)
            )
            fig.update_traces(hovertemplate="Model: %{x}<br>Time: %{y:.4f}s")
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
            st.markdown('</div>', unsafe_allow_html=True)

        # 3. Performance Metrics Detail
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Performance Metrics Breakdown")
        perf_cols = ["Accuracy", "Precision", "Recall", "F1 Score"]
        df_perf = df[["Model"] + perf_cols].copy()
        df_perf_melt = pd.melt(df_perf, id_vars=["Model"], value_vars=perf_cols, var_name="Metric", value_name="Value")
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
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(l=40, r=20, t=60, b=20),
            hovermode="x unified"
        )
        fig.update_traces(hovertemplate="Model: %{x}<br>Metric: %{marker.color}<br>Value: %{y:.3f}")
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

        # 4. Model Comparison Feature
        st.subheader("Compare Models")
        col1, col2 = st.columns(2)
        model_options = df["Model"].tolist()
        with col1:
            model_a = st.selectbox("Select Model A", model_options, index=0)
        with col2:
            model_b = st.selectbox("Select Model B", model_options, index=1)

        if model_a != model_b:
            comparison_df = df[df["Model"].isin([model_a, model_b])]
            st.table(comparison_df.set_index("Model").T)
        else:
            st.warning("Please select two different models to compare.")

        # Educational Expander for Metrics (unchanged)
        with st.expander("What do these metrics mean?"):
            st.markdown("""
            Here’s an explanation of the performance metrics displayed in the dashboard:

            - **Accuracy:** The proportion of correct predictions made by the model out of all predictions. It’s a general measure of how often the model is right.
            - **Precision:** The accuracy of positive predictions, showing how many of the predicted positive cases (e.g., a disease) are actually correct. High precision means fewer false positives.
            - **Recall:** The model’s ability to identify all actual positive cases. High recall means fewer false negatives, which is crucial in medical diagnosis to avoid missing cases.
            - **F1 Score:** The harmonic mean of precision and recall, providing a single score that balances both metrics. It’s useful when you need a trade-off between precision and recall.

            These metrics help evaluate how well the models perform, especially in medical applications where both identifying true cases (recall) and avoiding false alarms (precision) are important.
            """)

        st.info(
            "This dashboard shows performance metrics across models. Hover over chart elements to see "
            "detailed values. The bar charts show time and classification performance metrics. "
            "Note the log scale on prediction time for better visibility."
        )

# ------------------------------------------------------------------------------
# TAB 3: Privacy Insights (Updated)
# ------------------------------------------------------------------------------
with tab3:
    st.header("Privacy Insights")

    # File Paths (hardcoded for now based on your input)
    best_configs_file = "/home/isaacng33/individual_project/streamlit_app/artifacts/model_parameter/dp_best_configs_20250306_210837.json"
    results_file = "/home/isaacng33/individual_project/streamlit_app/artifacts/model_parameter/dp_grid_search_results_20250306_210837.csv"

    # Load best configurations
    with open(best_configs_file, 'r') as f:
        best_configs = json.load(f)

    # Load grid search results
    df = pd.read_csv(results_file)

    # Add rounded_epsilon column
    df['rounded_epsilon'] = df['epsilon'].apply(lambda x: int(round(x)))

    # Educational text
    st.markdown("""
    ### How This Model Was Optimized
    This dashboard showcases a differentially private logistic regression model optimized via **grid search**. We tested combinations of parameters to balance privacy (measured by ε, the privacy budget) and accuracy. Lower ε values indicate stronger privacy but may reduce accuracy due to added noise.

    #### Parameter Explanations
    - **Noise Multiplier:** Higher values increase privacy (lower ε) but add more noise, potentially reducing accuracy.
    - **Batch Size:** Larger sizes can stabilize training but increase ε, affecting privacy.
    - **Epochs:** More epochs may improve accuracy but can increase ε, weakening privacy.
    - **Max Gradient Norm:** Fixed at 1.0, caps gradient size for privacy.
    - **Learning Rate:** Scales with batch size (LR = 0.1 * (batch_size / 64)), influencing convergence speed.

    Use the controls below to explore the trade-offs between privacy and accuracy.
    """)

    # Section 1: Privacy Budget Selection
    st.write("### Select Privacy Budget")
    selected_epsilon = st.selectbox("Choose ε (Privacy Budget)", options=[0, 1, 2, 3, 4, 5], index=0)

    # Find the best configuration for the selected epsilon
    best_config = None
    for eps, config in best_configs.items():
        if int(eps) == selected_epsilon and config['params']:
            best_config = config
            break
    if best_config:
        st.markdown(f"#### Optimized Parameters for ε = {selected_epsilon}")
        st.write(f"- **Accuracy**: {best_config['accuracy']:.4f}")
        st.write(f"- **Noise Multiplier**: {best_config['params']['noise_multiplier']}")
        st.write(f"- **Batch Size**: {best_config['params']['batch_size']}")
        st.write(f"- **Epochs**: {best_config['params']['epochs']}")
        st.write(f"- **Max Grad Norm**: {best_config['params']['max_grad_norm']}")
        st.write(f"- **Learning Rate**: {best_config['params']['learning_rate']:.2f}")
    else:
        st.warning(f"No optimized configuration found for ε = {selected_epsilon}")
    # Note about learning rate
    st.write("**Note:** The learning rate is calculated as 0.1 * (batch_size / 64), scaling with batch size.")