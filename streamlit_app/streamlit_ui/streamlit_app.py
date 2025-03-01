import streamlit as st
import requests
import base64
import time
import os
import sys

this_file_dir = os.path.dirname(__file__) 
parent_dir = os.path.abspath(os.path.join(this_file_dir, ".."))
server_dir = os.path.join(parent_dir, "server")
sys.path.append(server_dir)

from client_handler import FHEMedicalClient
from nlp import extract_valid_symptoms

# Configuration
SERVER_URL = "http://127.0.0.1:5000"

# Set dark theme
st.set_page_config(page_title="FHE Medical Diagnosis", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    body {color: #fff; background-color: #0e1117;}
    .stApp {background-color: #0e1117;}
    .stTextArea textarea {background-color: #262730; color: #fff;}
    .stCheckbox label {color: #fff;}
    </style>
""", unsafe_allow_html=True)

# Initialize session
if 'session_id' not in st.session_state:
    with st.spinner("Initializing session..."):
        resp = requests.post(f"{SERVER_URL}/init_session")
        if resp.status_code == 200:
            st.session_state.session_id = resp.json()['session_id']
        else:
            st.error("Failed to initialize session")
            st.stop()

session_id = st.session_state.session_id

# Sidebar
with st.sidebar:
    st.header("Input")
    clinical_text = st.text_area("Enter clinical text", height=200, placeholder="e.g., I have a fever and depression...")
    use_lr = st.checkbox("Logistic Regression (LR)", value=True)
    use_xgb = st.checkbox("XGBoost (XGB)", value=True)
    process_button = st.button("Process")

# Main Processing
if process_button:
    if not clinical_text:
        st.error("Please enter some clinical text")
    elif not (use_lr or use_xgb):
        st.error("Please select at least one model")
    else:
        # Extract symptoms for encrypted inference
        with st.spinner("Extracting symptoms..."):
            symptoms = extract_valid_symptoms(clinical_text)
        st.success("Extracted Symptoms: " + ", ".join(symptoms))

        # Selected models
        selected_models = []
        if use_lr:
            selected_models.append("LR")
        if use_xgb:
            selected_models.append("XGB")

        # Encrypted Inference
        enc_results = {}
        enc_times = {}
        clients = {}
        encrypted_data_dict = {}
        public_keys_dict = {}
        encryption_times = {}

        for model_type in selected_models:
            with st.spinner(f"Preparing encrypted data for {model_type}..."):
                client = FHEMedicalClient(session_id, model_type)
                clients[model_type] = client
                pub_key = client.load_eval_keys()
                ciphertext, enc_time = client.process_and_encrypt(symptoms)
                encrypted_data_dict[model_type] = base64.b64encode(ciphertext).decode("utf-8")
                public_keys_dict[model_type] = base64.b64encode(pub_key).decode("utf-8")
                encryption_times[model_type] = enc_time

        if selected_models:
            # Construct payload for encrypted inference
            enc_payload = {
                "session_id": session_id,
                "model_type": ", ".join(selected_models),
            }
            for model_type in selected_models:
                enc_payload[f"encrypted_data_{model_type.lower()}"] = encrypted_data_dict[model_type]
                enc_payload[f"public_key_{model_type.lower()}"] = public_keys_dict[model_type]

            # Single request to /predict_encrypted
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

        # Plaintext Inference
        pt_results = {}
        pt_times = {}
        if selected_models:
            # Construct payload for plaintext inference
            pt_payload = {
                "clinical_text": clinical_text,
                "model_type": ", ".join(selected_models)
            }
            # Single request to /predict_plaintext
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

        # Display Results
        for model_type in selected_models:
            st.subheader(f"{model_type} Model")
            col1, col2 = st.columns(2)

            # Encrypted Results
            with col1:
                st.markdown("**Encrypted Inference**")
                # Instead of truncating the base64 encoded ciphertext directly,
                # decode it to bytes, convert to hex, and then truncate the hex string.
                ct_b64 = encrypted_data_dict.get(model_type, "")
                if ct_b64:
                    ct_bytes = base64.b64decode(ct_b64)
                    ct_hex = ct_bytes.hex()
                    truncated_ct = ct_hex[:50] + "..." if len(ct_hex) > 50 else ct_hex
                    st.markdown(f"**Ciphertext (truncated, hex):** `{truncated_ct}`")
                else:
                    st.write("No ciphertext available.")

                if model_type in enc_results:
                    preds = enc_results[model_type]
                    for disease, prob in preds[:3]:
                        st.write(f"{disease}: {prob*100:.2f}%")
                    with st.expander("Timing Details"):
                        times = enc_times[model_type]
                        st.write(f"Encryption: {times['encryption']:.3f}s")
                        st.write(f"Inference: {times['inference']:.3f}s")
                        st.write(f"Decryption: {times['decryption']:.3f}s")
                        st.write(f"Total: {times['total']:.3f}s")
                else:
                    st.write("No results available")

            # Plaintext Results
            with col2:
                st.markdown("**Plaintext Inference**")
                # Display a truncated version of the clinical text
                truncated_text = clinical_text[:50] + "..." if len(clinical_text) > 50 else clinical_text
                st.markdown(f"**Input Text (truncated):** `{truncated_text}`")
                
                if model_type in pt_results:
                    preds = pt_results[model_type]
                    for disease, prob in preds[:3]:
                        st.write(f"{disease}: {prob*100:.2f}%")
                    with st.expander("Timing Details"):
                        st.write(f"Inference: {pt_times[model_type]:.3f}s")
                else:
                    st.write("No results available")

        st.info("Note: Encrypted inference times are higher due to FHE's computational overhead, ensuring data privacy.")

