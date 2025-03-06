from flask import Flask, request, jsonify
import logging
import base64
import os
import time
import joblib
import numpy as np
from concrete.ml.deployment import FHEModelServer
from nlp import extract_valid_symptoms, create_feature_vector

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)

# 1. Load FHE Servers (LR and DP)
try:
    logger.info("Attempting to load LR FHE Server...")
    lr_server = FHEModelServer('/home/isaacng33/individual_project/streamlit_app/artifacts/encrypted/LR')
    lr_server.load()
    logger.info("LR FHE Server loaded successfully")
except Exception as e:
    logger.error(f"FATAL ERROR: Failed to initialize LR FHE Server - {str(e)}")
    raise SystemExit(1)

try:
    logger.info("Attempting to load DP FHE Server...")
    dp_server = FHEModelServer('/home/isaacng33/individual_project/streamlit_app/artifacts/encrypted/DP')
    dp_server.load()
    logger.info("DP FHE Server loaded successfully")
except Exception as e:
    logger.error(f"FATAL ERROR: Failed to initialize DP FHE Server - {str(e)}")
    raise SystemExit(1)

# Load Plaintext Model
try:
    logger.info("Loading Plaintext LR model...")
    lr_plain_model = joblib.load('/home/isaacng33/individual_project/streamlit_app/artifacts/models/final_logistic_model.pkl')
    logger.info("Plaintext LR loaded.")
except Exception as e:
    logger.warning(f"Could not load LR plaintext model: {e}")
    lr_plain_model = None

try:
    logger.info("Loading Plaintext DP model...")
    dp_plain_model = joblib.load('/home/isaacng33/individual_project/streamlit_app/artifacts/models/final_plaintext_dp_logistic_model.pkl')
    logger.info("Plaintext DP loaded.")
except Exception as e:
    logger.warning(f"Could not load DP plaintext model: {e}")
    dp_plain_model = None

# For plaintext inference, we also need the label encoder and symptom columns
try:
    le = joblib.load('/home/isaacng33/individual_project/streamlit_app/artifacts/label_encoder.pkl')
    symptom_columns = joblib.load('/home/isaacng33/individual_project/streamlit_app/artifacts/symptom_columns.pkl')
except Exception as e:
    logger.warning(f"Could not load label encoder or symptom columns: {e}")
    le = None
    symptom_columns = None

# Routes
@app.route('/init_session', methods=['POST'])
def init_session():
    """Initialize new client session and return a session_id. No key loading on the server side."""
    try:
        session_id = base64.urlsafe_b64encode(os.urandom(16)).decode('utf-8')
        return jsonify({'session_id': session_id}), 200
    except Exception as e:
        logger.error(f"Key generation failed: {e}")
        return jsonify({'error': f"Key generation failed: {str(e)}"}), 500

@app.route('/predict_encrypted', methods=['POST'])
def predict_encrypted():
    """
    Handle encrypted medical diagnosis request for one or more models.

    Expects JSON with:
      session_id: str (for user/session tracking, though not used for keys here)
      model_type: 'LR', 'DP', or combinations like 'LR, DP'
      encrypted_data_<model>: base64-encoded ciphertext (e.g., 'encrypted_data_dp')
      public_key_<model>: base64-encoded public/evaluation key (e.g., 'public_key_dp')

    Returns JSON of the form:
      {
        "results": {
          "LR": {
            "encrypted_result": base64-encoded str,
            "inference_time": float
          },
          "DP": {
            "encrypted_result": base64-encoded str,
            "inference_time": float
          }
        }
      }
    """
    try:
        data = request.json

        # 1. Validate session_id (optional if you want user tracking)
        session_id = data.get('session_id')
        if not session_id:
            return jsonify({'error': 'Invalid session'}), 401

        # 2. Parse model_type (could be "LR", "DP", or combinations)
        model_type_str = data.get('model_type', '').strip()
        if not model_type_str:
            return jsonify({'error': 'Missing or empty model_type'}), 400

        requested_models = [m.strip() for m in model_type_str.split(',')]
        valid_models = ['LR', 'DP']
        for mt in requested_models:
            if mt not in valid_models:
                return jsonify({'error': f'Unsupported model type: {mt}'}), 400

        # 3. Retrieve ciphertexts and public keys for each model
        ciphertexts = {}
        public_keys = {}

        for mt in requested_models:
            enc_data_key = f'encrypted_data_{mt.lower()}'
            pub_key_key = f'public_key_{mt.lower()}'
            enc_data_b64 = data.get(enc_data_key)
            pub_key_b64 = data.get(pub_key_key)
            if not enc_data_b64 or not pub_key_b64:
                return jsonify({'error': f'Missing {enc_data_key} or {pub_key_key} for {mt} model'}), 400
            ciphertexts[mt] = base64.b64decode(enc_data_b64)
            public_keys[mt] = base64.b64decode(pub_key_b64)

        # 4. Perform homomorphic inference for each requested model
        results = {}
        for mt in requested_models:
            if mt == 'LR':
                server = lr_server
            elif mt == 'DP':
                server = dp_server

            encrypted_data = ciphertexts[mt]
            eval_key = public_keys[mt]

            start_time = time.time()
            encrypted_result = server.run(encrypted_data, eval_key)
            end_time = time.time()

            results[mt] = {
                'encrypted_result': base64.b64encode(encrypted_result).decode('utf-8'),
                'inference_time': end_time - start_time
            }

        return jsonify({'results': results}), 200

    except Exception as e:
        logger.error(f"Encrypted Prediction failed: {e}")
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

@app.route('/predict_plaintext', methods=['POST'])
def predict_plaintext():
    """
    Handle plaintext diagnosis requests using clinical text input for one or more models.

    Expects JSON: {
      'clinical_text': str,           # Raw clinical text to process
      'model_type': 'LR', 'DP', or combinations like 'LR, DP'
    }

    Returns JSON: {
      'results': {
        'LR': {
          'predictions': [[class, prob], ...],
          'inference_time': float
        },
        'DP': {
          'predictions': [[class, prob], ...],
          'inference_time': float
        }
      }
    }
    """
    try:
        data = request.json

        # 1. Extract clinical text from payload
        clinical_text = data.get('clinical_text', '').strip()
        if not clinical_text:
            return jsonify({'error': 'Missing or empty clinical_text'}), 400

        # 2. Process clinical text into symptoms using the NLP pipeline
        symptoms = extract_valid_symptoms(clinical_text)
        if not symptoms:
            logger.warning(f"No valid symptoms extracted from text: {clinical_text}")

        # 3. Convert symptoms to feature vector
        if symptom_columns is None:
            return jsonify({'error': 'Symptom columns not available on server'}), 500
        feature_vector = create_feature_vector(symptoms)

        # 4. Parse the model_type string
        model_type_str = data.get('model_type', '').strip()
        if not model_type_str:
            return jsonify({'error': 'Missing or empty model_type'}), 400
        requested_models = [m.strip() for m in model_type_str.split(',')]
        valid_models = ['LR', 'DP']
        for mt in requested_models:
            if mt not in valid_models:
                return jsonify({'error': f'Unsupported model type: {mt}'}), 400

        # 5. Store results for each requested model
        all_results = {}

        # 6. Perform predictions for each model
        for mt in requested_models:
            if mt == 'LR':
                if lr_plain_model is None:
                    return jsonify({'error': 'LR plaintext model not loaded on server'}), 500
                model = lr_plain_model
            elif mt == 'DP':
                if dp_plain_model is None:
                    return jsonify({'error': 'DP plaintext model not loaded on server'}), 500
                model = dp_plain_model

            # Predict probabilities
            start_time = time.time()
            probs = model.predict_proba(feature_vector)
            end_time = time.time()

            # Format results
            decoded_classes = le.inverse_transform(model.classes_)
            predicted_results = [(disease, float(prob)) for disease, prob in zip(decoded_classes, probs[0])]
            predicted_results.sort(key=lambda x: x[1], reverse=True)

            all_results[mt] = {
                'predictions': predicted_results,
                'inference_time': end_time - start_time
            }

        return jsonify({'results': all_results}), 200

    except Exception as e:
        logger.error(f"Plaintext prediction failed: {e}")
        return jsonify({'error': f"Plaintext prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    try:
        logger.info("Starting Flask server...")
        app.run(
            host='0.0.0.0',
            port=5000,
        )
    except Exception as e:
        logger.error(f"Server crashed: {str(e)}")