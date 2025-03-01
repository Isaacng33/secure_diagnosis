from concrete.ml.deployment import FHEModelClient
from pathlib import Path
import numpy as np
import joblib
from nlp import extract_valid_symptoms
from concrete.ml.common.serialization.loaders import load
import time

class FHEMedicalClient:
    def __init__(self, session_id, model_type):
        """
        Initialize the FHEMedicalClient with a session ID and model type.
        
        Args:
            session_id (str): Unique identifier for the session.
            model_type (str): Type of model to use ('LR' for Logistic Regression, 'XGB' for XGBoost).
        """
        self.session_id = session_id
        if model_type =='LR':
            self.path_dir = Path('/home/isaacng33/individual_project/streamlit_app/artifacts/encrypted/LR/')
            self.key_dir = Path(f'/home/isaacng33/individual_project/streamlit_app/client_storage/client_keys/LR/{session_id}')
        elif model_type == 'XGB':
            self.path_dir = Path('/home/isaacng33/individual_project/streamlit_app/artifacts/encrypted/XGB/')
            self.key_dir = Path(f'/home/isaacng33/individual_project/streamlit_app/client_storage/client_keys/XGB/{session_id}')
        else:
            raise ValueError("Unsupported model type. Use 'LR' or 'XGB'.")
        
        # Initialize FHEModelClient
        self.client = FHEModelClient(
            path_dir=self.path_dir,
            key_dir=self.key_dir
        )
 
        self.symptom_columns = joblib.load('/home/isaacng33/individual_project/streamlit_app/artifacts/symptom_columns.pkl')
        self.le = joblib.load('/home/isaacng33/individual_project/streamlit_app/artifacts/label_encoder.pkl')

        # Load model class labels
        if model_type == 'LR':
            self.class_labels_path = '/home/isaacng33/individual_project/streamlit_app/artifacts/models/compiled_lr_model.json'
        elif model_type == 'XGB':
            self.class_labels_path = '/home/isaacng33/individual_project/streamlit_app/artifacts/models/compiled_xgb_model.json'

        self.class_lables = Path(self.class_labels_path)
        with self.class_lables.open('r') as f:
            self.model = load(f)

    def load_eval_keys(self):
        """Load evaluation keys from file"""
        return self.client.get_serialized_evaluation_keys()

    def nlp_process(self, clinical_text):
        """Extract symptoms from clinical text"""
        return extract_valid_symptoms(clinical_text)

    def process_and_encrypt(self, symptoms):
        start_time = time.time()
        features = self._create_feature_vector(symptoms)
        encrypted_data = self.client.quantize_encrypt_serialize(features)
        end_time = time.time()
        encryption_time = end_time - start_time
        # Optionally return encryption_time alongside encrypted_data
        return encrypted_data, encryption_time

    def _create_feature_vector(self, input_symptoms):
        """Create numpy feature vector from symptoms"""
        features = np.zeros(len(self.symptom_columns), dtype=np.float32)
        for symptom in input_symptoms:
            if symptom in self.symptom_columns:
                idx = self.symptom_columns.index(symptom)
                features[idx] = 1
        return features.reshape(1, -1)

    def decrypt_result(self, encrypted_result):
        """Decrypt server response"""
        decrypted_output = self.client.deserialize_decrypt_dequantize(encrypted_result)
        decoded_classes = self.le.inverse_transform(self.model.classes_)
        predicted_results = list(zip(decoded_classes, decrypted_output[0]))
        predicted_results.sort(key=lambda x: x[1], reverse=True)
        return predicted_results