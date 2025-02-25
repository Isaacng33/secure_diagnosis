from concrete.ml.deployment import FHEModelClient
from pathlib import Path
import numpy as np
import os
import joblib
from nlp_pipeline import nlp
from concrete.ml.common.serialization.loaders import load

class FHEMedicalClient:
    def __init__(self, session_id):
        self.session_id = session_id
        self.client = FHEModelClient(
            path_dir=Path('/home/isaacng33/individual_project/flask_app/artifacts/encrypted'),
            key_dir=Path(f'/home/isaacng33/individual_project/flask_app/artifacts/client_keys/{session_id}')
        )
        os.makedirs(f'/home/isaacng33/individual_project/flask_app/artifacts/client_keys/{session_id}', exist_ok=True)
        self.symptom_columns = joblib.load('/home/isaacng33/individual_project/flask_app/artifacts/symptom_columns.pkl')
        self.le = joblib.load('/home/isaacng33/individual_project/flask_app/artifacts/label_encoder.pkl')
        fhe_model_path = Path('/home/isaacng33/individual_project/flask_app/artifacts/evaluation_fhe_encrypted_model.json')
        with fhe_model_path.open('r') as f:
            self.model = load(f)

    def load_eval_keys(self):
        """Load evaluation keys from file"""
        return self.client.get_serialized_evaluation_keys()

    def nlp_process(self, clinical_text):
        """Extract symptoms from clinical text"""
        return nlp.extract_valid_symptoms(clinical_text)

    def process_and_encrypt(self, symptoms):
        """Create Features and encrypt it"""
        # Step 1: Create feature vector
        features = self._create_feature_vector(symptoms)
        
        # Step 2: Encrypt features
        return self.client.quantize_encrypt_serialize(features)

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