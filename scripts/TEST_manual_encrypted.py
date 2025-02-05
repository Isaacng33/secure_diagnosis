import os
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from concrete.ml.deployment import FHEModelClient, FHEModelServer

model = joblib.load('./models/evaluation_plaintext_encrypted_model.pkl')
le = joblib.load('./models/label_encoder.pkl')
symptom_columns = joblib.load('./models/symptom_columns.pkl')

# Function to Create a Feature Vector from Input Symptoms
def create_feature_vector(input_symptoms):
    """
    Given a list of symptom names, create a DataFrame row
    with 1 for present symptoms and 0 for absent symptoms.
    """
    # Initialize all symptom features to 0.
    feature_dict = {symptom: 0 for symptom in symptom_columns}
    
    # Mark the symptoms provided as 1.
    for symptom in input_symptoms:
        if symptom in feature_dict:
            feature_dict[symptom] = 1
        else:
            print(f"Warning: Symptom '{symptom}' not found in dataset features.")
    
    # Return a DataFrame matching the model input shape.
    return pd.DataFrame([feature_dict])

# Load Artifacts: Model and Keys
encrypted_model_path = "./data/encrypted/"
key_path = "./data/keys/"
os.makedirs(key_path, exist_ok=True)

client = FHEModelClient(path_dir=encrypted_model_path, key_dir=key_path)
serialized_evaluation_keys = client.get_serialized_evaluation_keys()

server = FHEModelServer(path_dir=encrypted_model_path)
server.load()

# Get User input
input_symptoms = [
    'anxiety and nervousness', 
    'depression', 
    'shortness of breath', 
    'depressive or psychotic symptoms', 
    'dizziness', 
    'insomnia', 
    'palpitations']

# Create a Feature Vector
input_vector = create_feature_vector(input_symptoms)
input_array = np.asarray(input_vector.values, dtype=np.float32)

# Encrypt the input vector
encrypted_input = client.quantize_encrypt_serialize(input_array)

# Send the encrypted input to the server for prediction
encrypted_output = server.run(encrypted_input, serialized_evaluation_keys)

# Decrypt the encrypted output
decrypted_output = client.deserialize_decrypt_dequantize(encrypted_output)
decoded_classes = le.inverse_transform(model.classes_)
predicted_results = list(zip(decoded_classes, decrypted_output[0]))
predicted_results.sort(key=lambda x: x[1], reverse=True)

print("Top predicted classes (from FHE inference):")
for disease, probability in predicted_results[:3]:
    print(f"{disease}: {(probability*100):.2f}%")
