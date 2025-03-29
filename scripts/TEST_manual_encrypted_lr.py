import os
import joblib
import numpy as np
from pathlib import Path
from concrete.ml.deployment import FHEModelClient, FHEModelServer
from concrete.ml.common.serialization.loaders import load
from TOOLS_create_vec import create_feature_vector

fhe_model_path = Path('./models/compiled_lr_model.json')
with fhe_model_path.open('r') as f:
    model = load(f)
le = joblib.load('./models/label_encoder.pkl')
symptom_columns = joblib.load('./models/symptom_columns.pkl')

# Load Artifacts: Model and Keys
encrypted_model_path = "./data/encrypted/LR/"
key_path = "./data/keys/LR/"
os.makedirs(key_path, exist_ok=True)

# Initialize the FHE Model Client
client = FHEModelClient(path_dir=encrypted_model_path, key_dir=key_path)
serialized_evaluation_keys = client.get_serialized_evaluation_keys()

# Initialize the FHE Model Server
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

print(model.coef_)
print(model.intercept_)
