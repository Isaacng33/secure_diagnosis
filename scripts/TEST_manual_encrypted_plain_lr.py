import joblib
from pathlib import Path
from concrete.ml.common.serialization.loaders import load
from TOOLS_create_vec import create_feature_vector

# Load Artifacts: Model, Label Encoder
fhe_model_path = Path('./models/compiled_lr_model.json')
with fhe_model_path.open('r') as f:
    model = load(f)
le = joblib.load('./models/label_encoder.pkl')

# Get User input
input_symptoms = [
    'anxiety and nervousness', 
    'depression', 
    'shortness of breath', 
    'depressive or psychotic symptoms', 
    'dizziness', 
    'insomnia', 
    'palpitations']

input_vector = create_feature_vector(input_symptoms)

# Predict the disease using the trained model (Top 3 Probability Classes).
predicted_proba = model.predict_proba(input_vector)
decoded_classes = le.inverse_transform(model.classes_)
predicted_results = list(zip(decoded_classes, predicted_proba[0]))
predicted_results.sort(key=lambda x: x[1], reverse=True)

# Display the top 3 predicted diseases with probabilities.
print("Top 3 Predicted Diseases:")
for disease, probability in predicted_results[:3]:
    print(f"{disease}: {(probability*100):.2f}%")

print(model.coef_)
print(model.intercept_)
