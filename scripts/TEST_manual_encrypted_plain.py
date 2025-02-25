import pandas as pd
import joblib

# Load Artifacts: Model, Label Encoder, and Symptom Columns
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

# Get User input
input_symptoms = ['abnormal involuntary movements', 'anxiety and nervousness', 'fever', 'depression']

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