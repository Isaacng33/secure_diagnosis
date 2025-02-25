import pandas as pd
import joblib

# Load the Symptom Columns
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