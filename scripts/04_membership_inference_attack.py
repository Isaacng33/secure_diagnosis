import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
import joblib
import os

# Paths to the models
plaintext_model_path = '/home/isaacng33/individual_project/models/final_logistic_model.pkl'
dp_model_path = '/home/isaacng33/individual_project/models/final_plaintext_dp_logistic_model.pkl'

# Load the dataset
df = pd.read_csv('/home/isaacng33/individual_project/data/raw/dataset_1/newdataset.csv')
X = df.drop('diseases', axis=1)
y = df['diseases']

# Encode labels for multiclass
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split the dataset
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42
)

# Load the models using joblib
plaintext_model = joblib.load(plaintext_model_path)
dp_model = joblib.load(dp_model_path)

# Step 1: Create the attack dataset
# Take a subset of training and test data to create a balanced dataset for the attack
n_attack_samples = min(len(X_train_full), len(X_test_full)) // 2  # Use half of the smaller set
X_train_subset = X_train_full.iloc[:n_attack_samples]
y_train_subset = y_train_full[:n_attack_samples]
X_test_subset = X_test_full.iloc[:n_attack_samples]
y_test_subset = y_test_full[:n_attack_samples]

# Combine into attack dataset
X_attack = pd.concat([X_train_subset, X_test_subset], axis=0).reset_index(drop=True)
y_attack = np.concatenate([y_train_subset, y_test_subset])
membership_labels = np.concatenate([np.ones(n_attack_samples), np.zeros(n_attack_samples)])  # 1 for train, 0 for test

# Step 2: Perform the membership inference attack
def perform_mia(model, X, y, threshold=0.5):
    # Get predicted probabilities for the true class
    probs = model.predict_proba(X)
    # Extract the probability corresponding to the true class for each sample
    true_class_probs = np.array([probs[i, y[i]] for i in range(len(y))])
    # Predict membership: 1 if probability > threshold, 0 otherwise
    membership_predictions = (true_class_probs > threshold).astype(int)
    return membership_predictions, true_class_probs

# Attack on non-DP model
non_dp_predictions, non_dp_probs = perform_mia(plaintext_model, X_attack, y_attack, threshold=0.5)

# Attack on DP model
dp_predictions, dp_probs = perform_mia(dp_model, X_attack, y_attack, threshold=0.5)

# Step 3: Evaluate the attack
def evaluate_mia(predictions, true_labels):
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    return accuracy, precision, recall

# Evaluate non-DP model attack
non_dp_accuracy, non_dp_precision, non_dp_recall = evaluate_mia(non_dp_predictions, membership_labels)

# Evaluate DP model attack
dp_accuracy, dp_precision, dp_recall = evaluate_mia(dp_predictions, membership_labels)

# Save this to result folder
result_dir = '/home/isaacng33/individual_project/results'
result_file = os.path.join(result_dir, 'mia_results.txt')

with open(result_file, 'w') as f:
    f.write("Non-DP Model MIA Results:\n")
    f.write(f"Accuracy: {non_dp_accuracy:.4f}\n")
    f.write(f"Precision: {non_dp_precision:.4f}\n")
    f.write(f"Recall: {non_dp_recall:.4f}\n\n")

    f.write("DP Model MIA Results:\n")
    f.write(f"Accuracy: {dp_accuracy:.4f}\n")
    f.write(f"Precision: {dp_precision:.4f}\n")
    f.write(f"Recall: {dp_recall:.4f}\n\n")

    f.write("Average Confidence Scores:\n")
    f.write(f"Non-DP Model - Training Data: {np.mean(non_dp_probs[:n_attack_samples]):.4f}\n")
    f.write(f"Non-DP Model - Test Data: {np.mean(non_dp_probs[n_attack_samples:]):.4f}\n")
    f.write(f"DP Model - Training Data: {np.mean(dp_probs[:n_attack_samples]):.4f}\n")
    f.write(f"DP Model - Test Data: {np.mean(dp_probs[n_attack_samples:]):.4f}\n")
print(f"Results saved to {result_file}")
