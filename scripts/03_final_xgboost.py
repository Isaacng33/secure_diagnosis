from concrete.ml.sklearn import XGBClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

X = joblib.load('../data/raw/raw_concrete_X.pkl')
y = joblib.load('../data/raw/raw_concrete_y.pkl')

class_counts = y.value_counts()
rare_classes = class_counts[class_counts < 2].index.tolist()

# Create mask to exclude rare classes
mask = ~y.isin(rare_classes)
X_filtered = X[mask]
y_filtered = y[mask]

# Encode disease names to numerical labels.
le = LabelEncoder()
y_encoded = le.fit_transform(y_filtered)

# Train the Logistic Regression model.
model = XGBClassifier(
    n_estimators=75,       # FHE-optimized (balance accuracy/circuit size)
    max_depth=4,           # Critical for FHE performance
    learning_rate=0.1,
    n_bits=3,              # Quantization bits (FHE requirement)
    objective="multi:softmax",
    num_class=len(le.classes_),
    tree_method="hist"     # Essential for large datasets
)
model.fit(X_filtered, y_encoded)

# Save the model, label encoder, and symptom columns.
joblib.dump(model, './models/final_xgb_model.pkl')
joblib.dump(le, './models/xgb_label_encoder.pkl')

print("Model, label encoder saved successfully.")
