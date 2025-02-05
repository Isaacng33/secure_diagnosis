import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
import joblib

# Load the dataset
df = pd.read_csv('./data/raw/dataset_1/newdataset.csv')

# First Column (Diseases) is the target variable.
# All other columns are symptoms.
X = df.drop('diseases', axis=1)
y = df['diseases']

# store X and y
joblib.dump(X, './data/raw/raw_X.pkl')
joblib.dump(y, './data/raw/raw_y.pkl')

# Encode disease names to numerical labels.
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Train the Logistic Regression model.
model = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
model.fit(X, y_encoded)

# Save the model, label encoder, and symptom columns.
joblib.dump(model, './models/final_logistic_model.pkl')
joblib.dump(le, './models/label_encoder.pkl')
joblib.dump(X.columns.tolist(), './models/symptom_columns.pkl')

print("Model, label encoder, and symptom columns saved successfully.")
