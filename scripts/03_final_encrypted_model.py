import os
from pathlib import Path
import joblib
from concrete.ml.common.serialization.loaders import load
from concrete.ml.deployment import FHEModelDev

X_train = joblib.load('./data/raw/X_train_concrete.pkl')
fhe_model_path = Path('./models/evaluation_fhe_encrypted_model.json')

with fhe_model_path.open('r') as f:
    fhe_model = load(f)

# Recompile the model is needed for fhe inference.
print("Recompiling the FHE model with calibration data from X_train...")
fhe_model.compile(X_train)
print("Compilation complete.")

encrypted_model_path = "./data/encrypted/"
os.makedirs(encrypted_model_path, exist_ok=True)

dev = FHEModelDev(path_dir=encrypted_model_path, model=fhe_model)
dev.save()
