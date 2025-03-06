import re

def parse_metrics_file(filepath):
    """
    Reads a metrics text file and extracts:
      - Training Time (seconds)
      - Compile Time (seconds)
      - Prediction Time (seconds)
      - Accuracy
      - Precision
      - Recall
      - F1 Score

    Returns a dict with these values or None if not found.
    """
    metrics = {
        "training_time": None,
        "compile_time": None,
        "prediction_time": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None
    }

    with open(filepath, 'r') as f:
        content = f.read()

    # 1) Training Time
    train_match = re.search(r"Training Time\s*:\s*([\d\.]+)\s*seconds", content)
    if train_match:
        metrics["training_time"] = float(train_match.group(1))

    # 2) Compile Time
    compile_match = re.search(r"Compile Time\s*:\s*([\d\.]+)\s*seconds", content)
    if compile_match:
        metrics["compile_time"] = float(compile_match.group(1))

    # 3) Prediction Time
    pred_match = re.search(r"Prediction Time\s*:\s*([\d\.]+)\s*seconds", content)
    if pred_match:
        metrics["prediction_time"] = float(pred_match.group(1))

    # 4) Accuracy, Precision, Recall, F1 Score
    for key in ["Accuracy", "Precision", "Recall", "F1 Score"]:
        pattern = rf"{key}\s*:\s*([\d\.]+)"
        match = re.search(pattern, content)
        if match:
            store_key = key.lower().replace(" ", "_")  # e.g. 'accuracy', 'f1_score'
            metrics[store_key] = float(match.group(1))

    return metrics
