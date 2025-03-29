# Secure Diagnosis Using Fully Homomorphic Encryption

## Overview
This project demonstrates a privacy-preserving healthcare diagnosis system that integrates Natural Language Processing (NLP) for symptom extraction and a machine learning model for disease prediction. The system leverages Fully Homomorphic Encryption (FHE) to perform secure inference on encrypted data, thereby preserving patient confidentiality without significantly compromising model performance.

## Models
The project includes three models:
- **Plaintext Logistic Regression:**  
  A baseline model trained on structured symptom data.
- **FHE Inference Model:**  
  A compiled FHE model (using Concrete ML) that performs inference on encrypted inputs.
- **FHE Inference Differential Privacy Model:**  
  A compiled FHE model (using Concrete ML) that performs inference on encrypted inputs trained with DP.

## Features
- **NLP-Driven Symptom Extraction:**  
  Converts free-text patient inputs into structured feature vectors.
- **Privacy-Preserving Inference:**  
  Uses Fully Homomorphic Encryption to securely process sensitive data.
- **Client-Server Architecture:**  
  Implements an end-to-end workflow—from input encryption, secure server-side processing, to output decryption.

## Getting Started

### Prerequisites
- Python 3.7 or later
- [Concrete ML](https://github.com/zama-ai/concrete-ml) (install via `pip install concrete-ml`)
- Other dependencies as listed in `requirements.txt`

### Installation
**Clone the Repository:**
```bash
git clone https://github.com/Isaacng33/secure_diagnosis.git
cd secure_diagnosis


