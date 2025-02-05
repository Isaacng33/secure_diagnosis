# Secure Diagnosis Using Fully Homomorphic Encryption

## Overview
This project demonstrates a privacy-preserving healthcare diagnosis system that integrates Natural Language Processing (NLP) for symptom extraction and a machine learning model for disease prediction. The system leverages Fully Homomorphic Encryption (FHE) to perform secure inference on encrypted data, thereby preserving patient confidentiality without significantly compromising model performance.

## Models

The project includes three models:
- **Plaintext Logistic Regression:**  
  A baseline model trained on structured symptom data.
- **Encrypted Plaintext Logistic Regression:**  
  A model that demonstrates how data can be encrypted while still being processed in plaintext.
- **FHE Inference Model:**  
  A compiled FHE model (using Concrete ML) that performs inference on encrypted inputs.

## Features

- **NLP-Driven Symptom Extraction:**  
  Converts free-text patient inputs into structured feature vectors.
- **Privacy-Preserving Inference:**  
  Uses Fully Homomorphic Encryption to securely process sensitive data.
- **Client-Server Architecture:**  
  Demonstrates a complete workflow from input encryption to secure inference and decryption.