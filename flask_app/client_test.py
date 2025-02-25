# client_test.py
from client_handler import FHEMedicalClient
import base64
import requests
import sys
import shutil

SERVER_URL = "http://127.0.0.1:5000"
STATIC_SESSION_ID = "test_session_123"  # Static session ID for all test runs
DUMMY_TEXT = "I have no headachees but I do have a fevver and depression. I do experience anxiety as well as abnormal involuntary movements. I have dont feel shortness of breath as well."

def cleanup_previous_session(client):
    """Remove existing keys from previous test runs"""
    try:
        if client.client.key_dir.exists():
            shutil.rmtree(client.client.key_dir)
            print(f"♻️  Cleaned up previous session: {client.session_id}")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {str(e)}")

def test_full_workflow():
    """End-to-end test with static session ID"""
    try:
        # 1. Initialize client with static session ID
        print("=== Initializing Client with Static Session ID ===")
        client = FHEMedicalClient(STATIC_SESSION_ID)
        print(f"✅ Client created with session ID: {STATIC_SESSION_ID}")
        
        # 2. Cleanup previous test artifacts
        cleanup_previous_session(client)
        
        # 3. Generate fresh keys
        print("\n=== Generating Encryption Keys ===")
        try:
            client.load_eval_keys()
            print("✅ Keys generated successfully")
        except Exception as e:
            print(f"❌ Key generation failed: {str(e)}")
            return False

        # 4. Test NLP processing
        print("\n=== Testing NLP Processing ===")
        symptoms = client.nlp_process(DUMMY_TEXT)
        print("Extracted symptoms:", symptoms)
        
        for symptom in symptoms:
            if symptom not in client.symptom_columns:
                print(f"❌ Invalid symptom: {symptom}")
                return False
        print("✅ Symptoms validated")
            
        # 5. Test feature engineering and encryption
        print("\n=== Testing Feature Encryption ===")
        try:
            encrypted_data = client.process_and_encrypt(symptoms)
            print(f"✅ Encryption successful ({len(encrypted_data)} bytes)")
        except Exception as e:
            print(f"❌ Encryption failed: {str(e)}")
            return False

        # 6. Test prediction endpoint
        print("\n=== Testing Prediction Request ===")
        try:
            response = requests.post(
                f"{SERVER_URL}/predict",
                json={
                    "session_id": STATIC_SESSION_ID,
                    "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8')
                }
            )
            if response.status_code != 200:
                print(f"❌ Prediction failed ({response.status_code}): {response.text}")
                return False
            print("✅ Prediction request successful")
        except Exception as e:
            print(f"❌ Prediction error: {str(e)}")
            return False

        # 7. Test decryption
        print("\n=== Testing Result Decryption ===")
        try:
            encrypted_result = base64.b64decode(response.json()['encrypted_result'])
            predictions = client.decrypt_result(encrypted_result)
            
            print("Top Predictions:")
            for disease, prob in predictions[:3]:
                print(f" - {disease}: {prob*100:.2f}%")
            return True
            
        except Exception as e:
            print(f"❌ Decryption failed: {str(e)}")
            return False

    except Exception as e:
        print(f"❌ Critical error: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Medical Diagnosis FHE Test (Static Session) ===")
    success = test_full_workflow()
    
    if success:
        print("\n=== End-to-End Test Successful ===")
        sys.exit(0)
    else:
        print("\n=== Test Failed ===")
        sys.exit(1)