# client_test.py
from client_handler import FHEMedicalClient
import base64
import requests
import sys

SERVER_URL = "http://127.0.0.1:5000"
DUMMY_TEXT =  "I have no headachees but I do have a fevver and depression. I do experience anxiety as well as abnormal involuntary movements. I have dont feel shortness of breath as well."

def test_full_workflow():
    try:
        # 1. Test session initialization
        print("=== Testing Session Initialization ===")
        session_response = requests.post(f"{SERVER_URL}/init_session")
        
        if session_response.status_code != 200:
            print(f"❌ Session init failed: {session_response.text}")
            return False
            
        session_id = session_response.json().get('session_id')
        if not session_id:
            print("❌ No session ID received")
            return False
            
        print(f"✅ Session initialized. Session ID: {session_id[:8]}...")

        # 2. Load client instance
        client = FHEMedicalClient(session_id, model_type='LR')
        
        # 3. Test NLP processing and encryption
        print("\n=== Testing NLP Processing and Encryption ===")
        try:
            symptoms = client.nlp_process(DUMMY_TEXT)
            encrypted_data = client.process_and_encrypt(symptoms)
            print(f"✅ Encryption successful. Data size: {len(encrypted_data)} bytes")
        except Exception as e:
            print(f"❌ Encryption failed: {str(e)}")
            return False

        # 4. Test prediction endpoint
        print("\n=== Testing Prediction Endpoint ===")
        try:
            prediction_response = requests.post(
                f"{SERVER_URL}/predict",
                json={
                    "session_id": session_id,
                    "encrypted_data": base64.b64encode(encrypted_data).decode('utf-8')
                }
            )
            
            if prediction_response.status_code != 200:
                print(f"❌ Prediction failed ({prediction_response.status_code}): {prediction_response.text}")
                return False
                
            print("✅ Prediction request successful")
            
        except requests.exceptions.ConnectionError:
            print("❌ Server connection failed. Is the Flask app running?")
            return False

        # 5. Test decryption
        print("\n=== Testing Result Decryption ===")
        try:
            encrypted_result = base64.b64decode(prediction_response.json()['encrypted_result'])
            predictions = client.decrypt_result(encrypted_result)
            
            if not predictions:
                print("❌ No predictions returned")
                return False
                
            print("✅ Decryption successful. Top predictions:")
            for disease, prob in predictions[:3]:
                print(f" - {disease}: {prob*100:.2f}%")
                
        except Exception as e:
            print(f"❌ Decryption failed: {str(e)}")
            return False

        return True

    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
        return False

if __name__ == "__main__":
    print("=== Starting FHE Medical Diagnosis Test ===")
    success = test_full_workflow()
    
    if success:
        print("\n=== All Tests Passed Successfully ===")
        sys.exit(0)
    else:
        print("\n=== Some Tests Failed ===")
        sys.exit(1)