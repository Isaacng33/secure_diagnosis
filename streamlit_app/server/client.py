import os
import shutil
import base64
import requests
import sys

from client_handler import FHEMedicalClient

SERVER_URL = "http://127.0.0.1:5000"

DUMMY_TEXT = (
    "I have no headachees but I do have a fevver and depression. "
    "I do experience anxiety as well as abnormal involuntary movements. "
    "I dont feel shortness of breath as well."
)

CHECKMARK = "\u2705"  # ✅
CROSSMARK = "\u274C"  # ❌

def print_success(message: str):
    print(f"{CHECKMARK} {message}")

def print_fail(message: str):
    print(f"{CROSSMARK} {message}")

def clear_client_keys():
    """
    Recursively delete all key directories under the 'client_keys' folder
    to ensure a clean slate before testing.
    """
    client_keys_path = "/home/isaacng33/individual_project/streamlit_app/client_storage/client_keys"
    if os.path.exists(client_keys_path):
        for subdir in os.listdir(client_keys_path):
            subdir_path = os.path.join(client_keys_path, subdir)
            if os.path.isdir(subdir_path):
                print(f"Removing directory: {subdir_path}")
                shutil.rmtree(subdir_path)
            else:
                os.remove(subdir_path)
    else:
        print("No client_keys folder found; nothing to clear.")

def test_lr_xgb_combined():
    """
    1) Clear old keys.
    2) /init_session -> server returns session_id (no keys loaded on server).
    3) Locally, create two FHEMedicalClient objects: lr_client, xgb_client.
    4) Generate or load public/eval keys from each client (these are 'public_key_lr' and 'public_key_xgb').
    5) Do local NLP for both, encrypt symptom vectors with each model's keys -> ciphertext_lr, ciphertext_xgb.
    6) POST /predict_encrypted with model_type='LR, XGB', and pass:
         encrypted_data_lr, public_key_lr
         encrypted_data_xgb, public_key_xgb
    7) Decrypt the combined results from the server.
    8) Plaintext compare with /predict_plaintext, model_type='LR, XGB'.
    """
    try:
        print("\n=== Starting Combined LR & XGB FHE Test ===")

        # 1. Clear old keys
        print("\n[1] Clearing existing client keys...")
        clear_client_keys()
        print_success("Client keys cleared")

        # 2. Init session
        print("\n[2] init_session -> retrieving session_id from server...")
        resp_init = requests.post(f"{SERVER_URL}/init_session")
        if resp_init.status_code != 200:
            print_fail(f"init_session failed: {resp_init.text}")
            return False
        
        session_id = resp_init.json().get("session_id")
        if not session_id:
            print_fail("No session ID received from server.")
            return False
        print_success(f"Session initialized: {session_id[:8]}")

        # 3. Create LR & XGB clients on the client side
        lr_client = FHEMedicalClient(session_id, "LR")
        xgb_client = FHEMedicalClient(session_id, "XGB")

        # 4. Load or generate public/eval keys
        print("\n[3] Loading or generating public keys locally for LR & XGB...")
        pub_key_lr = lr_client.load_eval_keys()
        pub_key_xgb = xgb_client.load_eval_keys()
        print_success("Public keys loaded")

        # 5. NLP & encryption for LR
        print("\n[4a] LR local NLP & encryption")
        symptoms_lr = lr_client.nlp_process(DUMMY_TEXT)
        ciphertext_lr, lr_enc_time = lr_client.process_and_encrypt(symptoms_lr)
        print_success(f"LR encryption time: {lr_enc_time:.3f}s, ciphertext size: {len(ciphertext_lr)} bytes")

        # 6. NLP & encryption for XGB
        print("\n[4b] XGB local NLP & encryption")
        symptoms_xgb = xgb_client.nlp_process(DUMMY_TEXT)
        ciphertext_xgb, xgb_enc_time = xgb_client.process_and_encrypt(symptoms_xgb)
        print_success(f"XGB encryption time: {xgb_enc_time:.3f}s, ciphertext size: {len(ciphertext_xgb)} bytes")

        # 7. Single /predict_encrypted call with "LR, XGB"
        print("\n[5] /predict_encrypted => 'LR, XGB'")
        enc_payload = {
            "session_id": session_id,
            "model_type": "LR, XGB",

            # LR
            "encrypted_data_lr": base64.b64encode(ciphertext_lr).decode("utf-8"),
            "public_key_lr": base64.b64encode(pub_key_lr).decode("utf-8"),

            # XGB
            "encrypted_data_xgb": base64.b64encode(ciphertext_xgb).decode("utf-8"),
            "public_key_xgb": base64.b64encode(pub_key_xgb).decode("utf-8")
        }
        resp_enc = requests.post(f"{SERVER_URL}/predict_encrypted", json=enc_payload)
        if resp_enc.status_code != 200:
            print_fail(f"/predict_encrypted failed: {resp_enc.text}")
            return False

        enc_data = resp_enc.json().get("results", {})
        if "LR" not in enc_data or "XGB" not in enc_data:
            print_fail("Missing LR or XGB in encrypted inference response.")
            return False
        print_success("Encrypted inference request successful")

        # 7a. Decrypt LR result
        lr_info = enc_data["LR"]
        lr_enc_result_b64 = lr_info["encrypted_result"]
        lr_inference_time = lr_info["inference_time"]
        lr_enc_result = base64.b64decode(lr_enc_result_b64)
        lr_preds = lr_client.decrypt_result(lr_enc_result)
        if not lr_preds:
            print_fail("No LR encrypted predictions returned.")
            return False

        print("\n--- LR Encrypted Predictions ---")
        for disease, prob in lr_preds[:3]:
            print(f"   {disease}: {prob*100:.2f}%")
        print_success(f"LR FHE server inference time: {lr_inference_time:.3f}s")

        # 7b. Decrypt XGB result
        xgb_info = enc_data["XGB"]
        xgb_enc_result_b64 = xgb_info["encrypted_result"]
        xgb_inference_time = xgb_info["inference_time"]
        xgb_enc_result = base64.b64decode(xgb_enc_result_b64)
        xgb_preds = xgb_client.decrypt_result(xgb_enc_result)
        if not xgb_preds:
            print_fail("No XGB encrypted predictions returned.")
            return False

        print("\n--- XGB Encrypted Predictions ---")
        for disease, prob in xgb_preds[:3]:
            print(f"   {disease}: {prob*100:.2f}%")
        print_success(f"XGB FHE server inference time: {xgb_inference_time:.3f}s")

        # 8. Plaintext Inference
        print("\n[6] /predict_plaintext => 'LR, XGB' for comparison")
        pt_symptoms = lr_client.nlp_process(DUMMY_TEXT)
        pt_feature_vec = lr_client._create_feature_vector(pt_symptoms).tolist()

        pt_payload = {
            "feature_vector": pt_feature_vec,
            "model_type": "LR, XGB"
        }
        resp_pt = requests.post(f"{SERVER_URL}/predict_plaintext", json=pt_payload)
        if resp_pt.status_code != 200:
            print_fail(f"/predict_plaintext failed: {resp_pt.text}")
            return False

        pt_data = resp_pt.json().get("results", {})
        if "LR" not in pt_data or "XGB" not in pt_data:
            print_fail("Missing LR or XGB in plaintext response.")
            return False
        print_success("Plaintext inference request successful")

        # LR Plaintext
        lr_pt_info = pt_data["LR"]
        lr_pt_preds = lr_pt_info["predictions"]
        lr_pt_time = lr_pt_info["inference_time"]
        if not lr_pt_preds:
            print_fail("No LR plaintext predictions.")
            return False

        print("\n--- LR Plaintext Predictions ---")
        for disease, prob in lr_pt_preds[:3]:
            print(f"   {disease}: {prob*100:.2f}%")
        print_success(f"LR plaintext inference time: {lr_pt_time:.3f}s")

        # XGB Plaintext
        xgb_pt_info = pt_data["XGB"]
        xgb_pt_preds = xgb_pt_info["predictions"]
        xgb_pt_time = xgb_pt_info["inference_time"]
        if not xgb_pt_preds:
            print_fail("No XGB plaintext predictions.")
            return False

        print("\n--- XGB Plaintext Predictions ---")
        for disease, prob in xgb_pt_preds[:3]:
            print(f"   {disease}: {prob*100:.2f}%")
        print_success(f"XGB plaintext inference time: {xgb_pt_time:.3f}s")

        print("\n=== Combined LR & XGB test completed successfully! ===")
        return True

    except Exception as e:
        print_fail(f"Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_lr_xgb_combined()
    if success:
        print_success("\nAll Tests Passed Successfully\n")
        sys.exit(0)
    else:
        print_fail("\nSome Tests Failed\n")
        sys.exit(1)
