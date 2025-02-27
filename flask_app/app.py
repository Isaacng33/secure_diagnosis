from flask import Flask, request, jsonify, session
import logging
import base64
import os
from concrete.ml.deployment import FHEModelServer
from client_handler import FHEMedicalClient

# Initialize logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = os.urandom(24)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB limit

try:
    # Initialize FHE Server first
    logger.info("Attempting to load FHE Server...")
    lr_server = FHEModelServer('/home/isaacng33/individual_project/flask_app/artifacts/encrypted/LR')
    lr_server.load()
    logger.info("FHE Server loaded successfully")
    
except Exception as e:
    logger.error(f"FATAL ERROR: Failed to initialize FHE Server - {str(e)}")
    raise SystemExit(1)  # Exit immediately if server fails


@app.route('/init_session', methods=['POST'])
def init_session():
    """Initialize new client session with fresh keys"""
    try:
        session_id = base64.urlsafe_b64encode(os.urandom(16)).decode('utf-8')
        session['session_id'] = session_id
        
        client = FHEMedicalClient(session_id)
        client.load_eval_keys()
            
        return jsonify({'session_id': session_id}), 200
    
    except Exception as e:
        return jsonify({'error': f"Key generation failed: {str(e)}"}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Handle encrypted medical diagnosis request"""
    try:
        # Validate session
        session_id = request.json.get('session_id')
        if not session_id:
            return jsonify({'error': 'Invalid session'}), 401
            
        # Load evaluation keys
        client = FHEMedicalClient(session_id)
        eval_keys = client.load_eval_keys()

        # Get and validate encrypted data
        if 'encrypted_data' not in request.json:
            return jsonify({'error': 'Missing encrypted data'}), 400
            
        encrypted_data = base64.b64decode(request.json['encrypted_data'])
        
        # Execute FHE inference
        encrypted_result = lr_server.run(encrypted_data, eval_keys)
        
        return jsonify({
            'encrypted_result': base64.b64encode(encrypted_result).decode('utf-8')
        }), 200

    except FileNotFoundError:
        return jsonify({'error': 'Session keys expired'}), 401
    except Exception as e:
        return jsonify({'error': f"Prediction failed: {str(e)}"}), 500

if __name__ == '__main__':
    try:
        logger.info("Starting Flask server...")
        app.run(
            host='0.0.0.0', 
            port=5000, 
            debug=True,  # Enable debug mode
            use_reloader=False  # Disable reloader for FHE compatibility
        )
    except Exception as e:
        logger.error(f"Server crashed: {str(e)}")