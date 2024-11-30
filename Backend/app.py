from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import logging
from Model.Script import load_model, preprocess_and_predict
from flask_cors import CORS
import traceback

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Configuration
UPLOAD_FOLDER = './uploads'
MODEL_PATH = './Model/alzheimer_model.pth'
ALLOWED_EXTENSIONS = {'mgz'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model and device
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    model, device = load_model(MODEL_PATH)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error("Failed to load model. Check the model path and file integrity.")
    logger.error(traceback.format_exc())
    model, device = None, None


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file is in the request
        if 'file' not in request.files:
            logger.error("No file part in the request.")
            return jsonify({'error': 'No file part'}), 400

        file = request.files['file']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            logger.info(f"File saved to {filepath}")

            # Get additional inputs
            age = request.form.get('age', type=float)
            sex = request.form.get('sex', type=int)

            if age is None or sex is None:
                logger.error("Missing age or sex input.")
                return jsonify({'error': 'Age and sex are required'}), 400

            # Perform prediction
            if model is None:
                logger.error("Model is not loaded.")
                return jsonify({'error': 'Model not loaded. Please contact the administrator.'}), 500

            try:
                prediction = preprocess_and_predict(filepath, age, sex, model, device)
                logger.info("Prediction successful.")
                return jsonify({'prediction': prediction})
            except FileNotFoundError as e:
                logger.error(f"File not found: {e}")
                logger.error(traceback.format_exc())
                return jsonify({'error': f'File not found: {str(e)}'}), 500
            except ValueError as e:
                logger.error(f"Value error: {e}")
                logger.error(traceback.format_exc())
                return jsonify({'error': f'Value error: {str(e)}'}), 500
            except Exception as e:
                logger.error(f"Error during prediction: {e}")
                logger.error(traceback.format_exc())
                return jsonify({'error': f'Prediction failed: {str(e)}'}), 500

        logger.error("Invalid file type. Only .mgz files are supported.")
        print("Invalid file type. Only .mgz files are supported.")
        return jsonify({'error': 'Invalid file type. Please upload an .mgz file.'}), 400

    except Exception as e:
        logger.error(f"An unknown error occurred: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'An unknown error occurred: {str(e)}'}), 500


if __name__ == '__main__':
    app.run(debug=True)
