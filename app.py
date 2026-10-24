import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import numpy as np
from PIL import Image

# Force low-memory CPU thread settings for free hosting
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

# Tell Flask explicitly where templates/html are located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATE_DIR = os.path.join(BASE_DIR, 'templates')

app = Flask(__name__, template_folder=TEMPLATE_DIR)
CORS(app)  # Prevents cross-origin browser blocking

# Load model once at startup safely
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'casting_model.keras')

model = None
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
elif os.path.exists('casting_model.keras'):
    model = tf.keras.models.load_model('casting_model.keras')
else:
    print("WARNING: Model file not found at expected paths!")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model file (.keras) is missing or failed to load on server.'}), 500

    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded in form submission.'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected.'}), 400

    try:
        # Preprocess Image - Fixed to (224, 224) for MobileNetV2
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((224, 224))
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Predict
        prediction = model.predict(img_array)
        score = float(prediction[0][0])
        
        result = "Defective Casting" if score > 0.5 else "Ok / Non-Defective"
        confidence = round(score * 100 if score > 0.5 else (1 - score) * 100, 2)
        
        return jsonify({
            'success': True,
            'prediction': result,
            'confidence': f"{confidence}%"
        })

    except Exception as e:
        return jsonify({'error': f"Processing error: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
