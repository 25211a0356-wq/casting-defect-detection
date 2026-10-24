import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image

# Optimize CPU threads for free tier memory limits
tf.config.threading.set_inter_op_parallelism_threads(1)
tf.config.threading.set_intra_op_parallelism_threads(1)

app = Flask(__name__)

# Load model once when server starts
MODEL_PATH = 'models/casting_model.keras'
model = tf.keras.models.load_model(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No image file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Process image
        img = Image.open(file.stream).convert('RGB')
        img = img.resize((300, 300))
        
        img_array = np.array(img, dtype=np.float32) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        
        # Run prediction
        predictions = model.predict(img_array)
        score = float(predictions[0][0])
        
        # Classification threshold
        result = "Defective Casting" if score > 0.5 else "Ok / Non-Defective"
        confidence = round(score * 100 if score > 0.5 else (1 - score) * 100, 2)
        
        return jsonify({
            'prediction': result,
            'confidence': f"{confidence}%"
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
