import os
import numpy as np
from PIL import Image
from flask import Flask, request, render_template, jsonify
import tensorflow as tf

app = Flask(__name__)

# Load trained MobileNetV2 model
MODEL_PATH = os.path.join("models", "casting_model.keras")
model = tf.keras.models.load_model(MODEL_PATH)

def preprocess_image(image_file):
    """Resize and normalize uploaded image for model prediction."""
    img = Image.open(image_file).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    try:
        processed_img = preprocess_image(file)
        prediction = model.predict(processed_img)[0][0]
        
        # Output interpretation (Sigmoid: > 0.5 is OK, <= 0.5 is Defective)
        if prediction > 0.5:
            result = "Defect-Free (OK)"
            confidence = float(prediction) * 100
        else:
            result = "Defective"
            confidence = (1.0 - float(prediction)) * 100

        return jsonify({
            "status": "success",
            "result": result,
            "confidence": f"{confidence:.2f}%"
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
