from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from flask_cors import CORS
import base64
import io
from mtcnn import MTCNN  # Import MTCNN for face detection

app = Flask(__name__)
CORS(app)  # To handle cross-origin requests

# Load the trained model
model = tf.keras.models.load_model('Fine_tuned_skin_tone_classifier.h5')  # Update with your model file path
input_shape = (224, 224)  # Model expects (224, 224, 3)

# Initialize MTCNN detector
detector = MTCNN()

# Map predicted classes to skin tone labels
skin_tone_labels = {
    0: "Black",
    1: "White",
    2: "Brown"
}

# **Face detection using MTCNN**
def detect_face(image):
    faces = detector.detect_faces(image)
    if len(faces) == 0:
        return None  # No face detected

    # Extract bounding box of the first detected face
    x, y, w, h = faces[0]['box']

    # Ensure positive coordinates
    x, y = max(0, x), max(0, y)

    cropped_face = image[y:y + h, x:x + w]
    return cropped_face

# **Preprocessing function (resizes and normalizes the image)**
def preprocess_image(image):
    image = cv2.resize(image, (input_shape[1], input_shape[0]))  # Resize to model's expected shape
    image = np.array(image) / 255.0  # Normalize pixel values between 0-1
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    image = Image.open(file.stream).convert('RGB')
    image_np = np.array(image)

    # **Detect face before passing to the model**
    face = detect_face(image_np)
    if face is None:
        return jsonify({'error': 'No face detected in the image'}), 400

    # **Preprocess the cropped face**
    processed_image = preprocess_image(face)

    # Predict with the model
    predictions = model.predict(processed_image)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Convert uploaded image to base64 for displaying
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    return jsonify({
        'skin_tone': skin_tone_labels.get(predicted_class, "Unknown"),
        'confidence': float(confidence),
        'uploaded_image': image_base64
    })

if __name__ == '__main__':
    app.run(debug=True)
# from flask import Flask, request, jsonify, render_template
# import tensorflow as tf
# import numpy as np
# import cv2
# from PIL import Image
# from flask_cors import CORS
# import base64
# import io
# from mtcnn import MTCNN  # Import MTCNN for face detection
# import random  # For random patch selection

# app = Flask(__name__)
# CORS(app)  # Handle cross-origin requests

# # Load the trained model
# model = tf.keras.models.load_model('Fine_tuned_skin_tone_classifier.h5')  # Update with your model path
# input_shape = (224, 224)  # Model expects (224, 224, 3)

# # Initialize MTCNN detector
# detector = MTCNN()

# # Map predicted classes to skin tone labels
# skin_tone_labels = {
#     0: "Black",
#     1: "White",
#     2: "Brown"
# }

# # **Face detection using MTCNN**
# def detect_face(image):
#     faces = detector.detect_faces(image)
#     if len(faces) == 0:
#         return None  # No face detected

#     # Extract bounding box of the first detected face
#     x, y, w, h = faces[0]['box']
   
#     # Ensure positive coordinates
#     x, y = max(0, x), max(0, y)

#     cropped_face = image[y:y + h, x:x + w]
#     return cropped_face

# # **Extract random skin patch (forehead, cheek, chin)**
# def extract_skin_patch(face_img):
#     h, w, _ = face_img.shape
#     patch_size = int(min(h, w) * 0.3)  # 30% of face size

#     # Randomly select region (forehead, cheek, chin)
#     regions = {
#         "forehead": (int(w * 0.3), int(h * 0.1)),
#         "cheek": (int(w * 0.3), int(h * 0.5)),
#         "chin": (int(w * 0.3), int(h * 0.8))
#     }
   
#     selected_region = random.choice(list(regions.values()))
#     x_start, y_start = selected_region

#     # Ensure patch does not exceed image boundaries
#     x_end = min(x_start + patch_size, w)
#     y_end = min(y_start + patch_size, h)

#     skin_patch = face_img[y_start:y_end, x_start:x_end]
#     return skin_patch

# # **Preprocessing function (resizes and normalizes the image)**
# def preprocess_image(image):
#     image = cv2.resize(image, (input_shape[1], input_shape[0]))  # Resize to model's expected shape
#     image = np.array(image) / 255.0  # Normalize pixel values between 0-1
#     image = np.expand_dims(image, axis=0)  # Add batch dimension
#     return image

# @app.route('/')
# def index():
#     return render_template('index.html')

# @app.route('/predict', methods=['POST'])
# def predict():
#     if 'file' not in request.files:
#         return jsonify({'error': 'No file uploaded'}), 400

#     file = request.files['file']
#     image = Image.open(file.stream).convert('RGB')
#     image_np = np.array(image)

#     # **Detect face before passing to the model**
#     face = detect_face(image_np)
#     if face is None:
#         return jsonify({'error': 'No face detected in the image'}), 400

#     # **Extract random skin patch**
#     skin_patch = extract_skin_patch(face)

#     # **Preprocess the cropped skin patch**
#     processed_image = preprocess_image(skin_patch)

#     # Predict with the model
#     predictions = model.predict(processed_image)
#     predicted_class = np.argmax(predictions[0])
#     confidence = np.max(predictions[0])

#     # Convert uploaded image to base64 for displaying
#     buffered = io.BytesIO()
#     image.save(buffered, format="JPEG")
#     image_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

#     return jsonify({
#         'skin_tone': skin_tone_labels.get(predicted_class, "Unknown"),
#         'confidence': float(confidence),
#         'uploaded_image': image_base64
#     })

# if __name__ == '__main__':
#     app.run(debug=True)
