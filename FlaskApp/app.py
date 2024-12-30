from datetime import datetime

import numpy as np
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
from werkzeug.exceptions import RequestEntityTooLarge
from werkzeug.utils import secure_filename
import os
import base64
import torch
from retinaface import RetinaFace
from torchvision import transforms
from PIL import Image
from torch.autograd import Variable
from ResEmoteNet import ResEmoteNet
# Initialize the Flask app
app = Flask(__name__)

# Configure upload folder and allowed file types
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit

# Create the upload folder if it doesn't exist
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Initialize the model
model = ResEmoteNet()

# Load the state_dict into the model
model_path = '../ResEmoteNet_model.pth'
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))  # Load weights
model.eval()  # Set the model to evaluation mode

emotion_labels = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'neutral',
    5: 'sad',
    6: 'surprise'
}
# Emotion detection transformation
transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.RandomCrop(64, padding=4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename == '':
            return "No selected file", 400
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)

            # Redirect to the result page with the filename
            return redirect(url_for('result', filename=filename))

    if 'photo-data' in request.form:
        import base64
        photo_data = request.form['photo-data']
        photo_data = photo_data.split(",")[1]  # Remove the base64 header
        photo_bytes = base64.b64decode(photo_data)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  # e.g., 2024-12-29_15-45-30
        filename = f'captured_photo_{timestamp}.png'
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(filepath, 'wb') as f:
            f.write(photo_bytes)

        # Redirect to the result page with the filename
        return redirect(url_for('result', filename=filename))

    return "Invalid upload or data", 400


@app.route('/result')
def result():
    filename = request.args.get('filename')

    if filename:
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)

        # Extract faces from the image using RetinaFace
        faces = RetinaFace.extract_faces(img_path=file_path, align=True)

        emotions = []

        if len(faces) > 0:
            # Process each face and predict emotions
            for i, face in enumerate(faces):
                face_img = Image.fromarray(face)
                face_img = face_img.convert('RGB')  # Ensure the full image is RGB
                face_img = transform(face_img)  # Apply transformations
                face_img = face_img.unsqueeze(0)  # Add batch dimension

                # Predict emotion for the face
                with torch.no_grad():
                    output = model(face_img)
                    _, predicted = torch.max(output, 1)
                    emotion_index = predicted.item()

                    # Map the predicted index to the emotion label
                    emotion_name = emotion_labels.get(emotion_index, 'Unknown')
                    emotions.append(emotion_name)

        else:
            # If no faces are detected, process the entire image
            full_img = Image.open(file_path)
            full_img = full_img.convert('RGB')  # Ensure the full image is RGB
            full_img = transform(full_img).unsqueeze(0)  # Apply transformation and add batch dimension

            with torch.no_grad():
                output = model(full_img)
                _, predicted = torch.max(output, 1)
                emotion_index = predicted.item()

                # Map the predicted index to the emotion label
                emotion_name = emotion_labels.get(emotion_index, 'Unknown')
                emotions.append(emotion_name)

        # Return the result page with the processed image and emotions
        return render_template('result.html', filename=filename, emotions=emotions)

    return "No image found", 400

@app.errorhandler(RequestEntityTooLarge)
def handle_large_file(error):
    return "File is too large. Maximum size is 16 MB.", 413

# Route to serve the uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
