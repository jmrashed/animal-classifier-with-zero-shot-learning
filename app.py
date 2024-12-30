from flask import Flask, request, jsonify, render_template
import os
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from models.resnet_model import load_model
from utils.preprocess import preprocess_image

# Initialize Flask app
app = Flask(__name__)

# Configurations
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load pre-trained model
model = load_model()
model.eval()  # Set model to evaluation mode

# Add a projection layer to reduce dimensions
class ProjectionLayer(nn.Module):
    def __init__(self, input_dim=2048, output_dim=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 3)  # Final projection to match class embedding dimension
        )
    
    def forward(self, x):
        if x.dim() == 1:  # If input is a 1D tensor
            x = x.unsqueeze(0)  # Add batch dimension
        return self.projection(x)

projection_layer = ProjectionLayer()
projection_layer.eval()

# Class embeddings for Zero-Shot Learning (3-dimensional)
class_embeddings = {
    "cat":      np.array([0.7, 0.1, 0.5]),
    "dog":      np.array([0.8, 0.2, 0.4]),
    "horse":    np.array([0.6, 0.4, 0.3]),
    "zebra":    np.array([0.5, 0.9, 0.6]),
    "elephant": np.array([0.4, 0.8, 0.7])
}

def allowed_file(filename):
    """Check if the file extension is allowed."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path):
    """Extracts features from an image and projects them to match class embedding dimensions."""
    try:
        # Get image tensor from preprocessing
        image_tensor = preprocess_image(image_path)
        
        # Ensure correct tensor type and shape
        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.from_numpy(image_tensor)
        
        # Handle different input shapes
        if image_tensor.dim() == 3:  # If [C, H, W]
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
        elif image_tensor.dim() == 5:  # If [1, 1, C, H, W]
            image_tensor = image_tensor.squeeze(0).squeeze(0)  # Remove extra dimensions
            image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension -> [1, C, H, W]
        
        # Print shape for debugging
        print(f"Input tensor shape: {image_tensor.shape}")
        
        # Move to appropriate device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        projection_layer.to(device)
        image_tensor = image_tensor.to(device)
        
        # Extract and project features
        with torch.no_grad():
            # Get features from the model
            features = model(image_tensor)
            print(f"Features shape before reshape: {features.shape}")
            
            # Ensure features are in the correct shape for projection
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            print(f"Features shape after reshape: {features.shape}")
            
            # Project features to 3 dimensions
            projected_features = projection_layer(features)
            print(f"Projected features shape: {projected_features.shape}")
            
            # Convert to numpy and flatten
            projected_features = projected_features.cpu().numpy().flatten()
        
        return projected_features
    except Exception as e:
        raise Exception(f"Error extracting features: {str(e)}")

def predict_animal(image_path):
    """Predicts the animal class using Zero-Shot Learning."""
    # Get projected features (3-dimensional)
    features = extract_features(image_path)
    
    all_classes = list(class_embeddings.keys())
    all_embeddings = np.array([class_embeddings[cls] for cls in all_classes])

    # Normalize features and embeddings
    features = features / np.linalg.norm(features)
    all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)

    # Compute cosine similarity
    similarity_scores = cosine_similarity([features], all_embeddings)[0]
    predicted_class = all_classes[np.argmax(similarity_scores)]
    confidence = float(similarity_scores.max())
    return predicted_class, confidence

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handles image upload and prediction."""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        # Create a secure filename
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Predict the animal class
        predicted_class, confidence = predict_animal(file_path)
        
        # Clean up the uploaded file
        os.remove(file_path)
        
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File is too large"}), 413

if __name__ == '__main__':
    app.run(debug=True, port=5001)