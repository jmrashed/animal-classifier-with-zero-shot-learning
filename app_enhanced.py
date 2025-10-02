from flask import Flask, request, jsonify, render_template
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import torch
import torch.nn as nn
import numpy as np
import json
import logging
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity
from models.resnet_model import load_model
from utils.preprocess import preprocess_image

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-key-change-in-production')

# Rate limiting
limiter = Limiter(
    app,
    key_func=get_remote_address,
    default_limits=["100 per hour"]
)

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configurations
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
CLASSES_FILE = 'classes.json'
HISTORY_FILE = 'history.json'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load model
model = load_model()
model.eval()

class ProjectionLayer(nn.Module):
    def __init__(self, input_dim=2048, output_dim=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, 3)
        )
    
    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.projection(x)

projection_layer = ProjectionLayer()
projection_layer.eval()

def load_classes():
    """Load class embeddings from file"""
    if os.path.exists(CLASSES_FILE):
        with open(CLASSES_FILE, 'r') as f:
            data = json.load(f)
            return {k: np.array(v) for k, v in data.items()}
    return {
        "cat": np.array([0.7, 0.1, 0.5]),
        "dog": np.array([0.8, 0.2, 0.4]),
        "horse": np.array([0.6, 0.4, 0.3]),
        "zebra": np.array([0.5, 0.9, 0.6]),
        "elephant": np.array([0.4, 0.8, 0.7])
    }

def save_classes(class_embeddings):
    """Save class embeddings to file"""
    data = {k: v.tolist() for k, v in class_embeddings.items()}
    with open(CLASSES_FILE, 'w') as f:
        json.dump(data, f)

def save_to_history(image_name, prediction, confidence, feedback=None):
    """Save prediction to history"""
    history = []
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            history = json.load(f)
    
    history.append({
        'timestamp': datetime.now().isoformat(),
        'image': image_name,
        'prediction': prediction,
        'confidence': confidence,
        'feedback': feedback
    })
    
    with open(HISTORY_FILE, 'w') as f:
        json.dump(history[-100:], f)  # Keep last 100 entries

class_embeddings = load_classes()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path):
    try:
        image_tensor = preprocess_image(image_path)
        
        if not isinstance(image_tensor, torch.Tensor):
            image_tensor = torch.from_numpy(image_tensor)
        
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)
        elif image_tensor.dim() == 5:
            image_tensor = image_tensor.squeeze(0).squeeze(0).unsqueeze(0)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)
        projection_layer.to(device)
        image_tensor = image_tensor.to(device)
        
        with torch.no_grad():
            features = model(image_tensor)
            if features.dim() > 2:
                features = features.view(features.size(0), -1)
            projected_features = projection_layer(features)
            projected_features = projected_features.cpu().numpy().flatten()
        
        return projected_features
    except Exception as e:
        logger.error(f"Feature extraction error: {str(e)}")
        raise

def predict_animal(image_path):
    features = extract_features(image_path)
    
    all_classes = list(class_embeddings.keys())
    all_embeddings = np.array([class_embeddings[cls] for cls in all_classes])

    features = features / np.linalg.norm(features)
    all_embeddings = all_embeddings / np.linalg.norm(all_embeddings, axis=1, keepdims=True)

    similarity_scores = cosine_similarity([features], all_embeddings)[0]
    predicted_class = all_classes[np.argmax(similarity_scores)]
    confidence = float(similarity_scores.max())
    return predicted_class, confidence

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
@limiter.limit("10 per minute")
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "File type not allowed"}), 400

    try:
        from werkzeug.utils import secure_filename
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        predicted_class, confidence = predict_animal(file_path)
        
        save_to_history(filename, predicted_class, confidence)
        os.remove(file_path)
        
        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(confidence * 100, 2)
        })
    except Exception as e:
        logger.error(f"Upload error: {str(e)}")
        return jsonify({"error": "Processing failed"}), 500

@app.route('/api/classes', methods=['GET'])
def get_classes():
    """Get all available classes"""
    return jsonify(list(class_embeddings.keys()))

@app.route('/api/classes', methods=['POST'])
@limiter.limit("5 per minute")
def add_class():
    """Add new animal class"""
    data = request.get_json()
    if not data or 'name' not in data or 'embedding' not in data:
        return jsonify({"error": "Name and embedding required"}), 400
    
    try:
        class_embeddings[data['name']] = np.array(data['embedding'])
        save_classes(class_embeddings)
        return jsonify({"message": f"Class '{data['name']}' added successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/classes/<class_name>', methods=['DELETE'])
@limiter.limit("5 per minute")
def delete_class(class_name):
    """Delete animal class"""
    if class_name not in class_embeddings:
        return jsonify({"error": "Class not found"}), 404
    
    del class_embeddings[class_name]
    save_classes(class_embeddings)
    return jsonify({"message": f"Class '{class_name}' deleted successfully"})

@app.route('/api/feedback', methods=['POST'])
@limiter.limit("20 per minute")
def submit_feedback():
    """Submit feedback for prediction"""
    data = request.get_json()
    if not data or 'image' not in data or 'correct_class' not in data:
        return jsonify({"error": "Image and correct_class required"}), 400
    
    save_to_history(data['image'], data.get('prediction'), data.get('confidence'), data['correct_class'])
    return jsonify({"message": "Feedback submitted successfully"})

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get prediction history"""
    if os.path.exists(HISTORY_FILE):
        with open(HISTORY_FILE, 'r') as f:
            return jsonify(json.load(f))
    return jsonify([])

@app.route('/api/batch', methods=['POST'])
@limiter.limit("2 per minute")
def batch_predict():
    """Batch image processing"""
    if 'files' not in request.files:
        return jsonify({"error": "No files provided"}), 400
    
    files = request.files.getlist('files')
    if len(files) > 10:
        return jsonify({"error": "Maximum 10 files allowed"}), 400
    
    results = []
    for file in files:
        if file and allowed_file(file.filename):
            try:
                from werkzeug.utils import secure_filename
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                predicted_class, confidence = predict_animal(file_path)
                results.append({
                    "filename": filename,
                    "predicted_class": predicted_class,
                    "confidence": round(confidence * 100, 2)
                })
                
                os.remove(file_path)
            except Exception as e:
                results.append({
                    "filename": file.filename,
                    "error": str(e)
                })
    
    return jsonify({"results": results})

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "File is too large"}), 413

@app.errorhandler(429)
def ratelimit_handler(e):
    return jsonify({"error": "Rate limit exceeded"}), 429

if __name__ == '__main__':
    app.run(debug=True, port=5001)