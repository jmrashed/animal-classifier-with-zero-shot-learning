# Animal Classifier with Zero-Shot Learning

![Animal Classifier Demo](./static/video/ezgif-2-002aecc9fd.gif)

A web-based animal classification system using Zero-Shot Learning and ResNet, built with Flask and PyTorch. This application can classify animals in images without being explicitly trained on them, using semantic embeddings to make predictions.

## Features

- 🖼️ Real-time image classification
- 🔄 Zero-Shot Learning capabilities
- 🎯 High accuracy with ResNet backbone
- 🌐 Web-based interface with drag-and-drop support
- 📊 Confidence scores for predictions
- 🚀 Fast inference time
- 📱 Responsive design

## Live Demo

Try it out: [Animal Classifier Demo](https://animal-classifier-with-zero-shot-learning.herokuapp.com)

## Technology Stack

- **Backend**: Python, Flask
- **Deep Learning**: PyTorch, ResNet
- **Frontend**: HTML5, CSS3, JavaScript
- **Image Processing**: PIL, OpenCV
- **Vector Similarity**: scikit-learn

## Installation

1. Clone the repository:
```bash
git clone https://github.com/jmrashed/Animal-classifier-with-zero-shot-learning.git
cd Animal-classifier-with-zero-shot-learning
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
# or
venv\Scripts\activate  # For Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Run the application:
```bash
python app.py
```

The application will be available at `http://localhost:5001`

## Project Structure

```
Animal-classifier-with-zero-shot-learning/
│
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
│
├── models/
│   ├── __init__.py
│   └── resnet_model.py   # ResNet model configuration
│
├── utils/
│   ├── __init__.py
│   └── preprocess.py     # Image preprocessing utilities
│
├── static/
│   ├── css/
│   └── js/
│
├── templates/
│   └── index.html        # Web interface
│
├── uploads/              # Temporary image storage
└── README.md
```

## API Endpoints

### POST /upload
Upload an image for classification.

**Request**:
- Method: POST
- Content-Type: multipart/form-data
- Body: file (image file)

**Response**:
```json
{
    "predicted_class": "cat",
    "confidence": 95.5
}
```

## Zero-Shot Learning

This classifier uses Zero-Shot Learning to predict animal classes without being explicitly trained on them. It works by:

1. Extracting visual features using ResNet
2. Projecting features to a semantic embedding space
3. Comparing embeddings with predefined class embeddings
4. Making predictions based on similarity scores

Currently supported animal classes:
- Cat
- Dog
- Horse
- Zebra
- Elephant

## Development

To add new animal classes:

1. Add class embeddings to `class_embeddings` dictionary in `app.py`:
```python
class_embeddings = {
    "new_animal": np.array([0.x, 0.y, 0.z])
}
```

2. Update the projection layer if needed:
```python
class ProjectionLayer(nn.Module):
    def __init__(self, input_dim=2048, output_dim=512):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.ReLU(),
            nn.Linear(output_dim, embedding_dim)
        )
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the MIT License. See `LICENSE` for more information.

## Contact

Md Rasheduzzaman - [@jmrashed](https://twitter.com/jmrashed) - jmrashed@gmail.com

Project Link: [https://github.com/jmrashed/Animal-classifier-with-zero-shot-learning](https://github.com/jmrashed/Animal-classifier-with-zero-shot-learning)

## Acknowledgments

- [PyTorch](https://pytorch.org)
- [Flask](https://flask.palletsprojects.com)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [Zero-Shot Learning Review](https://arxiv.org/abs/1907.11978)