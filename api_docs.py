from flask import Flask, jsonify
from flask_restx import Api, Resource, fields
from app_enhanced import app

# API Documentation with Flask-RESTX
api = Api(app, version='1.0', title='Animal Classifier API',
          description='Zero-Shot Learning Animal Classification API',
          doc='/docs/')

# Define models for documentation
upload_model = api.model('Upload', {
    'file': fields.Raw(required=True, description='Image file to classify')
})

class_model = api.model('Class', {
    'name': fields.String(required=True, description='Animal class name'),
    'embedding': fields.List(fields.Float, required=True, description='3D embedding vector')
})

feedback_model = api.model('Feedback', {
    'image': fields.String(required=True, description='Image filename'),
    'prediction': fields.String(description='Predicted class'),
    'confidence': fields.Float(description='Prediction confidence'),
    'correct_class': fields.String(required=True, description='Correct animal class')
})

prediction_response = api.model('PredictionResponse', {
    'predicted_class': fields.String(description='Predicted animal class'),
    'confidence': fields.Float(description='Confidence percentage')
})

@api.route('/upload')
class Upload(Resource):
    @api.expect(upload_model)
    @api.marshal_with(prediction_response)
    @api.doc('classify_image')
    def post(self):
        """Classify an uploaded image"""
        pass

@api.route('/api/classes')
class Classes(Resource):
    @api.doc('get_classes')
    def get(self):
        """Get all available animal classes"""
        pass
    
    @api.expect(class_model)
    @api.doc('add_class')
    def post(self):
        """Add a new animal class"""
        pass

@api.route('/api/classes/<string:class_name>')
class ClassItem(Resource):
    @api.doc('delete_class')
    def delete(self, class_name):
        """Delete an animal class"""
        pass

@api.route('/api/feedback')
class Feedback(Resource):
    @api.expect(feedback_model)
    @api.doc('submit_feedback')
    def post(self):
        """Submit feedback for a prediction"""
        pass

@api.route('/api/history')
class History(Resource):
    @api.doc('get_history')
    def get(self):
        """Get prediction history"""
        pass

@api.route('/api/batch')
class BatchProcess(Resource):
    @api.doc('batch_process')
    def post(self):
        """Process multiple images at once"""
        pass

if __name__ == '__main__':
    app.run(debug=True, port=5001)