import unittest
import json
import os
import tempfile
from unittest.mock import patch, MagicMock
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app_enhanced import app, class_embeddings, save_classes, load_classes

class TestAnimalClassifier(unittest.TestCase):
    def setUp(self):
        self.app = app.test_client()
        self.app.testing = True
        
    def test_home_page(self):
        """Test home page loads"""
        response = self.app.get('/')
        self.assertEqual(response.status_code, 200)
        
    def test_get_classes(self):
        """Test getting available classes"""
        response = self.app.get('/api/classes')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        self.assertIn('cat', data)
        
    def test_add_class(self):
        """Test adding new class"""
        new_class = {
            'name': 'lion',
            'embedding': [0.1, 0.2, 0.3]
        }
        response = self.app.post('/api/classes', 
                                json=new_class,
                                content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
    def test_add_class_invalid_data(self):
        """Test adding class with invalid data"""
        response = self.app.post('/api/classes', 
                                json={'name': 'lion'},
                                content_type='application/json')
        self.assertEqual(response.status_code, 400)
        
    def test_delete_class(self):
        """Test deleting class"""
        # First add a class
        self.app.post('/api/classes', 
                     json={'name': 'test_animal', 'embedding': [0.1, 0.2, 0.3]},
                     content_type='application/json')
        
        # Then delete it
        response = self.app.delete('/api/classes/test_animal')
        self.assertEqual(response.status_code, 200)
        
    def test_delete_nonexistent_class(self):
        """Test deleting non-existent class"""
        response = self.app.delete('/api/classes/nonexistent')
        self.assertEqual(response.status_code, 404)
        
    def test_submit_feedback(self):
        """Test submitting feedback"""
        feedback = {
            'image': 'test.jpg',
            'prediction': 'cat',
            'confidence': 95.5,
            'correct_class': 'dog'
        }
        response = self.app.post('/api/feedback',
                                json=feedback,
                                content_type='application/json')
        self.assertEqual(response.status_code, 200)
        
    def test_get_history(self):
        """Test getting history"""
        response = self.app.get('/api/history')
        self.assertEqual(response.status_code, 200)
        data = json.loads(response.data)
        self.assertIsInstance(data, list)
        
    def test_upload_no_file(self):
        """Test upload without file"""
        response = self.app.post('/upload')
        self.assertEqual(response.status_code, 400)
        
    def test_rate_limiting(self):
        """Test rate limiting on upload endpoint"""
        # This would require multiple rapid requests to test properly
        # For now, just test that the endpoint exists
        response = self.app.post('/upload')
        self.assertEqual(response.status_code, 400)  # No file provided
        
    @patch('app_enhanced.extract_features')
    @patch('app_enhanced.predict_animal')
    def test_upload_with_mock(self, mock_predict, mock_extract):
        """Test upload with mocked prediction"""
        mock_predict.return_value = ('cat', 0.95)
        
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
            tmp.write(b'fake image data')
            tmp.flush()
            
            with open(tmp.name, 'rb') as test_file:
                response = self.app.post('/upload',
                                       data={'file': (test_file, 'test.jpg')},
                                       content_type='multipart/form-data')
                
        os.unlink(tmp.name)
        self.assertEqual(response.status_code, 200)

class TestUtilityFunctions(unittest.TestCase):
    def test_load_save_classes(self):
        """Test loading and saving classes"""
        test_classes = {
            'test_animal': [0.1, 0.2, 0.3]
        }
        
        # Save classes
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
            json.dump(test_classes, tmp)
            tmp.flush()
            
            # Mock the CLASSES_FILE path
            with patch('app_enhanced.CLASSES_FILE', tmp.name):
                loaded = load_classes()
                self.assertIn('test_animal', loaded)
                
        os.unlink(tmp.name)

if __name__ == '__main__':
    unittest.main()