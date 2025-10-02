import os
from datetime import timedelta

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-key-change-in-production'
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 16 * 1024 * 1024))
    
    # Rate limiting
    RATELIMIT_STORAGE_URL = os.environ.get('REDIS_URL') or 'memory://'
    RATELIMIT_DEFAULT = os.environ.get('RATELIMIT_DEFAULT') or '100 per hour'
    
    # Model settings
    MODEL_DEVICE = os.environ.get('MODEL_DEVICE') or 'auto'  # auto, cpu, cuda
    BATCH_SIZE_LIMIT = int(os.environ.get('BATCH_SIZE_LIMIT', 10))
    
    # File paths
    CLASSES_FILE = os.environ.get('CLASSES_FILE') or 'classes.json'
    HISTORY_FILE = os.environ.get('HISTORY_FILE') or 'history.json'
    HISTORY_LIMIT = int(os.environ.get('HISTORY_LIMIT', 100))
    
    # Security
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp', 'webp'}
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL') or 'INFO'
    LOG_FILE = os.environ.get('LOG_FILE') or 'app.log'

class DevelopmentConfig(Config):
    DEBUG = True
    RATELIMIT_ENABLED = False

class ProductionConfig(Config):
    DEBUG = False
    RATELIMIT_ENABLED = True
    
class TestingConfig(Config):
    TESTING = True
    RATELIMIT_ENABLED = False
    CLASSES_FILE = 'test_classes.json'
    HISTORY_FILE = 'test_history.json'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}