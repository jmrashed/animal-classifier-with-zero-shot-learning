#!/bin/bash

# Animal Classifier Deployment Script

set -e

echo "🚀 Starting deployment..."

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    exit 1
fi

# Check if Docker Compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

# Create environment file if it doesn't exist
if [ ! -f .env ]; then
    echo "📝 Creating .env file..."
    cat > .env << EOF
SECRET_KEY=$(openssl rand -hex 32)
FLASK_ENV=production
MODEL_DEVICE=auto
RATELIMIT_DEFAULT=100 per hour
LOG_LEVEL=INFO
EOF
    echo "✅ .env file created"
fi

# Create necessary directories
echo "📁 Creating directories..."
mkdir -p uploads logs

# Build and start services
echo "🔨 Building Docker images..."
docker-compose build

echo "🚀 Starting services..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 10

# Check if services are running
if docker-compose ps | grep -q "Up"; then
    echo "✅ Services are running!"
    echo "🌐 Application is available at: http://localhost"
    echo "📚 API documentation: http://localhost/docs/"
else
    echo "❌ Services failed to start. Check logs:"
    docker-compose logs
    exit 1
fi

echo "🎉 Deployment completed successfully!"