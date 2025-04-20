#!/bin/bash
set -e

ENVIRONMENT=$1

if [ "$ENVIRONMENT" != "production" ] && [ "$ENVIRONMENT" != "staging" ]; then
    echo "❌ Invalid environment. Must be 'production' or 'staging'"
    exit 1
fi

echo "🚀 Deploying to $ENVIRONMENT..."

# Load environment-specific configurations
if [ "$ENVIRONMENT" == "production" ]; then
    DEPLOY_URL="https://your-bible-ai.com"
else
    DEPLOY_URL="https://staging.your-bible-ai.com"
fi

# Deploy using Docker
echo "📦 Deploying Docker containers..."
docker-compose -f docker-compose.yml -f docker-compose.$ENVIRONMENT.yml up -d

# Run database migrations if needed
echo "🔄 Running database migrations..."
python scripts/manage_space.py migrate

echo "✅ Deployment to $ENVIRONMENT completed"
