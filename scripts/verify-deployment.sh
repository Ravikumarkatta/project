#!/bin/bash
set -e

DEPLOY_URL=$1

if [ -z "$DEPLOY_URL" ]; then
    echo "❌ No deployment URL provided"
    exit 1
fi

echo "🔍 Verifying deployment at $DEPLOY_URL..."

# Wait for service to be available
MAX_RETRIES=30
RETRY_COUNT=0

while true; do
    if curl -s -f "${DEPLOY_URL}/health" > /dev/null; then
        echo "✅ Service is responding"
        break
    fi

    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -ge $MAX_RETRIES ]; then
        echo "❌ Service failed to respond after $MAX_RETRIES attempts"
        exit 1
    fi

    echo "⏳ Waiting for service to respond (attempt $RETRY_COUNT/$MAX_RETRIES)..."
    sleep 10
done

# Verify API endpoints
echo "🔍 Checking API endpoints..."
curl -s -f "${DEPLOY_URL}/api/v1/bible/versions" > /dev/null || { echo "❌ Bible versions endpoint failed"; exit 1; }

# Verify frontend
echo "🔍 Checking frontend..."
curl -s -f "${DEPLOY_URL}" | grep -q "Bible-AI" || { echo "❌ Frontend verification failed"; exit 1; }

echo "✅ Deployment verification successful"
