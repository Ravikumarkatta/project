#!/bin/bash
set -e


URL=$1
if [ -z "$URL" ]; then
    echo "Usage: $0 <deployment-url>"
    exit 1
fi

echo "Verifying deployment at $URL..."

# Wait for services to be ready
echo "Waiting for services to be ready..."
for i in {1..30}; do
    if curl -f "${URL}/health" 2>/dev/null; then
        echo "✅ Health check passed"
        
        # Additional verification
        if curl -f "${URL}/api/v1/status" 2>/dev/null | grep -q '"status":"healthy"'; then
            echo "✅ API status check passed"
            echo "Deployment verified successfully"
            exit 0
        fi
    fi
    echo "Waiting... ($i/30)"
    sleep 10
done

echo "❌ Deployment verification failed"
echo "Last known status:"
curl -v "${URL}/health" || true
exit 1
