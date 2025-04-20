#!/bin/bash
set -e

echo "Verifying deployment..."

# Wait for rollout to complete
kubectl rollout status deployment/bible-ai --timeout=300s

# Check application health
for i in {1..6}; do
  if curl -f http://bible-ai/health; then
    echo "Deployment verified successfully"
    exit 0
  fi
  sleep 10
done

echo "Deployment verification failed"
exit 1
