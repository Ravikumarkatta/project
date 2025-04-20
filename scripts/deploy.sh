#!/bin/bash
set -e

IMAGE_NAME="ghcr.io/${GITHUB_REPOSITORY}/bible-ai:${GITHUB_SHA}"

echo "Deploying $IMAGE_NAME to $ENVIRONMENT..."

# Update Kubernetes deployment
kubectl set image deployment/bible-ai bible-ai=$IMAGE_NAME
