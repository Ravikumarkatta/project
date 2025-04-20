#!/bin/bash
set -e

echo "Rolling back deployment..."
kubectl rollout undo deployment/bible-ai
kubectl rollout status deployment/bible-ai
