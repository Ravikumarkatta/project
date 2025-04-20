#!/bin/bash
set -e

ENVIRONMENT=$1
if [ -z "$ENVIRONMENT" ]; then
    echo "Usage: $0 <environment>"
    echo "Environment must be 'production' or 'staging'"
    exit 1
fi

if [ "$ENVIRONMENT" != "production" ] && [ "$ENVIRONMENT" != "staging" ]; then
    echo "Environment must be 'production' or 'staging'"
    exit 1
fi

# Set environment-specific variables
if [ "$ENVIRONMENT" = "production" ]; then
    NAMESPACE="bible-ai-prod"
    REPLICAS=3
    INGRESS_HOST="your-bible-ai.com"
else
    NAMESPACE="bible-ai-staging"
    REPLICAS=1
    INGRESS_HOST="staging.your-bible-ai.com"
fi

echo "Deploying to $ENVIRONMENT environment..."

# Ensure namespace exists
kubectl create namespace $NAMESPACE --dry-run=client -o yaml | kubectl apply -f -

# Apply environment-specific configs
kubectl -n $NAMESPACE create configmap bible-ai-config \
    --from-file=config/model_config.json \
    --from-file=config/data_config.json \
    --from-file=config/theological_rules.json \
    --dry-run=client -o yaml | kubectl apply -f -

# Update deployment
cat << EOF | kubectl apply -f -
apiVersion: apps/v1
kind: Deployment
metadata:
  name: bible-ai
  namespace: $NAMESPACE
spec:
  replicas: $REPLICAS
  selector:
    matchLabels:
      app: bible-ai
  template:
    metadata:
      labels:
        app: bible-ai
    spec:
      containers:
      - name: bible-ai
        image: ghcr.io/${GITHUB_REPOSITORY}/bible-ai:${GITHUB_SHA}
        env:
        - name: ENVIRONMENT
          value: "$ENVIRONMENT"
        - name: NODE_ENV
          value: "$ENVIRONMENT"
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
          limits:
            cpu: "2"
            memory: "2Gi"
        volumeMounts:
        - name: config
          mountPath: /app/config
      volumes:
      - name: config
        configMap:
          name: bible-ai-config
EOF

# Update service
cat << EOF | kubectl apply -f -
apiVersion: v1
kind: Service
metadata:
  name: bible-ai
  namespace: $NAMESPACE
spec:
  selector:
    app: bible-ai
  ports:
  - port: 80
    targetPort: 8000
EOF

# Update ingress
cat << EOF | kubectl apply -f -
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: bible-ai
  namespace: $NAMESPACE
  annotations:
    kubernetes.io/ingress.class: nginx
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
  - hosts:
    - $INGRESS_HOST
    secretName: bible-ai-tls
  rules:
  - host: $INGRESS_HOST
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: bible-ai
            port:
              number: 80
EOF

echo "Waiting for deployment to roll out..."
kubectl -n $NAMESPACE rollout status deployment/bible-ai --timeout=300s

echo "Deployment to $ENVIRONMENT completed successfully!"
