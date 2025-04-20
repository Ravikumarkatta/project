#!/bin/bash
set -e

BUILD_DIR="${1:-frontend/build}"
REQUIRED_FILES=(
    "index.html"
    "static/js/main.js"
    "static/css/main.css"
    "manifest.json"
    "favicon.ico"
)

echo "Verifying frontend build in $BUILD_DIR..."

# Check if build directory exists
if [ ! -d "$BUILD_DIR" ]; then
    echo "❌ Build directory not found!"
    exit 1
fi

# Check for required files
for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$BUILD_DIR/$file" ]; then
        echo "❌ Required file not found: $file"
        exit 1
    fi
done

# Check file sizes (warn if main bundle is too large)
JS_BUNDLE_SIZE=$(find "$BUILD_DIR" -name "main.*.js" -exec ls -l {} \; | awk '{print $5}')
if [ "$JS_BUNDLE_SIZE" -gt 2000000 ]; then
    echo "⚠️  Warning: JavaScript bundle size is larger than 2MB"
fi

# Verify sourcemaps if not production
if [ "$ENVIRONMENT" != "production" ]; then
    if ! find "$BUILD_DIR" -name "*.map" -quit; then
        echo "⚠️  Warning: No sourcemaps found in non-production build"
    fi
fi

# Basic content validation
if ! grep -q "Bible-AI" "$BUILD_DIR/index.html"; then
    echo "❌ index.html appears to be invalid (Bible-AI content not found)"
    exit 1
fi

echo "✅ Frontend build verification completed successfully!"
exit 0