#!/bin/bash
# Verify frontend build script

set -e  # Exit immediately if a command exits with a non-zero status

# Color codes for better readability
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}Verifying frontend build...${NC}"

# Check if we're in the frontend directory or parent directory
if [ -d "build" ]; then
  BUILD_DIR="build"
elif [ -d "frontend/build" ]; then
  BUILD_DIR="frontend/build"
else
  echo -e "${RED}❌ Build directory not found${NC}"
  exit 1
fi

# Check if index.html exists
if [ ! -f "$BUILD_DIR/index.html" ]; then
  echo -e "${RED}❌ index.html not found in build directory${NC}"
  exit 1
else
  echo -e "${GREEN}✓ index.html found${NC}"
fi

# Check for JavaScript files
JS_COUNT=$(find "$BUILD_DIR" -name "*.js" | wc -l)
if [ "$JS_COUNT" -eq 0 ]; then
  echo -e "${RED}❌ No JavaScript files found in build directory${NC}"
  exit 1
else
  echo -e "${GREEN}✓ Found $JS_COUNT JavaScript files${NC}"
fi

# Check for CSS files
CSS_COUNT=$(find "$BUILD_DIR" -name "*.css" | wc -l)
if [ "$CSS_COUNT" -eq 0 ]; then
  echo -e "${RED}❌ No CSS files found in build directory${NC}"
  exit 1
else
  echo -e "${GREEN}✓ Found $CSS_COUNT CSS files${NC}"
fi

# Check for asset files
ASSET_COUNT=$(find "$BUILD_DIR" -type f -not -name "*.html" -not -name "*.js" -not -name "*.css" -not -name "*.json" | wc -l)
echo -e "${GREEN}✓ Found $ASSET_COUNT additional asset files${NC}"

# Verify HTML contains expected content
if grep -q "Bible-AI" "$BUILD_DIR/index.html"; then
  echo -e "${GREEN}✓ index.html contains expected content${NC}"
else
  echo -e "${YELLOW}⚠️ index.html does not contain 'Bible-AI' - this might be intentional${NC}"
fi

# Check file sizes
echo -e "${YELLOW}Checking build size...${NC}"
TOTAL_SIZE=$(du -sh "$BUILD_DIR" | cut -f1)
echo -e "${GREEN}✓ Total build size: $TOTAL_SIZE${NC}"

# Final verdict
echo -e "${GREEN}==============================${NC}"
echo -e "${GREEN}✅ Frontend build verification completed successfully${NC}"
echo -e "${GREEN}==============================${NC}"

exit 0
