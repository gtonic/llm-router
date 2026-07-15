#!/usr/bin/env bash
# ============================================
# Build LLM Router image with apple/container
# ============================================
# Usage: ./scripts/container-build.sh [TAG]
#
# Prerequisites:
#   1. Start container system: container system start
#   2. Create a machine:       container machine create --arch arm64 llm-mac
#   3. Set default machine:    container machine set-default llm-mac
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="${LLM_ROUTER_IMAGE:-llm-router}"
IMAGE_TAG="${1:-latest}"

echo "🔨 Building $IMAGE_NAME:$IMAGE_TAG with apple/container..."
cd "$PROJECT_DIR"

# Build the image
container build \
    -t "$IMAGE_NAME:$IMAGE_TAG" \
    --progress plain \
    .

echo "✅ Image built successfully: $IMAGE_NAME:$IMAGE_TAG"
echo ""
echo "Run with:"
echo "  container run -d --name llm-router -p 8000:8000 $IMAGE_NAME:$IMAGE_TAG"
