#!/bin/bash
# Build the autoresearch-gaudi image and push to the container registry.
# Default: local in-cluster registry (localhost:30500). Override with REGISTRY env,
# e.g. REGISTRY=registry.example.com ./build-push-autoresearch-gaudi.sh
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

export REGISTRY="${REGISTRY:-localhost:30500}"
IMAGE_NAME="autoresearch-gaudi"
TAG="latest"
DOCKERFILE="${SCRIPT_DIR}/Dockerfile.autoresearch-gaudi"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

if [[ "${REGISTRY}" == *"localhost"* ]] || [[ "${REGISTRY}" == *"30500"* ]]; then
  echo -e "${YELLOW}Using local registry at ${REGISTRY}. Ensure it is configured as insecure (HTTP).${NC}"
fi

echo -e "${GREEN}Building ${IMAGE_NAME}:${TAG}...${NC}"
podman build \
    -f "${DOCKERFILE}" \
    -t "${REGISTRY}/${IMAGE_NAME}:${TAG}" \
    -t "${REGISTRY}/${IMAGE_NAME}:$(date +%Y%m%d)" \
    "${SCRIPT_DIR}"

echo -e "${GREEN}Build complete!${NC}"
echo -e "${YELLOW}Pushing to ${REGISTRY}...${NC}"
podman push "${REGISTRY}/${IMAGE_NAME}:${TAG}"
podman push "${REGISTRY}/${IMAGE_NAME}:$(date +%Y%m%d)"

echo -e "${GREEN}Push complete!${NC}"
echo -e "${GREEN}Image: ${REGISTRY}/${IMAGE_NAME}:${TAG}${NC}"

echo -e "${YELLOW}Verifying image in registry...${NC}"
if [[ "${REGISTRY}" == *"localhost"* ]] || [[ "${REGISTRY}" == *"30500"* ]]; then
  curl -s "http://${REGISTRY}/v2/${IMAGE_NAME}/tags/list" | jq '.' 2>/dev/null || echo "Note: curl/jq failed; image may still be pushed."
else
  curl -s "https://${REGISTRY}/v2/${IMAGE_NAME}/tags/list" | jq '.' 2>/dev/null || echo "Note: jq not installed or registry not HTTPS; image should be pushed."
fi
