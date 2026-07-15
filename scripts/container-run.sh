#!/usr/bin/env bash
# ============================================
# Run LLM Router container with apple/container
# ============================================
# Usage: ./scripts/container-run.sh [OPTIONS]
#
# Options:
#   --env-file FILE   Load environment variables from file
#   --profile FILE    Mount profile directory (repeatable)
#   --port PORT       Host port to expose (default: 8000)
#   --detach          Run in detached mode
#   --rm              Remove container on exit
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
IMAGE_NAME="${LLM_ROUTER_IMAGE:-llm-router}"
IMAGE_TAG="${LLM_ROUTER_TAG:-latest}"
CONTAINER_NAME="llm-router"
HOST_PORT="${LLM_ROUTER_PORT:-8000}"

# Parse arguments
DETACH=false
REMOVE=false
ENV_FILE=""
VOLUMES=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --env-file)
            ENV_FILE="$2"
            shift 2
            ;;
        --profile)
            VOLUMES+=("$2")
            shift 2
            ;;
        --port)
            HOST_PORT="$2"
            shift 2
            ;;
        --detach)
            DETACH=true
            shift
            ;;
        --rm)
            REMOVE=true
            shift
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --env-file FILE   Load environment variables from file"
            echo "  --profile DIR     Mount profile directory (repeatable)"
            echo "  --port PORT       Host port to expose (default: 8000)"
            echo "  --detach          Run in detached mode"
            echo "  --rm              Remove container on exit"
            echo "  --help, -h        Show this help"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# Build run command
RUN_CMD=(
    container run
    --name "$CONTAINER_NAME"
    -p "${HOST_PORT}:8000"
)

# Add detach flag
if [[ "$DETACH" == "true" ]]; then
    RUN_CMD+=(-d)
fi

# Add rm flag
if [[ "$REMOVE" == "true" ]]; then
    RUN_CMD+=(--rm)
fi

# Add environment variables
if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then
    RUN_CMD+=(--env-file "$ENV_FILE")
else
    # Use .env file from project root if exists
    if [[ -f "$PROJECT_DIR/.env" ]]; then
        RUN_CMD+=(--env-file "$PROJECT_DIR/.env")
    fi
fi

# Add environment variables from .env file if provided
if [[ -n "$ENV_FILE" && -f "$ENV_FILE" ]]; then
    # Source env vars for this script's use
    set -a
    source "$ENV_FILE"
    set +a
fi

# Add volumes for profiles
for vol in "${VOLUMES[@]}"; do
    if [[ -d "$vol" ]]; then
        RUN_CMD+=(--volume "$(realpath "$vol"):/app/profiles:ro")
    fi
done

# Add default volumes if no profiles mounted
if [[ ${#VOLUMES[@]} -eq 0 ]]; then
    if [[ -d "$PROJECT_DIR/profiles" ]]; then
        RUN_CMD+=(--volume "$(realpath "$PROJECT_DIR/profiles"):/app/profiles:ro")
    fi
    if [[ -d "$PROJECT_DIR/agent-policies" ]]; then
        RUN_CMD+=(--volume "$(realpath "$PROJECT_DIR/agent-policies"):/app/agent-policies:ro")
    fi
fi

# Add logging volume
RUN_CMD+=(--volume "$(realpath "$PROJECT_DIR/logs"):/app/logs:rw" 2>/dev/null || true)

# Add image
RUN_CMD+=("$IMAGE_NAME:$IMAGE_TAG")

echo "🚀 Starting $IMAGE_NAME:$IMAGE_TAG"
echo "   Container: $CONTAINER_NAME"
echo "   Port: $HOST_PORT -> 8000"
echo "   Command: ${RUN_CMD[*]}"
echo ""

# Run the container
"${RUN_CMD[@]}"

if [[ "$DETACH" == "true" ]]; then
    echo ""
    echo "✅ Container started in detached mode"
    echo ""
    echo "View logs:   container logs -f $CONTAINER_NAME"
    echo "Stop:        container stop $CONTAINER_NAME"
    echo "Remove:      container delete $CONTAINER_NAME"
else
    echo ""
    echo "✅ Container running (press Ctrl+C to stop)"
fi
