#!/usr/bin/env bash
# ============================================
# LLM Router - Container Compose (apple/container)
# ============================================
# Ersatz für docker-compose mit apple/container
# ============================================
# Usage:
#   ./scripts/container-compose.sh up       # Start alle Services
#   ./scripts/container-compose.sh down     # Stop alle Services
#   ./scripts/container-compose.sh logs     # Logs aller Services
#   ./scripts/container-compose.sh status   # Status aller Services
# ============================================

set -euo pipefail

# Configuration
IMAGE_NAME="${LLM_ROUTER_IMAGE:-llm-router}"
IMAGE_TAG="${LLM_ROUTER_TAG:-latest}"
LLM_ROUTER_NAME="llm-router"
JAEGER_NAME="jaeger"
OTEL_COLLECTOR_NAME="otel-collector"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${RED} $1"
}

# Check if container system is running
check_system() {
    if ! container system status &>/dev/null; then
        log_error "Container system is not running!"
        echo ""
        echo "Start it with:"
        echo "  sudo container system start"
        exit 1
    fi
}

# Check if machine exists
check_machine() {
    if ! container machine list &>/dev/null; then
        log_error "No container machine available!"
        echo ""
        echo "Create one with:"
        echo "  container machine create --arch arm64 llm-mac"
        echo "  container machine set-default llm-mac"
        exit 1
    fi
}

# Check prerequisites
check_prerequisites() {
    check_system
    check_machine
}

# Start all services
cmd_up() {
    log_info "Starting LLM Router services..."
    echo ""

    # Check if image exists
    if ! container image inspect "$IMAGE_NAME:$IMAGE_TAG" &>/dev/null; then
        log_error "Image $IMAGE_NAME:$IMAGE_TAG not found!"
        echo ""
        echo "Build it first with:"
        echo "  make build"
        exit 1
    fi

    # Start LLM Router
    log_info "Starting LLM Router..."
    container run -d --name "$LLM_ROUTER_NAME" \
        -p 8000:8000 \
        --volume "$(pwd)/profiles:/app/profiles:ro" \
        --volume "$(pwd)/agent-policies:/app/agent-policies:ro" \
        --volume "$(pwd)/logs:/app/logs:rw" \
        "$IMAGE_NAME:$IMAGE_TAG"

    log_info "✅ LLM Router started on port 8000"
    echo ""

    # Start Jaeger (optional, for tracing)
    if [[ "${ENABLE_JAEGER:-false}" == "true" ]]; then
        log_info "Starting Jaeger for tracing..."
        container run -d --name "$JAEGER_NAME" \
            -p 16686:16686 \
            -e COLLECTOR_OTLP_ENABLED=true \
            jaegertracing/all-in-one:latest
        log_info "✅ Jaeger started (http://localhost:16686)"
    fi

    echo ""
    log_info "Services started:"
    echo "  - LLM Router: http://localhost:8000"
    container list 2>/dev/null || true
}

# Stop all services
cmd_down() {
    log_info "Stopping all services..."
    echo ""

    # Stop LLM Router
    if container list --format '{{.Names}}' 2>/dev/null | grep -q "^${LLM_ROUTER_NAME}$"; then
        log_info "Stopping LLM Router..."
        container stop "$LLM_ROUTER_NAME" 2>/dev/null || true
    fi

    # Stop Jaeger
    if [[ "${ENABLE_JAEGER:-false}" == "true" ]]; then
        if container list --format '{{.Names}}' 2>/dev/null | grep -q "^${JAEGER_NAME}$"; then
            log_info "Stopping Jaeger..."
            container stop "$JAEGER_NAME" 2>/dev/null || true
        fi
    fi

    echo ""
    log_info "✅ All services stopped"
}

# Show logs
cmd_logs() {
    log_info "Logs for LLM Router:"
    container logs -f "$LLM_ROUTER_NAME" 2>&1 | tail -50 || true

    if [[ "${ENABLE_JAEGER:-false}" == "true" ]]; then
        echo ""
        log_info "Logs for Jaeger:"
        container logs -f "$JAEGER_NAME" 2>&1 | tail -20 || true
    fi
}

# Show status
cmd_status() {
    log_info "Service Status:"
    echo ""
    container list 2>/dev/null || true
    echo ""
    log_info "System Status:"
    container system status 2>&1 || true
}

# Show help
cmd_help() {
    echo "LLM Router - Container Compose (apple/container)"
    echo ""
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  up       Start all services"
    echo "  down     Stop all services"
    echo "  logs     Show service logs"
    echo "  status   Show service status"
    echo "  help     Show this help message"
    echo ""
    echo "Environment Variables:"
    echo "  LLM_ROUTER_IMAGE   Image name (default: llm-router)"
    echo "  LLM_ROUTER_TAG     Image tag (default: latest)"
    echo "  ENABLE_JAEGER      Enable Jaeger tracing (default: false)"
}

# Main command dispatcher
case "${1:-help}" in
    up)
        check_prerequisites
        cmd_up
        ;;
    down)
        cmd_down
        ;;
    logs)
        check_prerequisites
        cmd_logs
        ;;
    status)
        check_prerequisites
        cmd_status
        ;;
    help|--help|-h)
        cmd_help
        ;;
    *)
        log_error "Unknown command: $1"
        cmd_help
        exit 1
        ;;
esac
