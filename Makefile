# ============================================
# LLM Router - Apple Container Makefile
# ============================================
# Usage:
#   make container-start   # Start container system
#   make machine-create    # Create VM machine
#   make build             # Build container image
#   make run               # Run container
#   make shell             # Run shell in container
#   make logs              # View container logs
#   make stop              # Stop container
#   make clean             # Remove container and image
# ============================================

IMAGE_NAME ?= llm-router
IMAGE_TAG  ?= latest
CONTAINER_NAME ?= llm-router
PORT       ?= 8000
ENV_FILE   ?= .env

.PHONY: help container-start machine-create machine-delete build run stop shell logs clean ps

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

container-start: ## Start container system (requires sudo)
	@echo "Starting container system..."
	sudo container system start
	@echo "✅ Container system started"

machine-create: ## Create a new VM machine (default: llm-mac)
	@echo "Creating machine..."
	container machine create --arch arm64 llm-mac
	@echo "✅ Machine created: llm-mac"
	@echo ""
	@echo "Set as default:"
	@echo "  container machine set-default llm-mac"

machine-delete: ## Delete the VM machine
	@echo "Deleting machine llm-mac..."
	container machine delete llm-mac 2>/dev/null || true
	@echo "✅ Machine deleted"

build: ## Build container image
	@echo "Building $(IMAGE_NAME):$(IMAGE_TAG)..."
	./scripts/container-build.sh $(IMAGE_TAG)

run: ## Run container (requires: container system start + machine)
	@echo "Running $(IMAGE_NAME):$(IMAGE_TAG) on port $(PORT)..."
	./scripts/container-run.sh --port $(PORT) --detach

shell: ## Open shell in running container
	@echo "Opening shell in $(CONTAINER_NAME)..."
	container run -it --rm --entrypoint /bin/sh $(IMAGE_NAME):$(IMAGE_TAG)

logs: ## View container logs
	@echo "Logs for $(CONTAINER_NAME):"
	container logs -f $(CONTAINER_NAME) 2>&1 | tail -50

stop: ## Stop running container
	@echo "Stopping $(CONTAINER_NAME)..."
	-container stop $(CONTAINER_NAME) 2>/dev/null || true
	@echo "✅ Container stopped"

clean: stop ## Remove container, image, and machine
	@echo "Cleaning up..."
	-container rm $(CONTAINER_NAME) 2>/dev/null || true
	-container rmi $(IMAGE_NAME):$(IMAGE_TAG) 2>/dev/null || true
	@echo "✅ Cleanup complete"

ps: ## List running containers
	@echo "Running containers:"
	container list

check-system: ## Check container system status
	@echo "Checking container system status..."
	container system status 2>&1 || true
	@echo ""
	@echo "Available machines:"
	container machine list 2>&1 || true
	@echo ""
	@echo "Running containers:"
	container list 2>&1 || true
