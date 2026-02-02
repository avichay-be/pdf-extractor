.PHONY: help build run stop logs clean test dev prod health shell

# Default target
help:
	@echo "Michman PDF Extractor - Docker Commands"
	@echo ""
	@echo "Development:"
	@echo "  make dev          - Start development environment"
	@echo "  make build        - Build development Docker image"
	@echo "  make run          - Run development container"
	@echo "  make stop         - Stop development containers"
	@echo "  make logs         - View development logs"
	@echo "  make shell        - Open shell in development container"
	@echo ""
	@echo "Production:"
	@echo "  make prod         - Start production environment (with Nginx)"
	@echo "  make prod-build   - Build production Docker image"
	@echo "  make prod-stop    - Stop production containers"
	@echo "  make prod-logs    - View production logs"
	@echo ""
	@echo "Testing:"
	@echo "  make test         - Run unit tests in container"
	@echo "  make health       - Check API health"
	@echo ""
	@echo "Maintenance:"
	@echo "  make clean        - Remove containers and images"
	@echo "  make clean-all    - Remove everything including volumes"
	@echo "  make restart      - Restart development containers"
	@echo "  make rebuild      - Rebuild and restart development"

# Development commands
dev:
	@echo "Starting development environment..."
	docker-compose up -d
	@echo "API available at http://localhost:8000"
	@echo "Docs available at http://localhost:8000/docs"

build:
	@echo "Building development image..."
	docker-compose build

run: build
	docker-compose up -d

stop:
	@echo "Stopping development containers..."
	docker-compose down

logs:
	docker-compose logs -f

restart:
	@echo "Restarting development containers..."
	docker-compose restart

rebuild:
	@echo "Rebuilding and restarting..."
	docker-compose up -d --build

shell:
	@echo "Opening shell in container..."
	docker exec -it michman-pdf-extractor /bin/bash

# Production commands
prod:
	@echo "Starting production environment..."
	docker-compose -f docker-compose.prod.yml up -d
	@echo "API available at http://localhost:80"
	@echo "Direct access at http://localhost:8000"

prod-build:
	@echo "Building production image..."
	docker-compose -f docker-compose.prod.yml build

prod-stop:
	@echo "Stopping production containers..."
	docker-compose -f docker-compose.prod.yml down

prod-logs:
	docker-compose -f docker-compose.prod.yml logs -f

prod-shell:
	@echo "Opening shell in production container..."
	docker exec -it michman-pdf-extractor-prod /bin/sh

# Testing
test:
	@echo "Running tests in container..."
	docker run --rm \
		-v $(PWD):/app \
		-w /app \
		python:3.10-slim \
		/bin/bash -c "pip install -r requirements-dev.txt && pytest tests/ -v"

health:
	@echo "Checking API health..."
	@curl -s http://localhost:8000/health | python -m json.tool || \
		echo "API is not responding. Is the container running? (make dev)"

# Maintenance
clean:
	@echo "Removing containers and images..."
	docker-compose down --rmi local
	docker-compose -f docker-compose.prod.yml down --rmi local

clean-all:
	@echo "Removing everything (containers, images, volumes)..."
	docker-compose down -v --rmi all
	docker-compose -f docker-compose.prod.yml down -v --rmi all

# Utility commands
ps:
	@echo "Running containers:"
	@docker ps --filter "name=michman"

stats:
	@echo "Container statistics:"
	@docker stats --no-stream michman-pdf-extractor 2>/dev/null || \
		docker stats --no-stream michman-pdf-extractor-prod 2>/dev/null || \
		echo "No containers running"

# Setup
setup:
	@echo "Setting up environment..."
	@if [ ! -f .env ]; then \
		cp .env.example .env; \
		echo ".env file created from template"; \
		echo "Please edit .env and add your AZURE_API_KEY"; \
	else \
		echo ".env file already exists"; \
	fi

# Quick test
quick-test: dev
	@echo "Waiting for API to start..."
	@sleep 5
	@echo "Testing health endpoint..."
	@make health
