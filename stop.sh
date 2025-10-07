#!/bin/bash

# TubeSensei Stop Script
# Cleanly stops all services

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE} TubeSensei Stop Script${NC}"
echo -e "${BLUE}========================================${NC}"

# Stop Python services
log_info "Stopping Python services..."

# Kill honcho processes
if pkill -f honcho >/dev/null 2>&1; then
    log_info "Stopped Honcho process manager"
else
    log_info "No Honcho processes found"
fi

# Kill specific service processes
services=("uvicorn.*8000" "uvicorn.*8001" "celery.*worker")

for service in "${services[@]}"; do
    if pgrep -f "$service" >/dev/null 2>&1; then
        pkill -f "$service" >/dev/null 2>&1
        log_info "Stopped $service"
    fi
done

# Wait a moment for graceful shutdown
sleep 2

# Force kill if processes are still running
for service in "${services[@]}"; do
    if pgrep -f "$service" >/dev/null 2>&1; then
        pkill -9 -f "$service" >/dev/null 2>&1
        log_warning "Force killed $service"
    fi
done

# Option to stop Docker services
echo ""
read -p "Do you want to stop Docker services (PostgreSQL, Redis, Flower, Prometheus)? [y/N]: " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    log_info "Stopping Docker services..."
    if docker-compose down; then
        log_success "Docker services stopped"
    else
        log_error "Failed to stop Docker services"
    fi
else
    log_info "Docker services left running"
    log_info "You can stop them later with: docker-compose down"
fi

log_success "TubeSensei services stopped!"

# Show status
echo ""
log_info "Current Docker services status:"
docker-compose ps