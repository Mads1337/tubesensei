#!/bin/bash

# TubeSensei One-Command Startup Script
# This script starts all services with a single command

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Function to print colored output
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

log_header() {
    echo -e "${PURPLE}========================================${NC}"
    echo -e "${PURPLE} $1${NC}"
    echo -e "${PURPLE}========================================${NC}"
}

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for a service to be ready
wait_for_service() {
    local service=$1
    local host=${2:-localhost}
    local port=$3
    local max_attempts=${4:-30}
    
    log_info "Waiting for $service to be ready on $host:$port..."
    
    for ((i=1; i<=max_attempts; i++)); do
        if curl -f -s http://$host:$port/health >/dev/null 2>&1 || nc -z $host $port >/dev/null 2>&1; then
            log_success "$service is ready!"
            return 0
        fi
        
        if [ $i -eq $max_attempts ]; then
            log_error "$service failed to start after $max_attempts attempts"
            return 1
        fi
        
        echo -n "."
        sleep 2
    done
}

# Function to cleanup on exit
cleanup() {
    echo ""
    log_warning "Shutting down services..."
    
    # Kill honcho processes
    pkill -f honcho >/dev/null 2>&1 || true
    
    # Kill any remaining uvicorn/celery processes
    pkill -f "uvicorn.*8000" >/dev/null 2>&1 || true
    pkill -f "uvicorn.*8001" >/dev/null 2>&1 || true
    pkill -f "celery.*worker" >/dev/null 2>&1 || true
    
    log_info "Services stopped. Docker services will continue running."
    log_info "To stop Docker services, run: docker-compose down"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

# Main execution
log_header "TubeSensei Startup"

# Check prerequisites
log_info "Checking prerequisites..."

# Check if we're in the right directory
if [ ! -f "Procfile" ]; then
    log_error "Procfile not found. Make sure you're in the TubeSensei root directory."
    exit 1
fi

# Check if virtual environment is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    log_warning "Virtual environment not detected."
    log_info "Consider running: source venv/bin/activate"
fi

# Check if required packages are installed
if ! python -c "import honcho" >/dev/null 2>&1; then
    log_error "Honcho not installed. Installing now..."
    pip install honcho==1.1.0
fi

if ! python -c "import uvicorn" >/dev/null 2>&1; then
    log_error "Required packages not installed. Please run: pip install -r requirements.txt"
    exit 1
fi

# Check if .env file exists
if [ ! -f ".env" ]; then
    if [ -f ".env.example" ]; then
        log_warning ".env file not found. Copying from .env.example"
        cp .env.example .env
        log_warning "Please configure your .env file with proper API keys and settings."
    else
        log_error ".env file not found. Please create one based on the documentation."
        exit 1
    fi
fi

# Start Docker services
log_header "Starting Docker Services"
log_info "Starting PostgreSQL, Redis, Flower, and Prometheus..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    log_error "Docker is not running. Please start Docker and try again."
    exit 1
fi

# Start docker services
docker-compose up -d

# Wait for Docker services
log_info "Waiting for Docker services to be ready..."

# Wait for PostgreSQL
log_info "Checking PostgreSQL connection..."
for i in {1..30}; do
    if PGPASSWORD=tubesensei_dev psql -h localhost -p 5433 -U tubesensei -d tubesensei -c "SELECT 1;" >/dev/null 2>&1; then
        log_success "PostgreSQL is ready!"
        break
    fi
    if [ $i -eq 30 ]; then
        log_error "PostgreSQL failed to start"
        exit 1
    fi
    echo -n "."
    sleep 2
done

# Wait for Redis
log_info "Checking Redis connection..."
for i in {1..15}; do
    if timeout 2 bash -c "</dev/tcp/localhost/6379" >/dev/null 2>&1; then
        log_success "Redis is ready!"
        break
    fi
    if [ $i -eq 15 ]; then
        log_error "Redis failed to start"
        exit 1
    fi
    echo -n "."
    sleep 1
done

# Check port conflicts
log_info "Checking for port conflicts..."
if check_port 8000; then
    log_error "Port 8000 is already in use. Please stop the service using this port."
    exit 1
fi

if check_port 8001; then
    log_error "Port 8001 is already in use. Please stop the service using this port."
    exit 1
fi

# Run database migrations
log_header "Database Setup"
log_info "Running database migrations..."

cd tubesensei
if alembic upgrade head; then
    log_success "Database migrations completed!"
else
    log_warning "Database migrations failed or no migrations needed"
fi
cd ..

# Start all Python services with Honcho
log_header "Starting Application Services"
log_info "Starting FastAPI server, Celery workers, and Admin interface..."
log_info ""
log_info "Services will be available at:"
log_info "  • Main API:        http://localhost:8000"
log_info "  • Admin Interface: http://localhost:8001" 
log_info "  • Flower Monitor:  http://localhost:5555 (admin:admin)"
log_info "  • Prometheus:      http://localhost:9090"
log_info ""
log_info "Press Ctrl+C to stop all services"
log_info ""

# Start Honcho with colored output
exec honcho start