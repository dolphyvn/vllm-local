#!/bin/bash

# run.sh - Startup script for local financial assistant
# Handles vLLM and FastAPI service startup with proper configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
VLLM_PORT=8000
FASTAPI_PORT=8080
MODEL_PATH="/models/phi3"  # Adjust this path to where your model is stored
DEFAULT_MODEL="microsoft/phi-3-mini-128k-instruct"

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1

    print_status "Waiting for $service_name to be ready..."

    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" >/dev/null 2>&1; then
            print_success "$service_name is ready!"
            return 0
        fi

        echo -n "."
        sleep 2
        attempt=$((attempt + 1))
    done

    print_error "$service_name failed to start within expected time"
    return 1
}

# Function to start vLLM service
start_vllm() {
    print_status "Starting vLLM service..."

    # Check if vLLM is already running
    if check_port $VLLM_PORT; then
        print_warning "vLLM service is already running on port $VLLM_PORT"
        return 0
    fi

    # Determine model path
    if [ -d "$MODEL_PATH" ]; then
        MODEL_ARG="--model $MODEL_PATH"
        print_status "Using local model at: $MODEL_PATH"
    else
        MODEL_ARG="--model $DEFAULT_MODEL"
        print_status "Using HuggingFace model: $DEFAULT_MODEL"
        print_warning "This will download the model on first run (may take several minutes)"
    fi

    # Start vLLM in background
    python -m vllm.entrypoints.openai.api_server \
        $MODEL_ARG \
        --port $VLLM_PORT \
        --host 0.0.0.0 \
        --max-model-len 128000 \
        --trust-remote-code \
        --dtype auto \
        --api-key "token-abc123" \
        > vllm.log 2>&1 &

    VLLM_PID=$!
    echo $VLLM_PID > .vllm_pid

    print_status "vLLM started with PID: $VLLM_PID"
    print_status "Logs: vllm.log"

    # Wait for vLLM to be ready
    wait_for_service "http://localhost:$VLLM_PORT/health" "vLLM"
}

# Function to start FastAPI service
start_fastapi() {
    print_status "Starting FastAPI service..."

    # Check if FastAPI is already running
    if check_port $FASTAPI_PORT; then
        print_warning "FastAPI service is already running on port $FASTAPI_PORT"
        return 0
    fi

    # Start FastAPI in background
    uvicorn main:app --host 0.0.0.0 --port $FASTAPI_PORT --reload > fastapi.log 2>&1 &

    FASTAPI_PID=$!
    echo $FASTAPI_PID > .fastapi_pid

    print_status "FastAPI started with PID: $FASTAPI_PID"
    print_status "Logs: fastapi.log"

    # Wait for FastAPI to be ready
    wait_for_service "http://localhost:$FASTAPI_PORT/health" "FastAPI"
}

# Function to stop services
stop_services() {
    print_status "Stopping services..."

    # Stop vLLM
    if [ -f ".vllm_pid" ]; then
        VLLM_PID=$(cat .vllm_pid)
        if kill -0 $VLLM_PID 2>/dev/null; then
            print_status "Stopping vLLM (PID: $VLLM_PID)"
            kill $VLLM_PID
            rm .vllm_pid
            print_success "vLLM stopped"
        else
            print_warning "vLLM process not found"
            rm .vllm_pid
        fi
    fi

    # Stop FastAPI
    if [ -f ".fastapi_pid" ]; then
        FASTAPI_PID=$(cat .fastapi_pid)
        if kill -0 $FASTAPI_PID 2>/dev/null; then
            print_status "Stopping FastAPI (PID: $FASTAPI_PID)"
            kill $FASTAPI_PID
            rm .fastapi_pid
            print_success "FastAPI stopped"
        else
            print_warning "FastAPI process not found"
            rm .fastapi_pid
        fi
    fi
}

# Function to show status
show_status() {
    print_status "Service Status:"

    if check_port $VLLM_PORT; then
        print_success "vLLM: Running on port $VLLM_PORT"
    else
        print_warning "vLLM: Not running"
    fi

    if check_port $FASTAPI_PORT; then
        print_success "FastAPI: Running on port $FASTAPI_PORT"
    else
        print_warning "FastAPI: Not running"
    fi
}

# Function to test setup
test_setup() {
    print_status "Testing setup..."

    # Check dependencies
    print_status "Checking Python dependencies..."
    if ! python -c "import fastapi, uvicorn, chromadb, requests" 2>/dev/null; then
        print_error "Missing dependencies. Please run: pip install -r requirements.txt"
        exit 1
    fi
    print_success "Dependencies OK"

    # Check config file
    if [ ! -f "config.json" ]; then
        print_error "config.json not found"
        exit 1
    fi
    print_success "Config file found"

    # Check main files
    for file in "main.py" "memory.py"; do
        if [ ! -f "$file" ]; then
            print_error "$file not found"
            exit 1
        fi
    done
    print_success "Core files found"

    print_success "Setup test passed!"
}

# Function to run tests
run_tests() {
    print_status "Running API tests..."

    if ! check_port $FASTAPI_PORT; then
        print_error "FastAPI service is not running. Start services first."
        exit 1
    fi

    python test_client.py
}

# Main script logic
case "${1:-start}" in
    start)
        print_status "Starting Local Financial Assistant..."
        test_setup
        start_vllm
        start_fastapi
        print_success "All services started successfully!"
        echo
        print_status "API Endpoints:"
        echo "  • vLLM API: http://localhost:$VLLM_PORT"
        echo "  • FastAPI: http://localhost:$FASTAPI_PORT"
        echo "  • Health Check: http://localhost:$FASTAPI_PORT/health"
        echo "  • Chat Endpoint: http://localhost:$FASTAPI_PORT/chat"
        echo "  • Interactive Docs: http://localhost:$FASTAPI_PORT/docs"
        echo
        print_status "To test: ./run.sh test"
        print_status "To stop: ./run.sh stop"
        ;;

    stop)
        stop_services
        ;;

    restart)
        stop_services
        sleep 2
        start_vllm
        start_fastapi
        print_success "Services restarted successfully!"
        ;;

    status)
        show_status
        ;;

    test)
        run_tests
        ;;

    interactive)
        print_status "Starting interactive test mode..."
        python test_client.py interactive
        ;;

    setup)
        print_status "Running setup test..."
        test_setup
        ;;

    vllm-only)
        print_status "Starting vLLM only..."
        start_vllm
        print_success "vLLM started on port $VLLM_PORT"
        ;;

    fastapi-only)
        print_status "Starting FastAPI only..."
        start_fastapi
        print_success "FastAPI started on port $FASTAPI_PORT"
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status|test|interactive|setup|vllm-only|fastapi-only}"
        echo
        echo "Commands:"
        echo "  start        - Start all services (default)"
        echo "  stop         - Stop all services"
        echo "  restart      - Restart all services"
        echo "  status       - Show service status"
        echo "  test         - Run API tests"
        echo "  interactive  - Interactive test mode"
        echo "  setup        - Test setup requirements"
        echo "  vllm-only    - Start only vLLM service"
        echo "  fastapi-only - Start only FastAPI service"
        exit 1
        ;;
esac