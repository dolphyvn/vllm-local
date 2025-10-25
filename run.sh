#!/bin/bash

# run.sh - Startup script for local financial assistant
# Handles Ollama and FastAPI service startup with proper configuration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
OLLAMA_PORT=11434
FASTAPI_PORT=8080
DEFAULT_MODEL="llama3.2"

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

# Function to check if Ollama is installed
check_ollama() {
    if command -v ollama >/dev/null 2>&1; then
        return 0  # Ollama is installed
    else
        return 1  # Ollama is not installed
    fi
}

# Function to check if Ollama service is running
check_ollama_service() {
    if check_port $OLLAMA_PORT; then
        return 0  # Ollama is running
    else
        return 1  # Ollama is not running
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

# Function to start Ollama service
start_ollama() {
    print_status "Starting Ollama service..."

    # Check if Ollama is installed
    if ! check_ollama; then
        print_error "Ollama is not installed. Please install Ollama first:"
        echo "   macOS: brew install ollama"
        echo "   Linux: curl -fsSL https://ollama.com/install.sh | sh"
        echo "   Windows: Download from https://ollama.com/download"
        echo ""
        echo "After installation, run this script again."
        return 1
    fi

    # Check if Ollama is already running
    if check_ollama_service; then
        print_success "Ollama service is already running on port $OLLAMA_PORT"

        # Check if the required model is available
        if ollama list | grep -q "$DEFAULT_MODEL"; then
            print_success "Model '$DEFAULT_MODEL' is available"
        else
            print_warning "Model '$DEFAULT_MODEL' not found. Attempting to pull..."
            print_status "This may take several minutes depending on your internet connection and model size."
            ollama pull "$DEFAULT_MODEL" || {
                print_error "Failed to pull model '$DEFAULT_MODEL'"
                print_status "Please run 'ollama pull $DEFAULT_MODEL' manually"
                return 1
            }
            print_success "Model '$DEFAULT_MODEL' pulled successfully"
        fi
        return 0
    fi

    # Start Ollama service
    print_status "Starting Ollama service..."
    ollama serve > ollama.log 2>&1 &
    OLLAMA_PID=$!
    echo $OLLAM_PID > .ollama_pid

    print_status "Ollama started with PID: $OLLAMA_PID"
    print_status "Logs: ollama.log"

    # Wait for Ollama to be ready
    wait_for_service "http://localhost:$OLLAMA_PORT/api/tags" "Ollama"

    # Check and pull required model
    if ollama list | grep -q "$DEFAULT_MODEL"; then
        print_success "Model '$DEFAULT_MODEL' is available"
    else
        print_warning "Model '$DEFAULT_MODEL' not found. Pulling now..."
        print_status "This may take several minutes..."
        ollama pull "$DEFAULT_MODEL" || {
            print_error "Failed to pull model '$DEFAULT_MODEL'"
            print_status "Please run 'ollama pull $DEFAULT_MODEL' manually"
            return 1
        }
        print_success "Model '$DEFAULT_MODEL' pulled successfully"
    fi
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

    # Stop Ollama (if started by this script)
    if [ -f ".ollama_pid" ]; then
        OLLAMA_PID=$(cat .ollama_pid)
        if kill -0 $OLLAMA_PID 2>/dev/null; then
            print_status "Stopping Ollama (PID: $OLLAMA_PID)"
            kill $OLLAMA_PID
            rm .ollama_pid
            print_success "Ollama stopped"
        else
            print_warning "Ollama process not found"
            rm .ollama_pid
        fi
    else
        print_status "Ollama was not started by this script - leaving it running"
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

    if check_ollama_service; then
        print_success "Ollama: Running on port $OLLAMA_PORT"
        # Show available models
        if check_ollama; then
            MODELS=$(ollama list 2>/dev/null | tail -n +2 | awk '{print $1}' | tr '\n' ', ' | sed 's/,$//')
            if [ -n "$MODELS" ]; then
                print_status "Available models: $MODELS"
            fi
        fi
    else
        print_warning "Ollama: Not running"
        if ! check_ollama; then
            print_status "Ollama is not installed"
        fi
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

    # Check Ollama installation
    print_status "Checking Ollama installation..."
    if check_ollama; then
        print_success "Ollama is installed"
    else
        print_error "Ollama is not installed. Please install Ollama first:"
        echo "   macOS: brew install ollama"
        echo "   Linux: curl -fsSL https://ollama.com/install.sh | sh"
        echo "   Windows: Download from https://ollama.com/download"
        exit 1
    fi

    # Check Python dependencies
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
        start_ollama || exit 1
        start_fastapi
        print_success "All services started successfully!"
        echo
        print_status "API Endpoints:"
        echo "  • Ollama API: http://localhost:$OLLAMA_PORT"
        echo "  • FastAPI: http://localhost:$FASTAPI_PORT"
        echo "  • Health Check: http://localhost:$FASTAPI_PORT/health"
        echo "  • Models List: http://localhost:$FASTAPI_PORT/models"
        echo "  • Chat Endpoint: http://localhost:$FASTAPI_PORT/chat"
        echo "  • Web UI: http://localhost:$FASTAPI_PORT"
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
        start_ollama || exit 1
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

    ollama-only)
        print_status "Starting Ollama only..."
        start_ollama || exit 1
        print_success "Ollama started on port $OLLAMA_PORT"
        ;;

    fastapi-only)
        print_status "Starting FastAPI only..."
        start_fastapi
        print_success "FastAPI started on port $FASTAPI_PORT"
        ;;

    *)
        echo "Usage: $0 {start|stop|restart|status|test|interactive|setup|ollama-only|fastapi-only}"
        echo
        echo "Commands:"
        echo "  start        - Start all services (default)"
        echo "  stop         - Stop all services"
        echo "  restart      - Restart all services"
        echo "  status       - Show service status"
        echo "  test         - Run API tests"
        echo "  interactive  - Interactive test mode"
        echo "  setup        - Test setup requirements"
        echo "  ollama-only  - Start only Ollama service"
        echo "  fastapi-only - Start only FastAPI service"
        echo
        echo "Examples:"
        echo "  ./run.sh start              # Start Ollama and FastAPI"
        echo "  ./run.sh ollama-only        # Start only Ollama"
        echo "  ./run.sh status             # Check service status"
        echo "  ollama pull llama3.2        # Pull a specific model"
        echo "  ollama list                 # List available models"
        exit 1
        ;;
esac