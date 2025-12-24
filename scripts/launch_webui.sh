#!/usr/bin/env bash
# HighNoon Language Framework - WebUI Launch Script
# Copyright 2025 Verso Industries
#
# This script launches the HighNoon WebUI (React Dashboard + FastAPI Backend).
# Usage: ./scripts/launch_webui.sh [--host HOST] [--port PORT] [--api-port API_PORT]

set -e

# Default configuration
HOST="127.0.0.1"
PORT="5173"
API_PORT="8000"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
FRONTEND_DIR="$PROJECT_ROOT/highnoon/webui/frontend"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --host)
            HOST="$2"
            shift 2
            ;;
        --port)
            PORT="$2"
            shift 2
            ;;
        --api-port)
            API_PORT="$2"
            shift 2
            ;;
        --help|-h)
            echo "HighNoon Language Framework - WebUI Launcher"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --host HOST         Host address to bind (default: 127.0.0.1)"
            echo "  --port PORT         React frontend port (default: 5173)"
            echo "  --api-port PORT     FastAPI backend port (default: 8000)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                             # Run on localhost:5173 (UI) + :8000 (API)"
            echo "  $0 --host 0.0.0.0              # Expose to network"
            echo "  $0 --port 3000 --api-port 8080 # Custom ports"
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║         HighNoon Language Framework - React Dashboard        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Activate virtual environment if it exists
VENV_DIR="$PROJECT_ROOT/venv"
if [ -d "$VENV_DIR" ]; then
    echo -e "${GREEN}✓ Activating virtual environment...${NC}"
    source "$VENV_DIR/bin/activate"
else
    echo -e "${YELLOW}! Virtual environment not found. Using system Python.${NC}"
    echo -e "${YELLOW}  Run ./scripts/setup.sh first to create a venv.${NC}"
fi

# Check if uvicorn is available
if ! command -v uvicorn &> /dev/null && ! python -c "import uvicorn" 2>/dev/null; then
    echo -e "${RED}ERROR: uvicorn not found. Please run ./scripts/setup.sh first.${NC}"
    exit 1
fi

# Check if node_modules exists in frontend
if [ ! -d "$FRONTEND_DIR/node_modules" ]; then
    echo -e "${YELLOW}! Node modules not found. Installing dependencies...${NC}"
    cd "$FRONTEND_DIR"
    npm install
    cd "$PROJECT_ROOT"
fi

# Cleanup function to kill background processes on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down services...${NC}"
    if [ ! -z "$API_PID" ]; then
        kill $API_PID 2>/dev/null || true
    fi
    if [ ! -z "$FRONTEND_PID" ]; then
        kill $FRONTEND_PID 2>/dev/null || true
    fi
    echo -e "${GREEN}✓ Services stopped.${NC}"
    exit 0
}
trap cleanup SIGINT SIGTERM

# Start FastAPI backend in background
echo ""
echo -e "${CYAN}Starting FastAPI Backend...${NC}"
echo -e "  API Host: ${GREEN}$HOST${NC}"
echo -e "  API Port: ${GREEN}$API_PORT${NC}"
python -m uvicorn highnoon.webui.app:create_app --factory --host "$HOST" --port "$API_PORT" &
API_PID=$!
sleep 2

# Check if API started successfully
if ! kill -0 $API_PID 2>/dev/null; then
    echo -e "${RED}ERROR: Failed to start FastAPI backend.${NC}"
    exit 1
fi
echo -e "${GREEN}✓ FastAPI backend started (PID: $API_PID)${NC}"

# Start React frontend
echo ""
echo -e "${CYAN}Starting React Dashboard...${NC}"
echo -e "  UI Host: ${GREEN}$HOST${NC}"
echo -e "  UI Port: ${GREEN}$PORT${NC}"
echo ""
echo -e "${BLUE}────────────────────────────────────────────────────────────────${NC}"
echo -e "  Access the WebUI at: ${GREEN}http://$HOST:$PORT${NC}"
echo -e "  API available at:    ${GREEN}http://$HOST:$API_PORT/api${NC}"
echo -e "${BLUE}────────────────────────────────────────────────────────────────${NC}"
echo ""
echo -e "Press ${YELLOW}Ctrl+C${NC} to stop all services."
echo ""

cd "$FRONTEND_DIR"
npm run dev -- --host "$HOST" --port "$PORT"
