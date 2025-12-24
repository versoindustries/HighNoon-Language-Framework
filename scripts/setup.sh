#!/usr/bin/env bash
# HighNoon Language Framework - Setup Script
# Copyright 2025 Verso Industries
#
# This script sets up the development environment for the HighNoon Language Framework.
# Usage: ./scripts/setup.sh

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}╔══════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║       HighNoon Language Framework - Environment Setup        ║${NC}"
echo -e "${BLUE}╚══════════════════════════════════════════════════════════════╝${NC}"
echo ""

# Check Python version
check_python_version() {
    echo -e "${YELLOW}[1/4]${NC} Checking Python version..."

    # Prefer Python 3.10-3.12 for TensorFlow compatibility
    # TensorFlow doesn't support Python 3.13+ yet
    if command -v python3.12 &> /dev/null; then
        PYTHON_CMD="python3.12"
    elif command -v python3.11 &> /dev/null; then
        PYTHON_CMD="python3.11"
    elif command -v python3.10 &> /dev/null; then
        PYTHON_CMD="python3.10"
    elif command -v python3 &> /dev/null; then
        PYTHON_CMD="python3"
    elif command -v python &> /dev/null; then
        PYTHON_CMD="python"
    else
        echo -e "${RED}ERROR: Python not found. Please install Python 3.10-3.12.${NC}"
        exit 1
    fi

    # Get Python version
    PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    MAJOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.major)')
    MINOR=$($PYTHON_CMD -c 'import sys; print(sys.version_info.minor)')

    # Check Python version bounds (3.10 <= version <= 3.12 for TensorFlow)
    if [ "$MAJOR" -lt 3 ] || ([ "$MAJOR" -eq 3 ] && [ "$MINOR" -lt 10 ]); then
        echo -e "${RED}ERROR: Python 3.10+ required. Found Python $PYTHON_VERSION${NC}"
        exit 1
    fi

    if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -gt 12 ]; then
        echo -e "${YELLOW}  ! Warning: Python $PYTHON_VERSION detected. TensorFlow may not be compatible.${NC}"
        echo -e "${YELLOW}  ! Recommended: Python 3.10, 3.11, or 3.12${NC}"
    fi

    echo -e "${GREEN}  ✓ Python $PYTHON_VERSION detected ($PYTHON_CMD)${NC}"
}

# Create virtual environment
create_venv() {
    echo -e "${YELLOW}[2/4]${NC} Creating virtual environment..."

    VENV_DIR="$PROJECT_ROOT/venv"

    if [ -d "$VENV_DIR" ]; then
        echo -e "${YELLOW}  ! Virtual environment already exists. Skipping creation.${NC}"
    else
        $PYTHON_CMD -m venv "$VENV_DIR"
        echo -e "${GREEN}  ✓ Virtual environment created at ./venv${NC}"
    fi
}

# Activate venv and install requirements
install_requirements() {
    echo -e "${YELLOW}[3/4]${NC} Installing dependencies..."

    VENV_DIR="$PROJECT_ROOT/venv"

    # Activate virtual environment
    source "$VENV_DIR/bin/activate"

    # Upgrade pip
    pip install --upgrade pip --quiet

    # Install requirements
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        pip install -r "$PROJECT_ROOT/requirements.txt" --quiet
        echo -e "${GREEN}  ✓ Dependencies installed from requirements.txt${NC}"
    else
        echo -e "${YELLOW}  ! requirements.txt not found. Skipping.${NC}"
    fi
}

# Install package in editable mode
install_package() {
    echo -e "${YELLOW}[4/4]${NC} Installing highnoon package..."

    VENV_DIR="$PROJECT_ROOT/venv"
    source "$VENV_DIR/bin/activate"

    if [ -f "$PROJECT_ROOT/pyproject.toml" ]; then
        pip install -e "$PROJECT_ROOT" --quiet
        echo -e "${GREEN}  ✓ Package installed in editable mode${NC}"
    else
        echo -e "${YELLOW}  ! pyproject.toml not found. Skipping package install.${NC}"
    fi
}

# Print success message
print_success() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    Setup Complete!                           ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "To activate the virtual environment:"
    echo -e "  ${BLUE}source venv/bin/activate${NC}"
    echo ""
    echo -e "To launch the WebUI:"
    echo -e "  ${BLUE}./scripts/launch_webui.sh${NC}"
    echo ""
    echo -e "To run the CLI:"
    echo -e "  ${BLUE}highnoon --help${NC}"
    echo ""
}

# Main execution
cd "$PROJECT_ROOT"
check_python_version
create_venv
install_requirements
install_package
print_success
