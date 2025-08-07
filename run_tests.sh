#!/bin/bash

# TubeSensei Test Runner Script
# Provides various test running options with proper environment setup

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default values
TEST_TYPE="all"
COVERAGE=true
VERBOSE=false
PARALLEL=false
PERFORMANCE=false
INTEGRATION=false
UNIT_ONLY=false
HTML_REPORT=false

# Function to print usage
usage() {
    echo -e "${BLUE}TubeSensei Test Runner${NC}"
    echo ""
    echo "Usage: $0 [options]"
    echo ""
    echo "Options:"
    echo "  -t, --type TYPE        Test type: unit, integration, performance, or all (default: all)"
    echo "  -c, --no-coverage      Disable coverage reporting"
    echo "  -v, --verbose          Verbose output"
    echo "  -p, --parallel         Run tests in parallel"
    echo "  -f, --fast             Fast mode (unit tests only, no coverage)"
    echo "  -i, --integration      Run integration tests only"
    echo "  -P, --performance      Run performance tests only"
    echo "  -h, --html             Generate HTML coverage report"
    echo "  --help                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0                     Run all tests with coverage"
    echo "  $0 -f                  Fast unit tests only"
    echo "  $0 -i                  Integration tests only"
    echo "  $0 -P                  Performance tests only"
    echo "  $0 -p -h               All tests in parallel with HTML report"
    echo ""
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -t|--type)
            TEST_TYPE="$2"
            shift 2
            ;;
        -c|--no-coverage)
            COVERAGE=false
            shift
            ;;
        -v|--verbose)
            VERBOSE=true
            shift
            ;;
        -p|--parallel)
            PARALLEL=true
            shift
            ;;
        -f|--fast)
            UNIT_ONLY=true
            COVERAGE=false
            shift
            ;;
        -i|--integration)
            INTEGRATION=true
            TEST_TYPE="integration"
            shift
            ;;
        -P|--performance)
            PERFORMANCE=true
            TEST_TYPE="performance"
            shift
            ;;
        -h|--html)
            HTML_REPORT=true
            shift
            ;;
        --help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Setup environment
echo -e "${BLUE}Setting up test environment...${NC}"

# Check if we're in a virtual environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo -e "${YELLOW}Warning: Not in a virtual environment${NC}"
fi

# Check if required packages are installed
if ! python -c "import pytest" 2>/dev/null; then
    echo -e "${RED}Error: pytest not installed. Run: pip install -r requirements.txt${NC}"
    exit 1
fi

# Set test database URL
export TEST_DATABASE_URL="${TEST_DATABASE_URL:-postgresql+asyncpg://localhost:5432/tubesensei_test}"

# Create test database if it doesn't exist
echo -e "${BLUE}Checking test database...${NC}"
if ! python -c "
import asyncio
import asyncpg
import sys

async def check_db():
    try:
        conn = await asyncpg.connect('$TEST_DATABASE_URL')
        await conn.close()
        return True
    except:
        return False

if not asyncio.run(check_db()):
    print('Test database not accessible')
    sys.exit(1)
" 2>/dev/null; then
    echo -e "${YELLOW}Warning: Test database not accessible. Some tests may fail.${NC}"
fi

# Build pytest command
PYTEST_CMD="pytest"

# Add coverage options
if [[ "$COVERAGE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD --cov=tubesensei/app --cov-report=term-missing"
    
    if [[ "$HTML_REPORT" == true ]]; then
        PYTEST_CMD="$PYTEST_CMD --cov-report=html:htmlcov"
    fi
    
    PYTEST_CMD="$PYTEST_CMD --cov-report=xml --cov-fail-under=80"
fi

# Add verbosity
if [[ "$VERBOSE" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -v"
fi

# Add parallel execution
if [[ "$PARALLEL" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -n auto"
fi

# Add test selection based on type
case $TEST_TYPE in
    unit)
        PYTEST_CMD="$PYTEST_CMD tubesensei/tests/unit"
        ;;
    integration)
        PYTEST_CMD="$PYTEST_CMD tubesensei/tests/integration -m integration"
        ;;
    performance)
        PYTEST_CMD="$PYTEST_CMD tubesensei/tests/performance -m performance"
        ;;
    all)
        if [[ "$UNIT_ONLY" == true ]]; then
            PYTEST_CMD="$PYTEST_CMD tubesensei/tests/unit"
        else
            PYTEST_CMD="$PYTEST_CMD tubesensei/tests"
        fi
        ;;
esac

# Add specific marker exclusions for regular runs
if [[ "$TEST_TYPE" == "all" && "$PERFORMANCE" == false && "$INTEGRATION" == false ]]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not performance and not integration'"
elif [[ "$UNIT_ONLY" == true ]]; then
    PYTEST_CMD="$PYTEST_CMD -m 'not performance and not integration'"
fi

# Display run configuration
echo -e "${BLUE}Test Run Configuration:${NC}"
echo "  Test Type: $TEST_TYPE"
echo "  Coverage: $COVERAGE"
echo "  Verbose: $VERBOSE"
echo "  Parallel: $PARALLEL"
echo "  HTML Report: $HTML_REPORT"
echo ""

# Run the tests
echo -e "${BLUE}Running tests...${NC}"
echo "Command: $PYTEST_CMD"
echo ""

start_time=$(date +%s)

if eval $PYTEST_CMD; then
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo ""
    echo -e "${GREEN}✓ Tests completed successfully in ${duration}s${NC}"
    
    if [[ "$COVERAGE" == true ]]; then
        echo -e "${BLUE}Coverage report generated${NC}"
        
        if [[ "$HTML_REPORT" == true ]]; then
            echo -e "${BLUE}HTML coverage report: htmlcov/index.html${NC}"
        fi
    fi
    
    exit 0
else
    end_time=$(date +%s)
    duration=$((end_time - start_time))
    echo ""
    echo -e "${RED}✗ Tests failed after ${duration}s${NC}"
    exit 1
fi