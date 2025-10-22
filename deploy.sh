#!/bin/bash

# Usage: ./start_script.sh [cuda|local]
# Default: local

MODE=${1:-local}
SERVER_NAME=gen-image-detector

# Validate input

if [[ "$MODE" != "cuda" && "$MODE" != "local" ]]; then
    echo "Error: Invalid mode. Use 'local' or 'cuda'"
    echo "Usage: $0 [local|cuda]"
    exit 1
fi

echo "Starting GEN IMAGE DETECTOR in $MODE mode..."

# Common: Docker cleanup
cd scripts/docker
./docker_clear.sh $SERVER_NAME
cd ../..

echo "Starting local environment with docker-compose..."
docker-compose -f docker-compose.yml up

echo "$SERVER_NAME ($MODE mode) started successfully!"