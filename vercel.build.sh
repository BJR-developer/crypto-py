#!/bin/bash

# Exit on error
set -e

# Install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt

# Create necessary directories
mkdir -p .vercel/output/functions/api

# Copy files to the output directory
cp -r api/* .vercel/output/functions/api/
cp main.py .vercel/output/functions/api/
cp requirements.txt .vercel/output/functions/api/

# Create config file
cat > .vercel/output/config.json << EOF
{
  "version": 3,
  "routes": [
    { "src": "/(.*)", "dest": "/api/index.py" }
  ]
}
EOF
