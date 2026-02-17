#!/bin/bash
set -euo pipefail

# Only run in Claude Code on the web (remote sessions)
if [ "${CLAUDE_CODE_REMOTE:-}" != "true" ]; then
  echo "Skipping session start hook - not in remote environment"
  exit 0
fi

echo "🚀 Starting session setup..."

# Navigate to the parent directory where we'll clone repos
cd /home/user

# Clone all Wanly repos (idempotently)
echo "📦 Cloning repositories..."

if [ ! -d "wanly-api" ]; then
  echo "  - Cloning wanly-api..."
  git clone https://github.com/DavidJBarnes/wanly-api
else
  echo "  - wanly-api already exists, pulling latest changes..."
  cd wanly-api && git pull origin main 2>/dev/null || git pull 2>/dev/null || true
  cd /home/user
fi

if [ ! -d "wanly-console" ]; then
  echo "  - Cloning wanly-console..."
  git clone https://github.com/DavidJBarnes/wanly-console
else
  echo "  - wanly-console already exists, pulling latest changes..."
  cd wanly-console && git pull origin main 2>/dev/null || git pull 2>/dev/null || true
  cd /home/user
fi

if [ ! -d "wanly-gpu-daemon" ]; then
  echo "  - Cloning wanly-gpu-daemon..."
  git clone https://github.com/DavidJBarnes/wanly-gpu-daemon
else
  echo "  - wanly-gpu-daemon already exists, pulling latest changes..."
  cd wanly-gpu-daemon && git pull origin main 2>/dev/null || git pull 2>/dev/null || true
  cd /home/user
fi

if [ ! -d "wanly-runpod" ]; then
  echo "  - Cloning wanly-runpod..."
  git clone https://github.com/DavidJBarnes/wanly-runpod
else
  echo "  - wanly-runpod already exists, pulling latest changes..."
  cd wanly-runpod && git pull origin main 2>/dev/null || git pull 2>/dev/null || true
  cd /home/user
fi

# Install Python dependencies for wanly-api
echo "📚 Installing Python dependencies for wanly-api..."
cd /home/user/wanly-api
if [ -f "requirements.txt" ]; then
  pip install -q -r requirements.txt
  echo "  ✓ Dependencies installed"
else
  echo "  ! No requirements.txt found"
fi

echo "✅ Session setup complete!"
echo ""
echo "Available repos:"
echo "  - /home/user/wanly-api"
echo "  - /home/user/wanly-console"
echo "  - /home/user/wanly-gpu-daemon"
echo "  - /home/user/wanly-runpod"
