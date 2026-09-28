#!/usr/bin/env bash
# transfer_to_pi.sh
# ------------------
# Deploys the Smart Helmet edge package to a Raspberry Pi over your local network.
#
# Usage:
#   ./transfer_to_pi.sh <PI_IP_ADDRESS> [PI_USERNAME] [REMOTE_DEST_DIR]
#
# Example:
#   ./transfer_to_pi.sh 192.168.1.100 pi ~/smart_helmet_ai
#
# Or generate a standalone tarball to copy via USB or AirDrop:
#   ./transfer_to_pi.sh --archive

set -e

PI_HOST="${1:-}"
PI_USER="${2:-pi}"
REMOTE_DIR="${3:-~/smart_helmet_ai}"
ARCHIVE_NAME="smart_helmet_pi_deploy.tar.gz"

cd "$(dirname "$0")"

# Option 1: Standalone tarball generation
if [ "$PI_HOST" == "--archive" ] || [ -z "$PI_HOST" ]; then
    echo "=========================================================="
    echo "Creating lightweight Raspberry Pi deployment archive..."
    echo "=========================================================="
    
    tar --exclude='.venv' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='results' \
        --exclude='data/*.csv' \
        -czvf "$ARCHIVE_NAME" \
        models \
        predict.py \
        rpi_helmet_edge.py \
        rpi_arduino_bridge.py \
        dashboard_server.py \
        requirements_pi.txt \
        README.md

    echo ""
    echo "Deployment archive successfully created: $ARCHIVE_NAME"
    echo "Archive size: $(du -h "$ARCHIVE_NAME" | cut -f1)"
    echo ""
    echo "To copy this archive manually to your Pi:"
    echo "  scp $ARCHIVE_NAME pi@<PI_IP>:~/"
    echo ""
    echo "On your Raspberry Pi, extract and run:"
    echo "  tar -xzvf $ARCHIVE_NAME"
    echo "  cd smart_helmet_ai (or extracted folder)"
    echo "  pip install -r requirements_pi.txt"
    echo "  python3 dashboard_server.py --host 0.0.0.0 --port 8080"
    echo "  (or: python3 rpi_arduino_bridge.py)"
    echo "=========================================================="
    exit 0
fi

# Option 2: Direct network sync via rsync / scp
echo "=========================================================="
echo "Deploying Smart Helmet package to $PI_USER@$PI_HOST:$REMOTE_DIR"
echo "=========================================================="

# Check if rsync is available, fallback to scp
if command -v rsync &> /dev/null; then
    echo "Using rsync..."
    ssh "$PI_USER@$PI_HOST" "mkdir -p $REMOTE_DIR"
    rsync -avz --progress \
        --exclude='.venv' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='results' \
        --exclude='data/*.csv' \
        ./ "$PI_USER@$PI_HOST:$REMOTE_DIR/"
else
    echo "rsync not found, using scp..."
    tar --exclude='.venv' \
        --exclude='__pycache__' \
        --exclude='.git' \
        --exclude='results' \
        --exclude='data/*.csv' \
        -czvf "$ARCHIVE_NAME" \
        models predict.py rpi_helmet_edge.py rpi_arduino_bridge.py dashboard_server.py requirements_pi.txt README.md
    
    ssh "$PI_USER@$PI_HOST" "mkdir -p $REMOTE_DIR"
    scp "$ARCHIVE_NAME" "$PI_USER@$PI_HOST:$REMOTE_DIR/"
    ssh "$PI_USER@$PI_HOST" "tar -xzvf $REMOTE_DIR/$ARCHIVE_NAME -C $REMOTE_DIR && rm $REMOTE_DIR/$ARCHIVE_NAME"
fi

echo ""
echo "Deployment transfer complete!"
echo ""
echo "Next steps on your Raspberry Pi:"
echo "  1. SSH into the Pi:"
echo "     ssh $PI_USER@$PI_HOST"
echo "  2. Go to the project directory:"
echo "     cd $REMOTE_DIR"
echo "  3. Connect Arduino Uno to Raspberry Pi via USB cable"
echo "  4. Install edge dependencies:"
echo "     pip install -r requirements_pi.txt"
echo "  5. Start the Full Web HUD + GPS Navigation Dashboard:"
echo "     python3 dashboard_server.py --host 0.0.0.0 --port 8080"
echo "     (Or run headless AI console: python3 rpi_arduino_bridge.py)"
echo "=========================================================="
