#!/bin/bash
set -e

DEST_DIR="/workspace/Advanced-Reading-On-Computer-Vision/Lap03_3DCNN_Action_Recognition/dataset"
mkdir -p "$DEST_DIR"
cd "$DEST_DIR"

URL="https://www.crcv.ucf.edu/data/UCF50.rar"
TARGET_SIZE=3233554570

echo "[+] Resuming download of UCF50.rar..."

while true; do
    CURRENT_SIZE=$(stat -c%s "UCF50.rar" 2>/dev/null || echo 0)
    if [ "$CURRENT_SIZE" -ge "$TARGET_SIZE" ]; then
        echo "[+] Download complete: $CURRENT_SIZE bytes."
        break
    fi
    echo "[+] Current size: $CURRENT_SIZE / $TARGET_SIZE bytes. Downloading..."
    wget -c --no-check-certificate --timeout=30 --tries=5 "$URL" || true
    sleep 2
done

echo "[+] Verifying and extracting archive..."
unar -f -d UCF50.rar

if [ -d "UCF50/UCF50" ]; then
    mv UCF50/UCF50/* UCF50/
    rmdir UCF50/UCF50 2>/dev/null || true
fi

echo "[+] Cleaning up archive..."
rm -f UCF50.rar

echo "[+] Dataset UCF50 is ready at $DEST_DIR/UCF50"
