#!/bin/bash

# Set NMON parameters
INTERVAL=5
COUNT=60

# Capture start time to filter new files
START_TIME=$(date +%s)

# Start nmon in background (no -o, use default)
nmon -f -s $INTERVAL -c $COUNT &
NMON_PID=$!
echo "Started nmon (PID: $NMON_PID)"

# Run your Python script
echo "Running test13.py"
python3 test13.py
echo "Finished test13.py"

# Wait for nmon to complete (optional, or you can kill if needed)
wait $NMON_PID
echo "nmon finished"

# Find the most recent .nmon file created after script start
LATEST_NMON=$(find . -maxdepth 1 -type f -name "*.nmon" -newermt "@$START_TIME" | sort | tail -n 1)

if [[ -n "$LATEST_NMON" ]]; then
    FULL_PATH=$(readlink -f "$LATEST_NMON")
    echo "✅ NMON file saved at: $FULL_PATH"
else
    echo "❌ No .nmon file found."
fi

