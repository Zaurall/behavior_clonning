#!/bin/bash

NUM_WORKERS=6
MAX_SEGMENT=10  # Change this as needed

echo "Starting $NUM_WORKERS workers for PGTO optimization..."

# Start all workers in background
for ((WORKER_ID=0; WORKER_ID<NUM_WORKERS; WORKER_ID++)); do
    echo "Starting worker $WORKER_ID..."
    python -m offline.run \
        -n $NUM_WORKERS \
        -w $WORKER_ID \
        -m $MAX_SEGMENT \
        -d cuda \
        > worker_${WORKER_ID}.log 2>&1 &
done

echo "All workers started. Run 'jobs' to see status, 'fg' to bring to foreground."
echo "Logs are being written to worker_*.log files"

# Wait for all workers to complete
wait
echo "All workers completed!"