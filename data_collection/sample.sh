#!/bin/bash

# ========================
# Simulate CPU and Memory load for ML time series dataset
# ========================

# Set up trap to handle Ctrl+C
trap 'echo "\nStopping load test..."; exit 0' SIGINT

# Configuration
HEY="./hey.exe"  # Change path if needed
BASE_URL="http://localhost:3000"
DURATION="12h"    # Total test duration (12 hours)

# Function to print status
function log {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] ============================"
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
    echo "[$(date '+%m-%d %H:%M:%S')] ============================"
}

# Function to run memory test with 40-request limit
function run_memory_test {
    local duration=$1
    local qps=$2
    local concurrency=$3
    
    log "Running memory test for ${duration} with ${qps} QPS and ${concurrency} workers"
    
    # Run memory test in background
    $HEY -z ${duration} -q ${qps} -c ${concurrency} -t 60 ${BASE_URL}/memory &
    local mem_pid=$!
    
    # Monitor and clear memory every 40 requests
    while kill -0 $mem_pid 2>/dev/null; do
        # Clear memory every 40 requests
        curl -s "${BASE_URL}/clear-memory"
        sleep 0.5
    done
    
    wait $mem_pid
}

# Main test sequence
log "Starting 12-hour load test sequence"
start_time=$(date +%s)
target_time=$((start_time + 12 * 3600))  # 12 hours from now

while [ $(date +%s) -lt $target_time ]; do
    # Phase 1: Idle (5% of time)
    log "Phase 1: Idle period (5 minutes)"
    sleep 300

    # Phase 2: Moderate CPU Load (10 minutes)
    log "Phase 2: Moderate CPU Load (10 minutes)"
    $HEY -z 10m -q 20 -c 50 -t 30 ${BASE_URL}/cpu

    # Phase 3: Memory Load with 40-request limit (15 minutes)
    log "Phase 3: Memory Load (15 minutes)"
    run_memory_test "15m" 5 20

    # Phase 4: High CPU Load (20 minutes)
    log "Phase 4: High CPU Load (20 minutes)"
    $HEY -z 20m -q 50 -c 100 -t 30 ${BASE_URL}/cpu

    # Phase 5: Mixed Load - CPU + Memory (30 minutes)
    log "Phase 5: Mixed Load - CPU + Memory (30 minutes)"
    
    # Start CPU load in background
    $HEY -z 30m -q 30 -c 50 -t 30 ${BASE_URL}/cpu &
    cpu_pid=$!
    
    # Run memory test in parallel
    run_memory_test "30m" 3 15
    
    # Wait for CPU test to finish
    wait $cpu_pid

    # Phase 6: Recovery Period (10 minutes)
    log "Phase 6: Recovery Period (10 minutes)"
    sleep 600

    # Phase 7: Burst Load (5 minutes)
    log "Phase 7: Burst Load (5 minutes)"
    $HEY -z 5m -q 100 -c 200 -t 10 ${BASE_URL}/cpu &
    cpu_pid=$!
    
    # Run memory burst in parallel
    run_memory_test "5m" 10 30
    
    wait $cpu_pid
    
    # Clear memory after burst
    curl -s "${BASE_URL}/clear-memory"
    
    log "Completed one full test cycle"
done

log " 12-hour load test completed successfully"
