#!/bin/bash
"""
Enhanced ResiLink: Hybrid Implementation Runner
==============================================

Simple script to run the complete hybrid implementation.
This script handles the full workflow from controller to optimization.
"""

set -e  # Exit on any error

echo "🚀 Enhanced ResiLink: Hybrid Implementation Setup"
echo "=================================================="

# Check if running as root for Mininet
if [[ $EUID -eq 0 ]]; then
    echo "⚠️  Running as root - Mininet topology will be available"
    MININET_AVAILABLE=true
else
    echo "ℹ️  Not running as root - Mininet topology not available"
    echo "   Run 'sudo ./run_hybrid_implementation.sh' for full functionality"
    MININET_AVAILABLE=false
fi

# Function to check if process is running
check_process() {
    pgrep -f "$1" > /dev/null
}

# Function to wait for service
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1
    
    echo "⏳ Waiting for $service_name to be ready..."
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "✅ $service_name is ready!"
            return 0
        fi
        
        echo "   Attempt $attempt/$max_attempts - waiting..."
        sleep 2
        ((attempt++))
    done
    
    echo "❌ $service_name failed to start after $max_attempts attempts"
    return 1
}

# Function to cleanup processes
cleanup() {
    echo "🧹 Cleaning up processes..."
    
    # Kill Ryu controller
    if check_process "ryu-manager"; then
        echo "   Stopping Ryu controller..."
        pkill -f "ryu-manager" || true
    fi
    
    # Clean Mininet
    if [ "$MININET_AVAILABLE" = true ]; then
        echo "   Cleaning Mininet..."
        mn -c > /dev/null 2>&1 || true
    fi
    
    echo "✅ Cleanup complete"
}

# Trap cleanup on exit
trap cleanup EXIT

# Step 1: Start Ryu Controller
echo ""
echo "📡 Step 1: Starting Enhanced Academic SDN Controller"
echo "---------------------------------------------------"

if check_process "ryu-manager"; then
    echo "⚠️  Ryu controller already running - stopping it first"
    pkill -f "ryu-manager" || true
    sleep 2
fi

echo "🚀 Starting Ryu controller..."
ryu-manager src/sdn_controller/enhanced_academic_controller.py \
    --observe-links \
    --wsapi-host 0.0.0.0 \
    --wsapi-port 8080 \
    --verbose &

RYU_PID=$!
echo "   Controller PID: $RYU_PID"

# Wait for controller to be ready
if ! wait_for_service "http://localhost:8080/v1.0/topology/switches" "Ryu Controller"; then
    echo "❌ Failed to start Ryu controller"
    exit 1
fi

# Step 2: Create Mininet Topology (if running as root)
if [ "$MININET_AVAILABLE" = true ]; then
    echo ""
    echo "🌐 Step 2: Creating Mininet Topology"
    echo "------------------------------------"
    
    # Clean any existing Mininet state
    mn -c > /dev/null 2>&1 || true
    
    echo "🏗️  Creating linear topology with 4 switches..."
    
    # Create topology in background
    python examples/mininet_topology_demo.py \
        --topology linear \
        --switches 4 \
        --hosts-per-switch 2 \
        --controller-ip 127.0.0.1 \
        --controller-port 6653 \
        --duration 300 &  # Run for 5 minutes
    
    MININET_PID=$!
    echo "   Mininet PID: $MININET_PID"
    
    # Wait for topology to be established
    echo "⏳ Waiting for topology to be established..."
    sleep 10
    
    # Check if topology is visible to controller
    SWITCH_COUNT=0
    for i in {1..15}; do
        SWITCH_COUNT=$(curl -s "http://localhost:8080/v1.0/topology/switches" | jq '. | length' 2>/dev/null || echo "0")
        if [ "$SWITCH_COUNT" -gt 0 ]; then
            echo "✅ Topology established: $SWITCH_COUNT switches detected"
            break
        fi
        echo "   Waiting for switches... (attempt $i/15)"
        sleep 2
    done
    
    if [ "$SWITCH_COUNT" -eq 0 ]; then
        echo "⚠️  No switches detected - continuing anyway"
    fi
else
    echo ""
    echo "⚠️  Step 2: Mininet Topology Skipped"
    echo "-----------------------------------"
    echo "   Run as root (sudo) to enable Mininet topology"
    echo "   The implementation will work with any existing topology"
fi

# Step 3: Run Hybrid Implementation
echo ""
echo "🤖 Step 3: Running Hybrid Implementation"
echo "---------------------------------------"

echo "🚀 Starting hybrid optimization..."
echo "   - GNN: Graph Neural Network for pattern learning"
echo "   - RL: Reinforcement Learning for adaptive optimization"
echo "   - Academic justification for all parameters"

# Parse command line arguments for the implementation
IMPL_ARGS=""
CYCLES=5
INTERVAL=30
TRAINING_MODE=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cycles)
            CYCLES="$2"
            shift 2
            ;;
        --interval)
            INTERVAL="$2"
            shift 2
            ;;
        --training)
            TRAINING_MODE="--training-mode"
            shift
            ;;
        --single)
            IMPL_ARGS="$IMPL_ARGS --single-cycle"
            shift
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--cycles N] [--interval N] [--training] [--single]"
            exit 1
            ;;
    esac
done

# Run the implementation
python hybrid_resilink_implementation.py \
    --ryu-url "http://localhost:8080" \
    --max-cycles $CYCLES \
    --cycle-interval $INTERVAL \
    $TRAINING_MODE \
    $IMPL_ARGS

IMPL_EXIT_CODE=$?

# Step 4: Results Summary
echo ""
echo "📊 Step 4: Results Summary"
echo "-------------------------"

if [ $IMPL_EXIT_CODE -eq 0 ]; then
    echo "✅ Hybrid implementation completed successfully!"
    
    # Show generated files
    echo ""
    echo "📁 Generated Files:"
    if ls link_suggestion_cycle_*.json 1> /dev/null 2>&1; then
        for file in link_suggestion_cycle_*.json; do
            echo "   📄 $file"
        done
    fi
    
    if [ -f "hybrid_optimization_history.json" ]; then
        echo "   📄 hybrid_optimization_history.json (complete history)"
    fi
    
    if [ -f "hybrid_resilink.log" ]; then
        echo "   📄 hybrid_resilink.log (detailed logs)"
    fi
    
    # Show sample result if available
    if [ -f "link_suggestion_cycle_1.json" ]; then
        echo ""
        echo "📋 Sample Link Suggestion:"
        echo "-------------------------"
        cat link_suggestion_cycle_1.json | jq '{
            src_dpid: .src_dpid,
            dst_dpid: .dst_dpid,
            src_port: .src_port,
            dst_port: .dst_port,
            score: .score,
            implementation_feasible: .implementation_feasible,
            academic_justification: .academic_justification
        }' 2>/dev/null || cat link_suggestion_cycle_1.json
    fi
    
    echo ""
    echo "🎯 Implementation Notes:"
    echo "   • All link suggestions include academic justification"
    echo "   • GNN learns patterns from network structure"
    echo "   • RL adapts optimization strategy over time"
    echo "   • Ensemble method combines both approaches"
    echo "   • Ready for thesis defense with complete citations"
    
else
    echo "❌ Hybrid implementation failed (exit code: $IMPL_EXIT_CODE)"
    echo ""
    echo "🔍 Troubleshooting:"
    echo "   • Check if Ryu controller is accessible: curl http://localhost:8080/v1.0/topology/switches"
    echo "   • Verify network topology exists"
    echo "   • Check logs in hybrid_resilink.log"
    echo "   • Ensure PyTorch is installed for ML components"
fi

echo ""
echo "🏁 Implementation Complete"
echo "========================="

exit $IMPL_EXIT_CODE