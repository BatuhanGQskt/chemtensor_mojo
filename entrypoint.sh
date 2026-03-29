#!/bin/bash
# NOTE: Not using 'set -e' to prevent early exit on non-critical failures

# =============================================================================
# RunPod Entrypoint Script
# Supports both interactive (SSH) and non-interactive (batch) modes
# =============================================================================

# Colors for logging
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_header() {
    echo -e "${BLUE}========================================${NC}"
    echo -e "${BLUE}$1${NC}"
    echo -e "${BLUE}========================================${NC}"
}

# -----------------------------------------------------------------------------
# SSH Setup
# -----------------------------------------------------------------------------
setup_ssh() {
    log_header "Setting up SSH"

    # Handle RunPod's PUBLIC_KEY environment variable
    if [ -n "$PUBLIC_KEY" ]; then
        log_info "Adding public key from PUBLIC_KEY env"
        mkdir -p /root/.ssh
        echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
        chmod 700 /root/.ssh
    fi

    # Handle RUNPOD_PUBLIC_KEY (alternative env var name)
    if [ -n "$RUNPOD_PUBLIC_KEY" ]; then
        log_info "Adding RunPod public key from RUNPOD_PUBLIC_KEY env"
        mkdir -p /root/.ssh
        echo "$RUNPOD_PUBLIC_KEY" >> /root/.ssh/authorized_keys
        chmod 600 /root/.ssh/authorized_keys
        chmod 700 /root/.ssh
    fi

    # Set root password if provided (fallback authentication)
    if [ -n "$ROOT_PASSWORD" ]; then
        log_info "Setting root password from ROOT_PASSWORD env"
        echo "root:$ROOT_PASSWORD" | chpasswd
    else
        # Set a default password for RunPod (user can change it)
        log_info "Setting default root password: 'runpod'"
        echo "root:runpod" | chpasswd
    fi

    # Ensure SSH directory permissions are correct
    mkdir -p /var/run/sshd
    chmod 755 /var/run/sshd

    # Make sure sshd_config allows root login
    sed -i 's/^#*PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config
    sed -i 's/^#*PasswordAuthentication.*/PasswordAuthentication yes/' /etc/ssh/sshd_config

    # Start SSH daemon
    log_info "Starting SSH daemon..."
    /usr/sbin/sshd -D &
    SSHD_PID=$!
    sleep 1

    # Verify SSH is running
    if kill -0 $SSHD_PID 2>/dev/null; then
        log_info "SSH server started successfully (PID: $SSHD_PID)"
    else
        log_error "SSH server failed to start!"
        # Try starting without -D for debugging
        /usr/sbin/sshd
    fi

    log_info "SSH listening on port 22"
}

# -----------------------------------------------------------------------------
# Environment Setup
# -----------------------------------------------------------------------------
setup_environment() {
    log_header "Setting up environment"

    # Add pixi to PATH
    export PATH="/root/.pixi/bin:${PATH}"

    # CUDA environment (if available)
    if [ -d "/usr/local/cuda" ]; then
        export PATH="/usr/local/cuda/bin:${PATH}"
        export LD_LIBRARY_PATH="/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}"
        log_info "CUDA environment configured"
    fi

    # Print environment info
    log_info "Python: $(python3 --version 2>&1 || echo 'not found')"
    log_info "pixi: $(pixi --version 2>&1 || echo 'not found')"
    
    # Check if Mojo is available via pixi (non-blocking check)
    if [ -d "/workspace/chemtensor_mojo" ] && [ -f "/workspace/chemtensor_mojo/pixi.toml" ]; then
        log_info "Mojo project found at /workspace/chemtensor_mojo"
    fi

    # Check NVIDIA GPU
    if command -v nvidia-smi &> /dev/null; then
        log_info "GPU Status:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>/dev/null || log_warn "nvidia-smi failed"
    else
        log_warn "nvidia-smi not available"
    fi

    return 0
}

# -----------------------------------------------------------------------------
# Workspace Setup
# -----------------------------------------------------------------------------
setup_workspace() {
    log_header "Setting up workspace"

    # Create standard directories
    mkdir -p /workspace/data
    mkdir -p /workspace/outputs
    mkdir -p /workspace/logs

    # If chemtensor_mojo exists, set up pixi environment
    if [ -d "/workspace/chemtensor_mojo" ]; then
        log_info "Found chemtensor_mojo project"
        cd /workspace/chemtensor_mojo
        if [ -f "pixi.toml" ]; then
            log_info "Checking pixi environment..."
            # Only install if not already installed (to save time on restart)
            if [ ! -d ".pixi" ]; then
                log_info "Installing pixi dependencies (this may take a while)..."
                pixi install 2>&1 || log_warn "pixi install had warnings/errors"
            else
                log_info "Pixi environment already exists"
            fi
        fi
        cd /workspace
    fi

    log_info "Workspace ready at /workspace"
    return 0
}

# -----------------------------------------------------------------------------
# Run Command (for non-interactive mode)
# -----------------------------------------------------------------------------
run_command() {
    log_header "Running command"
    log_info "Command: $*"

    # Execute the command
    exec "$@"
}

# -----------------------------------------------------------------------------
# Interactive Mode (keep container alive for SSH)
# -----------------------------------------------------------------------------
interactive_mode() {
    log_header "Starting interactive mode"
    log_info "Container is running. SSH is available."

    # Print connection help
    echo ""
    echo "============================================"
    echo "SSH Connection Info:"
    echo "============================================"
    if [ -n "$RUNPOD_PUBLIC_IP" ] && [ -n "$RUNPOD_TCP_PORT_22" ]; then
        echo "  ssh root@${RUNPOD_PUBLIC_IP} -p ${RUNPOD_TCP_PORT_22}"
    else
        echo "  Check RunPod dashboard for SSH connection details"
        echo "  Or use: ssh root@<pod-ip> -p <ssh-port>"
    fi
    echo ""
    echo "Default password: runpod"
    echo "(or use your PUBLIC_KEY if set)"
    echo ""
    echo "Useful paths:"
    echo "  Workspace:  /workspace"
    echo "  Project:    /workspace/chemtensor_mojo"
    echo "  Data:       /workspace/data"
    echo "  Outputs:    /workspace/outputs"
    echo "============================================"
    echo ""

    # Keep container alive indefinitely
    log_info "Container ready. Waiting for connections..."
    
    # Use sleep infinity (more reliable than tail -f /dev/null)
    while true; do
        sleep 3600
    done
}

# -----------------------------------------------------------------------------
# Jupyter Mode
# -----------------------------------------------------------------------------
jupyter_mode() {
    log_header "Starting Jupyter Lab"

    cd /workspace

    # Start Jupyter with no token/password for RunPod (it handles auth)
    exec jupyter lab \
        --ip=0.0.0.0 \
        --port=8888 \
        --no-browser \
        --allow-root \
        --NotebookApp.token='' \
        --NotebookApp.password='' \
        --notebook-dir=/workspace
}

# -----------------------------------------------------------------------------
# Health Check
# -----------------------------------------------------------------------------
health_check() {
    log_header "Health Check"

    # Check SSH
    if pgrep -x "sshd" > /dev/null; then
        log_info "SSH: Running"
    else
        log_error "SSH: Not running"
    fi

    # Check GPU
    if nvidia-smi &> /dev/null; then
        log_info "GPU: Available"
    else
        log_warn "GPU: Not available"
    fi

    # Check pixi
    if command -v pixi &> /dev/null; then
        log_info "Pixi: $(pixi --version)"
    else
        log_warn "Pixi: Not found"
    fi

    # Check Mojo via pixi
    if [ -d "/workspace/chemtensor_mojo" ]; then
        cd /workspace/chemtensor_mojo
        if pixi run mojo --version &> /dev/null; then
            log_info "Mojo: $(pixi run mojo --version 2>&1)"
        else
            log_warn "Mojo: Not installed (run 'pixi install' in /workspace/chemtensor_mojo)"
        fi
        cd /workspace
    fi

    # Check workspace
    if [ -d "/workspace" ]; then
        log_info "Workspace: OK"
        ls -la /workspace
    fi
}

# -----------------------------------------------------------------------------
# Main Entry Point
# -----------------------------------------------------------------------------
main() {
    log_header "RunPod Container Starting"
    log_info "Timestamp: $(date)"
    log_info "Hostname: $(hostname)"
    
    # Print RunPod environment info for debugging
    if [ -n "$RUNPOD_POD_ID" ]; then
        log_info "RunPod Pod ID: $RUNPOD_POD_ID"
    fi
    if [ -n "$RUNPOD_PUBLIC_IP" ]; then
        log_info "RunPod Public IP: $RUNPOD_PUBLIC_IP"
    fi
    if [ -n "$RUNPOD_TCP_PORT_22" ]; then
        log_info "RunPod SSH Port: $RUNPOD_TCP_PORT_22"
    fi

    # Start SSH FIRST (critical for RunPod access)
    setup_ssh

    # Set up environment (non-critical, don't fail if issues)
    setup_environment || log_warn "Environment setup had issues, continuing..."

    # Set up workspace (non-critical)
    setup_workspace || log_warn "Workspace setup had issues, continuing..."

    # Parse command/mode
    case "${1:-start}" in
        start|interactive)
            # Default: interactive mode with SSH
            interactive_mode
            ;;
        jupyter)
            # Start Jupyter Lab
            jupyter_mode
            ;;
        health)
            # Health check and exit
            health_check
            ;;
        run)
            # Run a specific command (non-interactive batch mode)
            shift
            run_command "$@"
            ;;
        bash|shell)
            # Start an interactive bash shell
            exec /bin/bash
            ;;
        *)
            # Run whatever command was passed
            run_command "$@"
            ;;
    esac
}

# Run main with all arguments
main "$@"
