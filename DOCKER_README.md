# Docker Setup for RunPod

This Docker setup provides a CUDA-enabled container with SSH support, suitable for both interactive development and batch job execution on RunPod.

## Files

- `Dockerfile` - CUDA 12.4 base with Python and pixi (Mojo/MAX installed via pixi from conda channels)
- `entrypoint.sh` - Flexible entrypoint supporting multiple modes
- `.dockerignore` - Excludes unnecessary files from the build

## Building the Image

```bash
# Build locally
docker build -t chemtensor-mojo:latest .

# Build with specific public key
docker build --build-arg PUBLIC_KEY="ssh-rsa AAAA..." -t chemtensor-mojo:latest .
```

## Usage Modes

### 1. Interactive Mode (Default)

Starts SSH daemon and keeps container alive for remote access:

```bash
docker run -d -p 22:22 chemtensor-mojo:latest

# Or explicitly
docker run -d -p 22:22 chemtensor-mojo:latest start
```

### 2. Batch/Command Mode

Run a specific command and exit:

```bash
# Run a script
docker run chemtensor-mojo:latest run python3 /workspace/script.py

# Run pixi command
docker run chemtensor-mojo:latest run pixi run mojo /workspace/chemtensor_mojo/main.mojo
```

### 3. Jupyter Mode

Start Jupyter Lab server:

```bash
docker run -d -p 8888:8888 -p 22:22 chemtensor-mojo:latest jupyter
```

### 4. Health Check

Check container status:

```bash
docker run chemtensor-mojo:latest health
```

## Environment Variables

| Variable | Description |
|----------|-------------|
| `PUBLIC_KEY` | SSH public key for authentication |
| `RUNPOD_PUBLIC_KEY` | Alternative public key variable (RunPod) |
| `ROOT_PASSWORD` | Root password for SSH (fallback auth) |

## RunPod Deployment

### Push to Docker Hub

```bash
# Tag for Docker Hub
docker tag chemtensor-mojo:latest yourusername/chemtensor-mojo:latest

# Push
docker push yourusername/chemtensor-mojo:latest
```

### RunPod Template Settings

1. **Container Image**: `yourusername/chemtensor-mojo:latest`
2. **Docker Command**: Leave empty for interactive mode, or:
   - `jupyter` for Jupyter Lab
   - `run your-command` for batch jobs
3. **Exposed Ports**: 
   - `22/tcp` (SSH)
   - `8888/tcp` (Jupyter, if needed)
4. **Volume Mounts**:
   - `/workspace/data` - Persistent data
   - `/workspace/outputs` - Results

### SSH Connection

After deploying on RunPod:

```bash
# Find your pod's SSH connection info in the RunPod dashboard
ssh root@<pod-ip> -p <ssh-port> -i ~/.ssh/your_key
```

## Directory Structure in Container

```
/workspace/
├── chemtensor_mojo/    # Project files
├── data/               # Data directory (mount point)
├── outputs/            # Output directory (mount point)
└── logs/               # Log files
```

## Customization

### Adding Dependencies

Edit the `Dockerfile`:

```dockerfile
# Add apt packages
RUN apt-get update && apt-get install -y your-package

# Add Python packages
RUN pip3 install your-package
```

### Custom Startup Tasks

Edit `entrypoint.sh` and add to the `setup_workspace()` function:

```bash
setup_workspace() {
    # ... existing code ...
    
    # Your custom setup
    your_custom_setup_command
}
```

## Troubleshooting

### SSH Connection Refused

1. Check if sshd is running: `docker exec <container> pgrep sshd`
2. Check logs: `docker logs <container>`
3. Verify port mapping: `docker ps`

### GPU Not Available

1. Ensure NVIDIA Docker runtime is installed
2. Run with GPU: `docker run --gpus all ...`
3. Check GPU inside: `docker exec <container> nvidia-smi`

### Pixi Issues

```bash
# Reinstall pixi environment
cd /workspace/chemtensor_mojo
pixi install --force
```
