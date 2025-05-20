#!/bin/bash
set -e

# Default values
DEFAULT_SHM_SIZE="8g"
DEFAULT_GPU_FLAGS="all"

# Parse command line arguments
POSITIONAL_ARGS=()
CUDA_VERSION=""
UBUNTU_MIRROR="https://mirrors.bfsu.edu.cn/ubuntu/"
PYPI_MIRROR="https://mirrors.aliyun.com/pypi/simple/"
HUGGINGFACE_MIRROR="https://hf-mirror.com"
GPU_FLAGS=$DEFAULT_GPU_FLAGS
SHM_SIZE=$DEFAULT_SHM_SIZE

function print_usage() {
    echo "Usage: $0 [options] [command]"
    echo ""
    echo "Commands:"
    echo "  setup                   Setup the development container"
    echo "  [other commands]        Run command in the container"
    echo ""
    echo "Options:"
    echo "  --cuda-version VERSION  Set CUDA version (e.g. 12.2.0)"
    echo "  --ubuntu-mirror URL     Set Ubuntu mirror URL"
    echo "  --pypi-mirror URL       Set PyPI mirror URL"
    echo "  --hf-mirror URL         Set Hugging Face mirror URL"
    echo "  --gpu-flags FLAGS       Set GPU flags (default: all)"
    echo "  --shm-size SIZE         Set shared memory size (default: 8g)"
    echo "  --                      End of script options (following arguments are passed to container)"
    echo "  --help                  Show this help message"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-version)
            CUDA_VERSION="$2"
            shift 2
            ;;
        --ubuntu-mirror)
            UBUNTU_MIRROR="$2"
            shift 2
            ;;
        --pypi-mirror)
            PYPI_MIRROR="$2"
            shift 2
            ;;
        --hf-mirror)
            HUGGINGFACE_MIRROR="$2"
            shift 2
            ;;
        --gpu-flags)
            GPU_FLAGS="$2"
            shift 2
            ;;
        --shm-size)
            SHM_SIZE="$2"
            shift 2
            ;;
        --help)
            print_usage
            ;;
        --)
            # Everything after -- is considered part of the command
            shift
            POSITIONAL_ARGS+=("$@")
            break
            ;;
        -*)
            if [ -z "$COMMAND" ]; then
                echo "Unknown option: $1"
                print_usage
            else
                # If we've already found a command, store this and all remaining args
                POSITIONAL_ARGS+=("$@")
                break
            fi
            ;;
        *)
            POSITIONAL_ARGS+=("$1")
            shift
            ;;
    esac
done

set -- "${POSITIONAL_ARGS[@]}"
COMMAND=${1:-""}

# Check if the user is root
if [ "$(id -u)" -eq 0 ]; then
    echo "Error: This script should not be run as root."
    echo "Please run as a regular user with Docker permissions."
    exit 1
fi

# Get user information
USER_NAME=$(whoami)
USER_UID=$(id -u)
USER_GID=$(id -g)

# Detect CUDA version if not provided
if [ -z "$CUDA_VERSION" ]; then
    if ! command -v nvidia-smi &> /dev/null; then
        echo "Error: nvidia-smi not found. Please install NVIDIA drivers or specify CUDA version manually."
        exit 1
    fi
    
    CUDA_VERSION_PREFIX=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
    if [ -z "$CUDA_VERSION_PREFIX" ]; then
        echo "Error: Could not detect CUDA version."
        exit 1
    fi
    
    # Add .0 suffix to convert from 12.2 to 12.2.0 format
    CUDA_VERSION="${CUDA_VERSION_PREFIX}.0"
    echo "Detected CUDA version: $CUDA_VERSION"
else
    # Verify that the provided CUDA version is compatible with the installed driver
    if command -v nvidia-smi &> /dev/null; then
        DRIVER_CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | awk '{print $9}')
        PROVIDED_MAJOR_MINOR=$(echo $CUDA_VERSION | cut -d. -f1,2)
        
        if [ "$DRIVER_CUDA_VERSION" != "$PROVIDED_MAJOR_MINOR" ]; then
            echo "Warning: Provided CUDA version ($CUDA_VERSION) may not be compatible with installed driver (CUDA $DRIVER_CUDA_VERSION)"
            echo "Continuing anyway, but this might cause issues..."
            sleep 2
        fi
    fi
fi

# Generate image tag
IMAGE_TAG="uv-devcontainer:cuda${CUDA_VERSION}-${USER_NAME}"
DOCKER_FILE_PATH=".devcontainer/Dockerfile"
DEV_CONTAINER_TEMPLATE=".devcontainer/devcontainer.template.json"
DEV_CONTAINER_JSON=".devcontainer/devcontainer.json"

# Get the repository directory name
REPO_DIR=$(basename "$(pwd)")

# Ensure cache directories exist with correct permissions
mkdir -p ~/.cache/huggingface
mkdir -p ~/.cache/uv
mkdir -p ~/.cache/torch/hub

if [ "$COMMAND" = "setup" ]; then
    echo "Setting up development container..."
    
    # Build the Docker image
    echo "Building Docker image: $IMAGE_TAG"
    docker build \
        --build-arg CUDA_VERSION=$CUDA_VERSION \
        --build-arg UBUNTU_MIRROR=$UBUNTU_MIRROR \
        --build-arg USER_NAME=$USER_NAME \
        --build-arg USER_UID=$USER_UID \
        --build-arg USER_GID=$USER_GID \
        --build-arg PYPI_MIRROR=$PYPI_MIRROR \
        --build-arg HUGGINGFACE_MIRROR=$HUGGINGFACE_MIRROR \
        -t $IMAGE_TAG \
        -f $DOCKER_FILE_PATH .devcontainer
    
    # Generate the devcontainer.json file from template
    echo "Generating devcontainer.json from template..."
    sed \
        -e "s|%BUILT_IMAGE%|$IMAGE_TAG|g" \
        -e "s|%USER_NAME%|$USER_NAME|g" \
        -e "s|%GPU_FLAGS%|$GPU_FLAGS|g" \
        -e "s|%SHM_SIZE%|$SHM_SIZE|g" \
        $DEV_CONTAINER_TEMPLATE > $DEV_CONTAINER_JSON
    
    echo "Setup complete! You can now open the project in VSCode and use the 'Reopen in Container' option."

else
    if [ -z "$COMMAND" ]; then
        # If no command is provided, show help
        print_usage
    else
        # Check if the image exists
        if ! docker image inspect $IMAGE_TAG &> /dev/null; then
            echo "Error: Docker image $IMAGE_TAG not found. Please run '$0 setup' first."
            exit 1
        fi
        
        # Run the container with the provided command
        echo "Running command in container: $@"
        
        # Create a complete command with all necessary Docker flags
        docker run --rm -it \
            --add-host=host.docker.internal:host-gateway \
            --gpus $GPU_FLAGS \
            --shm-size=$SHM_SIZE \
            -v $(pwd):/workspaces/$REPO_DIR \
            -v ~/.cache/huggingface:/home/$USER_NAME/.cache/huggingface \
            -v ~/.cache/uv:/home/$USER_NAME/.cache/uv \
            -v ~/.cache/torch/hub:/home/$USER_NAME/.cache/torch/hub \
            -w /workspaces/$REPO_DIR \
            --user $USER_NAME \
            $IMAGE_TAG \
            "$@"
    fi
fi