import torch

def main():
    print("Hello from torch-devcontainer!")
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. Number of GPUs:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA is not available. Using CPU.")


if __name__ == "__main__":
    main()
