import os
import sys
import torch


def check_debugger():
    has_trace = hasattr(sys, 'gettrace') and sys.gettrace() is not None
    has_breakpoint = sys.breakpointhook.__module__ != "sys"
    is_debug = has_trace or has_breakpoint
    
    if is_debug:
        print("To debug C++ code, use the following command:")
        print(f"  gdb -p {os.getpid()}")
        print(
            "After attaching gdb, set a breakpoint in the C++ code and continue the Python script."
        )
        breakpoint()


def check_cuda():
    # Check if CUDA is available
    if torch.cuda.is_available():
        print("CUDA is available. Number of GPUs:", torch.cuda.device_count())
        print("Current device:", torch.cuda.current_device())
        print("Device name:", torch.cuda.get_device_name(torch.cuda.current_device()))
    else:
        print("CUDA is not available.")
        sys.exit(1)


def test_custom_ops():
    print("Test custom operators ...")
    try:
        from .libops import my_add, my_add_cuda
        check_debugger()
    except ImportError as e:
        print("Error importing custom operations:", e)
        return
    a = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    b = torch.tensor([4.0, 5.0, 6.0], device="cuda")
    result = my_add(a, b)
    print("Result of my_add:", result)
    # Example usage of the custom operations with CUDA
    result_cuda = my_add_cuda(a, b)
    print("Result of my_add_cuda:", result_cuda)


def main():
    print("Hello from torch-devcontainer!")
    check_cuda()
    test_custom_ops()


if __name__ == "__main__":
    main()
