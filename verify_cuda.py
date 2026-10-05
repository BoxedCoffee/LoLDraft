#!/usr/bin/env python3
"""
CUDA Verification Script
Checks if CUDA is properly installed and available for PyTorch
"""

import torch
import sys

def verify_cuda():
    print("=== CUDA Verification ===")

    # Check Python version
    print(f"Python version: {sys.version}")

    # Check PyTorch version
    print(f"PyTorch version: {torch.__version__}")

    # Check if CUDA is available
    cuda_available = torch.cuda.is_available()
    print(f"CUDA available: {cuda_available}")

    if cuda_available:
        # Get CUDA version
        try:
            cuda_version = torch.version.cuda
            print(f"CUDA version: {cuda_version}")
        except Exception as e:
            print(f"Could not determine CUDA version: {e}")

        # Get GPU count
        gpu_count = torch.cuda.device_count()
        print(f"Number of GPUs: {gpu_count}")

        # Get device properties for first GPU
        if gpu_count > 0:
            device_name = torch.cuda.get_device_name(0)
            print(f"GPU 0 name: {device_name}")

            # Get memory info
            try:
                memory_info = torch.cuda.get_device_properties(0)
                total_memory = memory_info.total_memory / (1024**3)  # Convert to GB
                print(f"GPU 0 memory: {total_memory:.2f} GB")
            except Exception as e:
                print(f"Could not get GPU memory info: {e}")

        # Test CUDA tensor operations
        print("\nTesting CUDA operations...")
        try:
            x = torch.randn(3, 3).cuda()
            y = torch.randn(3, 3).cuda()
            z = x + y
            print("SUCCESS: CUDA tensor operations work correctly")
        except Exception as e:
            print(f"ERROR: CUDA tensor operations failed: {e}")

    else:
        print("WARNING: CUDA is NOT available. Training will use CPU only.")
        print("This means training will be significantly slower than GPU training.")

    # Check PyTorch build configuration
    print("\n=== PyTorch Build Information ===")
    try:
        print(f"CUDA available in build: {torch.backends.cudnn.is_available()}")
        print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    except Exception as e:
        print(f"Could not check cuDNN info: {e}")

if __name__ == "__main__":
    verify_cuda()