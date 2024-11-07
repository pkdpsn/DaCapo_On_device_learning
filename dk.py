import psutil
import os

# Measure memory before importing PyTorch
process = psutil.Process(os.getpid())
mem_before = process.memory_info().rss / 1024 ** 2  # in MB
print(f"Memory before importing PyTorch: {mem_before:.2f} MB")

# Import PyTorch
import torch

# Measure memory after importing PyTorch
mem_after = process.memory_info().rss / 1024 ** 2  # in MB
print(f"Memory after importing PyTorch: {mem_after:.2f} MB")

# Calculate memory used by PyTorch itself
memory_used = mem_after - mem_before
print(f"Memory used by PyTorch core library: {memory_used:.2f} MB")
