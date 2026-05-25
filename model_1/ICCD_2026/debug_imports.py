
import sys
import os

print("Testing imports...")
try:
    import numpy
    print("numpy imported successfully")
except Exception as e:
    print(f"numpy failed: {e}")

try:
    import torch
    print(f"torch imported successfully (Version: {torch.__version__})")
except Exception as e:
    print(f"torch failed: {e}")

try:
    import torchvision
    print("torchvision imported successfully")
except Exception as e:
    print(f"torchvision failed: {e}")

try:
    import timm
    print("timm imported successfully")
except Exception as e:
    print(f"timm failed: {e}")

print("All basic imports tested.")
