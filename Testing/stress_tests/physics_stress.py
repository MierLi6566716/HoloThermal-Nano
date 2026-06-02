
import os
import sys
import torch
import time
import numpy as np
import torchvision
from PIL import Image
import torchvision.transforms as T

# 1. Pathing to DPRC codes
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Testing/stress_tests -> Testing -> root
# Note: The user has a nested structure /home/mier/Desktop/2026/model_1/model_1/
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CODES_DIR = os.path.join(REPO_ROOT, 'DPRC', 'codes')

# Surgical Namespace Fix:
# Both 'Testing' and 'DPRC/codes' have a folder named 'utils'.
# We MUST clear the 'utils' cache so that DPRC can import its own 'utils.utils'.
if os.path.isdir(CODES_DIR):
    if CODES_DIR not in sys.path:
        sys.path.insert(0, CODES_DIR)
    
    # Force Python to reload 'utils' from the DPRC directory
    if 'utils' in sys.modules:
        del sys.modules['utils']
else:
    print(f"Error: CODES_DIR not found at {CODES_DIR}")
    sys.exit(1)

try:
    from reconstruction import holo_propagator
except ImportError as e:
    print(f"Import Error: {e}")
    # Show the actual files in CODES_DIR to help debugging
    if os.path.exists(CODES_DIR):
        print(f"Files in CODES_DIR: {os.listdir(CODES_DIR)[:5]}")
    sys.exit(1)

# CUDA-to-CPU Bridge
torch.cuda.is_available = lambda: False
torch.Tensor.cuda = lambda self: self.to("cpu")

def run_units(units, config_thermal):
    """
    Stress test using pure ASM Physics (FFTs).
    Generates high CPU load by calculating wave diffraction.
    """
    # Setup physical parameters
    wavelength = 520e-9  # Green laser
    prop_dist = 0.2      # 20 cm
    feature_size = (6.4e-6, 6.4e-6) # 6.4 um pixel pitch
    
    # Initialize the Physics Engine
    propagator = holo_propagator(wavelength, prop_dist, feature_size)
    
    # Load input.png as the source
    input_path = os.path.join(os.path.dirname(SCRIPT_DIR), "holography", "input.png")
    res = (448, 448) # Standard resolution for stress testing
    
    if os.path.exists(input_path):
        img = Image.open(input_path).convert('L')
        transform = T.Compose([
            T.Resize(res),
            T.ToTensor(),
        ])
        # In this physics test, we treat the image as a phase map
        # Map [0, 1] to [-pi, pi]
        phase_data = (transform(img).unsqueeze(0) - 0.5) * 2 * np.pi
    else:
        print(f"Warning: {input_path} not found. Using random phase.")
        phase_data = (torch.rand((1, 1, res[0], res[1])) - 0.5) * 2 * np.pi

    print(f"[{os.getpid()}] Starting Physics Stress Test: {units} units of ASM Propagation.")

    # Create output dir
    output_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "holography", "results", "physics_outputs")
    os.makedirs(output_dir, exist_ok=True)

    start_time = time.time()
    for i in range(int(units)):
        # The ASM forward pass is the "stress" part (FFTs happen here)
        with torch.no_grad():
            recon_amp = propagator(phase_data)
        
        # Save a sample every 10 units to verify it's working
        if i % 10 == 0:
            out_path = os.path.join(output_dir, f"phys_unit_{i}_pid_{os.getpid()}.png")
            torchvision.utils.save_image(recon_amp, out_path, normalize=True)
            
    end_time = time.time()
    print(f"[{os.getpid()}] Physics Stress Test Complete. Avg time/unit: {(end_time-start_time)/units:.4f}s")
