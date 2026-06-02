
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
REPO_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CODES_DIR = os.path.join(REPO_ROOT, 'DPRC', 'codes')

# Surgical Namespace Fix
if os.path.isdir(CODES_DIR):
    if CODES_DIR not in sys.path:
        sys.path.insert(0, CODES_DIR)
    if 'utils' in sys.modules:
        del sys.modules['utils']

try:
    from reconstruction import holo_propagator
    import propagation_utils as prop_utils
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

# CUDA-to-CPU Bridge
torch.cuda.is_available = lambda: False
torch.Tensor.cuda = lambda self: self.to("cpu")

def run_units(units, config_thermal):
    """
    Balanced Stress test using Iterative Phase Retrieval.
    Performs 40 FFT iterations per unit. 
    Generates consistent heat without being painfully slow.
    """
    wavelength = 520e-9 
    prop_dist = 0.2     
    feature_size = (6.4e-6, 6.4e-6)
    
    # Initialize the Physics Engine
    propagator = holo_propagator(wavelength, prop_dist, feature_size)
    
    # DYNAMIC RESOLUTION: Use MATRIX_SIZE from config
    m_size = config_thermal.get('matrix_size', 512)
    res = (int(m_size), int(m_size)) 
    
    # Load input.png
    input_path = os.path.join(os.path.dirname(SCRIPT_DIR), "holography", "input.png")
    
    if os.path.exists(input_path):
        img = Image.open(input_path).convert('L')
        transform = T.Compose([
            T.Resize(res),
            T.ToTensor(),
        ])
        target_amp = transform(img).unsqueeze(0)
    else:
        print(f"Warning: {input_path} not found. Using random target.")
        target_amp = torch.rand((1, 1, res[0], res[1]))

    print(f"[{os.getpid()}] STARTING ITERATIVE PHYSICS: {units} units.")
    print(f"[{os.getpid()}] Res: {res[0]}x{res[1]}, 40 iterations/unit.")

    output_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "holography", "results", "iterative_outputs")
    os.makedirs(output_dir, exist_ok=True)

    for u in range(int(units)):
        # Start with random phase
        estimated_phase = (torch.rand((1, 1, res[0], res[1])) - 0.5) * 2 * np.pi
        
        # --- BALANCED ITERATION LOOP (Gerchberg-Saxton) ---
        for i in range(40): 
            # 1. Propagate to reconstruction plane
            with torch.no_grad():
                # We use the internal math for the iteration
                real, imag = prop_utils.polar_to_rect(torch.ones_like(estimated_phase), estimated_phase)
                field = torch.stack((real, imag), -1)
                
                # Forward propagation (SLM -> Recon)
                recon_field = prop_utils.propagate_field(field, propagator.propagator, prop_dist, wavelength, feature_size)
                
                # 2. Replace amplitude with target, keep phase
                recon_field_new = prop_utils.replace_amplitude(recon_field, target_amp)
                
                # 3. Propagate back to SLM plane (Recon -> SLM)
                slm_field_new = prop_utils.propagate_field(recon_field_new, propagator.propagator, -prop_dist, wavelength, feature_size)
                
                # 4. Extract phase for next iteration
                _, estimated_phase = prop_utils.rect_to_polar(slm_field_new[..., 0], slm_field_new[..., 1])

        # After iterations, the quality is much higher
        with torch.no_grad():
            final_recon = propagator(estimated_phase)
            
        # Save EVERY unit now for visibility
        out_path = os.path.join(output_dir, f"iter_unit_{u}_pid_{os.getpid()}.png")
        torchvision.utils.save_image(final_recon, out_path, normalize=True)
        print(f"[{os.getpid()}] Unit {u} complete.")

    print(f"[{os.getpid()}] Completed Heavy Iterative Physics Stress.")
