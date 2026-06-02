
import os
import sys
import torch
import time
import numpy as np
import torchvision

# 1. CUDA-to-CPU Bridge: Monkey-patching to redirect CUDA calls to CPU
torch.cuda.is_available = lambda: False
def to_cpu(self, *args, **kwargs):
    return self.to("cpu")

torch.Tensor.cuda = to_cpu
torch.nn.Module.cuda = to_cpu
torch.cuda.FloatTensor = torch.FloatTensor

# --- TIMM COMPATIBILITY PATCH ---
# We globally patch timm.create_model to catch the 'embed_dim' collision
import timm
_orig_create_model = timm.create_model

def _patched_create_model(model_name, **kwargs):
    if model_name == 'vit_tiny_patch16_224':
        try:
            from timm.models.vision_transformer import _create_vision_transformer
            pretrained = kwargs.pop('pretrained', False)
            custom_embed = kwargs.pop('embed_dim', 192)
            features_only = kwargs.pop('features_only', False)
            
            # FORCE RESOLUTION: Ignore the 1344x1792 request and force 448x448
            # This makes the model much smaller in memory.
            img_size = (448, 448)
            kwargs.pop('img_size', None) 
            
            num_classes = kwargs.pop('num_classes', 0)
            in_chans = kwargs.pop('in_chans', 3)
            global_pool = kwargs.pop('global_pool', '')

            model_kwargs = dict(
                patch_size=16, 
                embed_dim=custom_embed, 
                depth=12, 
                num_heads=3, 
                img_size=img_size,
                num_classes=num_classes,
                in_chans=in_chans,
                global_pool=global_pool,
                **kwargs
            )
            model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
            
            if features_only:
                # Wrap the model to simulate features_only behavior for ViT
                class VitFeaturesWrapper(torch.nn.Module):
                    def __init__(self, vit_model):
                        super().__init__()
                        self.model = vit_model
                    def forward(self, x):
                        # Capture input resolution for reshaping
                        h, w = x.shape[-2], x.shape[-1]
                        # forward_features returns (B, 1 + L, D) where 1 is CLS token
                        x = self.model.forward_features(x)
                        # Remove CLS token and reshape back to spatial (B, D, H/16, W/16)
                        if self.model.cls_token is not None:
                            x = x[:, 1:, :]
                        B, L, D = x.shape
                        feat_h, feat_w = h // 16, w // 16
                        x = x.transpose(1, 2).reshape(B, D, feat_h, feat_w)
                        return [x] # Return as list so [-1] index works
                return VitFeaturesWrapper(model)
            return model
        except Exception as e:
            print(f"DEBUG: Patch failed, falling back to original: {e}")
            return _orig_create_model(model_name, **kwargs)
    return _orig_create_model(model_name, **kwargs)

# Apply patch to both possible entry points
timm.create_model = _patched_create_model
if hasattr(timm.models, 'create_model'):
    timm.models.create_model = _patched_create_model
# --------------------------------

# --- CONFIGURATION ---
# Path to your trained checkpoint file (e.g., 'model_stage2/ckpt_epoch30.pth')
# Update this to the actual location on your Jetson
WEIGHTS_PATH = "" 
# ---------------------

# 2. Add DPRC codes to path
# Robust path discovery: search upwards for the 'codes' directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Current path is model_1/Testing/stress_tests/
# We need to go up two levels to reach the project root, then into DPRC/codes
DPRC_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CODES_DIR = os.path.join(DPRC_ROOT, 'DPRC', 'codes')

# Surgical Namespace Fix:
# Both model_1 and codes have a folder named 'utils'. We must force Python
# to look at codes/utils instead of model_1/utils during this stress test.
if os.path.isdir(CODES_DIR):
    # Insert at the VERY BEGINNING to override existing paths
    sys.path.insert(0, CODES_DIR)
    # Clear any cached 'utils' module from model_1 so it can be re-imported from codes
    if 'utils' in sys.modules:
        del sys.modules['utils']
else:
    # Backup: try to find it if structure is slightly different
    possible_codes = os.path.abspath(os.path.join(SCRIPT_DIR, '..', '..', 'DPRC', 'codes'))
    if os.path.isdir(possible_codes):
        sys.path.insert(0, possible_codes)
        CODES_DIR = possible_codes

# We need to import these after monkey patching and path priority setup
try:
    from common import Config
    from agent import get_agent
except ImportError as e:
    # Fallback in case pathing is still tricky in some environments
    print(f"Pathing debug: SCRIPT_DIR={SCRIPT_DIR}, CODES_DIR={CODES_DIR}")
    raise e

_AGENT_CACHE = {}

def get_cached_agent():
    """Lazy-load the model to avoid re-initializing every unit."""
    if 'agent' not in _AGENT_CACHE:
        print("Initializing DPRC Agent on CPU (Stress Test Mode)...")
        
        # Mocking minimal args for DPRC Config
        class Args:
            def __init__(self):
                self.gpu_ids = None
                self.proj_dir = "holography/results/running"
                self.holo_data_root = "holography/data"
                self.exp_name = "jetson_thermal"
                self.dataset = "collected" if os.path.exists("holography/data/collected") else "fake"
                self.batch_size = 1
                self.num_workers = 1
                self.channel = "r"
                self.fake = not os.path.exists("holography/data/collected")
                self.prop_dist = 20
                self.pixel_pitch = 6.4
                self.pretrain_path = ""
                self.ckpt = "latest"
                self.compress = True
                self.quality = "high"
                self.is_train = False
                self.model_name = "stage2"
                self.w_mse = 1.0
                self.w_vgg = 0.025
                self.w_ssim = 0.05
                self.w_wfft = 1e-8
                self.nr_epochs = 1
                self.lr = 1e-3
                self.lr_s = 5e-5
                self.lr_step_size = 5
                self.cont = False
                self.vis = False
                self.save_frequency = 1
                self.val_frequency = 5
                self.vis_frequency = 40
                self.output = "holography/results/outputs"

                self.postfix = ""

        import argparse
        # Monkey patch argparse to return our fake Args
        old_parse = argparse.ArgumentParser.parse_args
        argparse.ArgumentParser.parse_args = lambda self, *args, **kwargs: Args()
        
        try:
            config = Config("test")
            config.device = "cpu"
            agent = get_agent(config)
            
            # --- ACTUALLY LOAD WEIGHTS ---
            if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
                print(f"Loading weights from {WEIGHTS_PATH}...")
                # The agent has a built-in load_ckpt method
                agent.load_ckpt(load_path=WEIGHTS_PATH)
            else:
                print("WARNING: No weights loaded. Output will be noise/lines.")
            
            # BUILD TABLES: Required for compression modules to have 'CDF' attribute
            if hasattr(agent.net, 'hyper_prior'):
                print("Building compression tables (this may take a minute)...")
                agent.net.hyper_prior.hyperprior_entropy_model.build_tables()
                
            _AGENT_CACHE['agent'] = agent
        finally:
            # Restore original parser
            argparse.ArgumentParser.parse_args = old_parse
        
        # Create output dir
        if not os.path.exists("holography/results/outputs"):
            os.makedirs("holography/results/outputs")
            
    return _AGENT_CACHE['agent']

def run_units(units, config_thermal):
    """Entry point for model_1 framework."""
    import gc
    from PIL import Image
    import torchvision.transforms as T
    
    agent = get_cached_agent()
    agent.net.eval()
    
    # REDUCED RESOLUTION: 448x448 is much safer for 4GB RAM
    # than the original 1344x1792.
    res = (1, 1, 448, 448)
    
    # Try to load input.png
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    # Check in holography folder
    input_path = os.path.join(os.path.dirname(SCRIPT_DIR), "holography", "input.png")
    
    if os.path.exists(input_path):
        print(f"[{os.getpid()}] Using custom input: {input_path}")
        # Load, convert to grayscale, resize, and convert to tensor
        img = Image.open(input_path).convert('L')
        transform = T.Compose([
            T.Resize((res[2], res[3])),
            T.ToTensor(),
        ])
        base_input = transform(img).unsqueeze(0) # (1, 1, 448, 448)
    else:
        print(f"[{os.getpid()}] No 'input.png' found at {input_path}. Using random noise.")
        base_input = torch.randn(res)

    print(f"[{os.getpid()}] Running {units} units at {res[2]}x{res[3]}...")
    
    for i in range(int(units)):
        # Clone to avoid modifying the base in-place if needed
        input_data = base_input.clone()
        
        with torch.no_grad():
            # In DPRC, target_amp is the input
            output, _ = agent.forward([input_data, torch.ones_like(input_data)])
            
            if i % 5 == 0:
                recon = output[-1]
                out_path = f"holography/results/outputs/unit_{i}_pid_{os.getpid()}.png"
                torchvision.utils.save_image(recon, out_path, normalize=True)
                print(f"[{os.getpid()}] Saved Reconstruction for unit {i}.")
        
        # Explicitly clear memory after each unit
        del input_data
        del output
        gc.collect()
                
    print(f"[{os.getpid()}] Completed.")
