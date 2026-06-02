
import os
import sys
import torch

# Add 'codes' to path so we can import modules
# The codes folder is now inside DPRC/
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Testing/holography -> Testing -> root
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
CODES_DIR = os.path.join(PROJECT_ROOT, 'DPRC', 'codes')
sys.path.append(CODES_DIR)

from common import Config
from agent import get_agent

def main():
    # Mocking arguments for Config
    class Args:
        def __init__(self):
            self.gpu_ids = None
            self.proj_dir = "results/running"
            self.holo_data_root = "data"
            self.exp_name = "test_cpu"
            self.dataset = "DIV2K"
            self.batch_size = 1
            self.num_workers = 1
            self.channel = "r"
            self.fake = True
            self.prop_dist = 20
            self.pixel_pitch = 6.4
            self.pretrain_path = ""
            self.ckpt = "latest"
            self.compress = True
            self.w_mse = 1.0
            self.w_vgg = 0.025
            self.w_ssim = 0.05
            self.w_wfft = 1e-8
            self.quality = "high"
            self.nr_epochs = 1
            self.lr = 1e-3
            self.lr_s = 5e-5
            self.lr_step_size = 5
            self.cont = False
            self.vis = False
            self.save_frequency = 1
            self.val_frequency = 5
            self.vis_frequency = 40
            self.model_name = "stage2"
            self.output = "results/outputs"
            self.postfix = ""

    # Monkey patch argparse.ArgumentParser.parse_args
    import argparse
    def mock_parse_args(self, args=None, namespace=None):
        return Args()
    argparse.ArgumentParser.parse_args = mock_parse_args

    print("Initializing config...")
    config = Config("test")
    config.device = "cpu" # Force CPU
    
    print("Initializing agent...")
    try:
        tr_agent = get_agent(config)
        print("Agent initialized successfully!")
        
        # Load weights if available
        # Example: model_path = os.path.join(PROJECT_ROOT, "DPRC", "running", "DPRC", "model_stage2", "ckpt_epoch30.pth")
        model_path = "" # SET THIS to see real results
        if model_path and os.path.exists(model_path):
            print(f"Loading weights from {model_path}...")
            tr_agent.load_ckpt(load_path=model_path)

        # Run a single test image
        print("Running reconstruction...")
        from PIL import Image
        import torchvision.transforms as T
        import torchvision
        
        input_path = os.path.join(SCRIPT_DIR, "input.png")
        res = (1, 1, 448, 448) # Matching the reduced res
        
        if os.path.exists(input_path):
            img = Image.open(input_path).convert('L')
            transform = T.Compose([
                T.Resize((res[2], res[3])),
                T.ToTensor(),
            ])
            input_data = transform(img).unsqueeze(0)
            print(f"Loaded input from {input_path}")
        else:
            input_data = torch.randn(res)
            print("Using random noise as input")

        with torch.no_grad():
            output, _ = tr_agent.forward([input_data, torch.ones_like(input_data)])
            recon = output[-1]
            out_path = os.path.join(SCRIPT_DIR, "results", "outputs", "test_recon.png")
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            torchvision.utils.save_image(recon, out_path, normalize=True)
            print(f"Reconstruction saved to {out_path}")

    except Exception as e:
        print(f"Error initializing agent: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
