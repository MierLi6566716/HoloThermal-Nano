
import os
import sys
import torch
import time
import numpy as np
import torchvision
import gc
from PIL import Image
import torchvision.transforms as T

# 1. CUDA-to-CPU Bridge
torch.cuda.is_available = lambda: False
def to_cpu(self, *args, **kwargs): return self.to("cpu")
torch.Tensor.cuda = to_cpu
torch.nn.Module.cuda = to_cpu
torch.cuda.FloatTensor = torch.FloatTensor

# --- TIMM COMPATIBILITY PATCH ---
import timm
_orig_create_model = timm.create_model
def _patched_create_model(model_name, **kwargs):
    if model_name == 'vit_tiny_patch16_224':
        try:
            from timm.models.vision_transformer import _create_vision_transformer
            pretrained = kwargs.pop('pretrained', False)
            custom_embed = kwargs.pop('embed_dim', 192)
            features_only = kwargs.pop('features_only', False)
            img_size = (256, 144) 
            num_classes = kwargs.pop('num_classes', 0)
            in_chans = kwargs.pop('in_chans', 3)
            global_pool = kwargs.pop('global_pool', '')

            model_kwargs = dict(
                patch_size=16, embed_dim=custom_embed, depth=12, num_heads=3, 
                img_size=img_size, num_classes=num_classes, in_chans=in_chans,
                global_pool=global_pool, **kwargs
            )
            model = _create_vision_transformer('vit_tiny_patch16_224', pretrained=pretrained, **model_kwargs)
            if features_only:
                class VitFeaturesWrapper(torch.nn.Module):
                    def __init__(self, vit_model):
                        super().__init__()
                        self.model = vit_model
                    def forward(self, x):
                        h, w = x.shape[-2], x.shape[-1]
                        x = self.model.forward_features(x)
                        if self.model.cls_token is not None: x = x[:, 1:, :]
                        B, L, D = x.shape
                        feat_h, feat_w = h // 16, w // 16
                        x = x.transpose(1, 2).reshape(B, D, feat_h, feat_w)
                        return [x]
                return VitFeaturesWrapper(model)
            return model
        except Exception: return _orig_create_model(model_name, **kwargs)
    return _orig_create_model(model_name, **kwargs)

timm.create_model = _patched_create_model

# --- CONFIGURATION ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, 'DPRC', 'pretrain_networks', 'model_r.pth') 
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CODES_DIR = os.path.join(PROJECT_ROOT, 'DPRC', 'codes')

if os.path.isdir(CODES_DIR):
    sys.path.insert(0, CODES_DIR)
    if 'utils' in sys.modules: del sys.modules['utils']

try:
    from common import Config
    from agent import get_agent
except ImportError: pass

from pytorch_msssim import ssim
import threading
from queue import Queue

_AGENT_CACHE = {}

def get_cached_agent():
    if 'agent' not in _AGENT_CACHE:
        class Args:
            def __init__(self):
                self.gpu_ids = None; self.proj_dir = "results/running"; self.holo_data_root = "data"
                self.exp_name = "jetson_thermal"; self.dataset = "collected"; self.batch_size = 1
                self.num_workers = 1; self.channel = "r"; self.fake = True; self.prop_dist = 20
                self.pixel_pitch = 6.4; self.pretrain_path = ""; self.ckpt = "latest"; self.compress = True
                self.quality = "high"; self.is_train = False; self.model_name = "stage2"
                self.w_mse = 1.0; self.w_vgg = 0.025; self.w_ssim = 0.05; self.w_wfft = 1e-8; self.nr_epochs = 1
                self.lr = 1e-3; self.lr_s = 5e-5; self.lr_step_size = 5; self.cont = False
                self.vis = False; self.save_frequency = 1; self.val_frequency = 5
                self.vis_frequency = 40; self.output = "results/outputs"; self.postfix = ""

        import argparse
        old_parse = argparse.ArgumentParser.parse_args
        argparse.ArgumentParser.parse_args = lambda self, *args, **kwargs: Args()
        try:
            torch.set_num_threads(1)
            config = Config("test")
            config.device = "cpu"
            agent = get_agent(config)
            if WEIGHTS_PATH and os.path.exists(WEIGHTS_PATH):
                state_dict = torch.load(WEIGHTS_PATH, map_location='cpu')
                if 'model_state_dict' in state_dict: agent.net.load_state_dict(state_dict['model_state_dict'], strict=False)
                elif 'state_dict' in state_dict: agent.net.load_state_dict(state_dict['state_dict'], strict=False)
                else: agent.net.load_state_dict(state_dict, strict=False)
            if hasattr(agent.net, 'hyper_prior'): agent.net.hyper_prior.hyperprior_entropy_model.build_tables()
            warmup_input = torch.randn(1, 1, 144, 256)
            with torch.no_grad(): _ = agent.propagator(warmup_input)
            _AGENT_CACHE['agent'] = agent
        finally: argparse.ArgumentParser.parse_args = old_parse
    return _AGENT_CACHE['agent']

_output_queue = Queue()
_ssim_scores = []
def _background_output_loop():
    try: os.nice(19) 
    except: pass
    while True:
        task = _output_queue.get()
        if task is None: break
        recon_img, base_input, out_path, scheme, i, unit_duration, fps, status, num_threads, os_pid, save_visual = task
        score = ssim(recon_img, base_input, data_range=1.0, size_average=True).item()
        _ssim_scores.append(score)
        if save_visual:
            torchvision.utils.save_image(recon_img, out_path, normalize=True)
            print(f"[{os_pid}] Unit {i}: {unit_duration:.2f}s ({fps:.2f} FPS) | SSIM={score:.4f} | {status} | Threads={num_threads}")
        _output_queue.task_done()

_output_thread = threading.Thread(target=_background_output_loop, daemon=True)
_output_thread.start()

def run_units(units, config_thermal):
    num_threads = int(config_thermal.get('current_threads', 1))
    torch.set_num_threads(num_threads)
    
    active_cores = config_thermal.get('active_cores', [0, 1, 2, 3])
    try: os.sched_setaffinity(0, set(active_cores))
    except: pass

    agent = get_cached_agent()
    agent.net.eval()
    scheme = config_thermal.get("scheme_name", "unknown")
    res = (1, 1, 144, 256) 
    
    input_path = os.path.join(os.path.dirname(SCRIPT_DIR), "input.png")
    if os.path.exists(input_path):
        img = Image.open(input_path).convert('L')
        transform = T.Compose([T.Resize((res[2], res[3])), T.ToTensor()])
        single_input = transform(img).unsqueeze(0)
    else: single_input = torch.randn(res)

    output_dir = "results/outputs"
    if not os.path.exists(output_dir): os.makedirs(output_dir, exist_ok=True)

    deadline_s = float(config_thermal.get("deadline_s", 0))
    global _ssim_scores; _ssim_scores = [] 
    success_count = 0

    print(f"[{os.getpid()}] Running {units} units ({scheme}) [INTRA-OP]... Deadline: {deadline_s}s")

    for i in range(int(units)):
        unit_start = time.perf_counter()
        with torch.no_grad():
            output = agent.net(single_input, False)
            pred_phase = output[1]
            recon_img = agent.propagator(pred_phase)
            recon_img = torch.sqrt(torch.pow(recon_img, 2) * 0.95)
            
            unit_duration = time.perf_counter() - unit_start
            fps = 1.0 / unit_duration if unit_duration > 0 else 0
            
            if deadline_s > 0 and unit_duration > deadline_s:
                status = "FAIL"; noise = (unit_duration - deadline_s) * 0.8
                recon_img = torch.clamp(recon_img + (torch.randn_like(recon_img) * noise), 0, 1)
            else: status = "PASS"; success_count += 1

            save_visual = (i % 20 == 0 or units <= 5)
            out_path = os.path.join(output_dir, f"recon_{scheme}_unit_{i}.png")
            data_for_bg = (recon_img.detach().clone(), single_input, out_path, scheme, i, unit_duration, fps, status, num_threads, os.getpid(), save_visual)
            _output_queue.put(data_for_bg)
            
            output = pred_phase = recon_img = None
                
    if int(units) > 0: _output_queue.join()
    gc.collect()

    avg_ssim = sum(_ssim_scores) / len(_ssim_scores) if _ssim_scores else 0
    print(f"\n[{os.getpid()}] === WORKLOAD SUMMARY ({scheme}) ===")
    print(f"[{os.getpid()}] Average SSIM: {avg_ssim:.4f}")
    print(f"[{os.getpid()}] Success Rate: {(success_count/int(units))*100:.1f}%\n")
