
import os
import torch
import torchvision.transforms as T
from PIL import Image
from pytorch_msssim import ssim

def calculate_ssim(img1_path, img2_path):
    if not os.path.exists(img1_path) or not os.path.exists(img2_path):
        print(f"Error: One of the paths does not exist: {img1_path} or {img2_path}")
        return None

    img1 = Image.open(img1_path).convert('L')
    img2 = Image.open(img2_path).convert('L')

    transform = T.ToTensor()
    t1 = transform(img1).unsqueeze(0)
    t2 = transform(img2).unsqueeze(0)

    # Resize t2 to match t1 if they differ
    if t1.shape != t2.shape:
        t2 = torch.nn.functional.interpolate(t2, size=t1.shape[2:], mode='bicubic', align_corners=False)

    score = ssim(t1, t2, data_range=1.0, size_average=True)
    return score.item()

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python3 calc_ssim.py <ground_truth_path> <reconstruction_path>")
    else:
        gt = sys.argv[1]
        recon = sys.argv[2]
        score = calculate_ssim(gt, recon)
        if score is not None:
            print(f"SSIM Score: {score:.4f}")
