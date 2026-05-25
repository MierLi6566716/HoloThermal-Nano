from transformer_networks.vision_transformer import SwinUnet
import torch

swin_unet = SwinUnet().cuda()

x = torch.randn((4, 1, 1344, 1792)).cuda()
y = swin_unet(x)
print(y.shape)