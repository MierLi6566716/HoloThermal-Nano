import torch
import torch.nn as nn
import timm

x = torch.randn([1, 2, 1344, 1792]).cuda()
model = timm.create_model('vit_tiny_patch16_224', pretrained=False, num_classes=0, global_pool="",
                          img_size=(1344, 1792), features_only=True, in_chans=2).cuda()
y = model(x)
for fe in y:
    print(fe.shape)