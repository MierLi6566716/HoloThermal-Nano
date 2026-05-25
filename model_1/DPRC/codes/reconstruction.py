import numpy as np
import propagation_utils as utils
import torch
import torch.nn.functional as F
from propagation_ASM import propagation_ASM


def bgr2rgb(input_img):
    return input_img[:, :, ::-1]


class holo_propagator(torch.nn.Module):
    """the class used for calculate the wave propagation"""

    def __init__(
        self, wavelength, prop_dist, feature_size=(6.4e-6, 6.4e-6), precomped_H=None
    ):
        super(holo_propagator, self).__init__()
        self.precomped_H = (
            precomped_H  # precomputed H matrix in the ASM propagation formula
        )
        self.wavelength = wavelength  # the wavelength that will be used during the diffraction calculation
        self.prop_dist = prop_dist  # propagation distance (here we give it in meters)
        self.feature_size = feature_size  # the pixel pitch size (in meters)
        self.propagator = propagation_ASM  # the function used for calculating the wavefield after propagation

    def forward(self, input_phase):
        slm_phase = input_phase
        real, imag = utils.polar_to_rect(
            torch.ones_like(slm_phase), slm_phase
        )  # transform amp-phase representation to real-image representation
        slm_field = torch.stack(
            (real, imag), -1
        )  # since the codes are based on pytorch 1.6, the complex tensor is represented as [B, C, H, W, 2]
        if self.precomped_H is None:
            self.precomped_H = self.propagator(
                slm_field,
                self.feature_size,
                self.wavelength,
                self.prop_dist,
                return_H=True,
            )
            self.precomped_H = self.precomped_H.to(input_phase).detach()
            self.precomped_H.requires_grad = False

        recon_field = utils.propagate_field(
            slm_field,
            self.propagator,
            self.prop_dist,
            self.wavelength,
            self.feature_size,
            "ASM",
            dtype=torch.float32,
            precomputed_H=self.precomped_H,
        )

        # get the amplitude map from the propagated wavefield
        recon_amp_c, _ = utils.rect_to_polar(recon_field[..., 0], recon_field[..., 1])
        return recon_amp_c


def vis_img(x):
    x = x.permute(1, 0, 2).permute(0, 2, 1).contiguous()
    x = x.numpy()
    x = x.astype(np.uint8)
    return x
