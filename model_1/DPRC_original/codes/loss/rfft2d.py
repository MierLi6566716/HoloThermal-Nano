import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def my_fft(input, signal_ndim, normalized=False):
    if signal_ndim < 1 or signal_ndim > 3:
        print(
            "Signal ndim out of range, was",
            signal_ndim,
            "but expected a value between 1 and 3, inclusive",
        )
        return

    dims = -1
    if signal_ndim == 2:
        dims = (-2, -1)
    if signal_ndim == 3:
        dims = (-3, -2, -1)

    norm = "backward"
    if normalized:
        norm = "ortho"

    return torch.view_as_real(
        torch.fft.fftn(torch.view_as_complex(input), dim=dims, norm=norm)
    )


def my_ifft(input, signal_ndim, normalized=False):
    if signal_ndim < 1 or signal_ndim > 3:
        print(
            "Signal ndim out of range, was",
            signal_ndim,
            "but expected a value between 1 and 3, inclusive",
        )
        return

    dims = -1
    if signal_ndim == 2:
        dims = (-2, -1)
    if signal_ndim == 3:
        dims = (-3, -2, -1)

    norm = "backward"
    if normalized:
        norm = "ortho"

    return torch.view_as_real(
        torch.fft.ifftn(torch.view_as_complex(input), dim=dims, norm=norm)
    )


def my_rfft(input, signal_ndim, normalized=False):
    if signal_ndim < 1 or signal_ndim > 3:
        print(
            "Signal ndim out of range, was",
            signal_ndim,
            "but expected a value between 1 and 3, inclusive",
        )
        return

    dims = -1
    if signal_ndim == 2:
        dims = (-2, -1)
    if signal_ndim == 3:
        dims = (-3, -2, -1)
    norm = "backward"
    if normalized:
        norm = "ortho"

    return torch.view_as_real(torch.fft.rfftn(input, dim=dims, norm=norm))


def my_irfft(input, signal_ndim, normalized=False, signal_sizes=None):
    if signal_ndim < 1 or signal_ndim > 3:
        print(
            "Signal ndim out of range, was",
            signal_ndim,
            "but expected a value between 1 and 3, inclusive",
        )
        return

    assert signal_ndim == len(signal_sizes), "signal_sizes must have length signal_ndim"

    dims = -1
    if signal_ndim == 2:
        dims = (-2, -1)
    if signal_ndim == 3:
        dims = (-3, -2, -1)

    norm = "backward"
    if normalized:
        norm = "ortho"

    return torch.view_as_real(
        torch.fft.irfftn(
            torch.view_as_complex(input), s=signal_sizes, dim=dims, norm=norm
        )
    )


class Rfft2d(nn.Module):
    """
    Blockwhise 2D FFT
    for fixed blocksize of 8x8
    """

    def __init__(self, blocksize=8, interleaving=False):
        """
        Parameters:
        """
        super().__init__()  # call super constructor

        self.blocksize = blocksize
        self.interleaving = interleaving
        if interleaving:
            self.stride = self.blocksize // 2
        else:
            self.stride = self.blocksize

        self.unfold = torch.nn.Unfold(
            kernel_size=self.blocksize, padding=0, stride=self.stride
        )
        return

    def forward(self, x):
        """
        performs 2D blockwhise DCT

        Parameters:
        x: tensor of dimension (N, 1, h, w)

        Return:
        tensor of dimension (N, k, b, b/2, 2)
        where the 2nd dimension indexes the block. Dimensions 3 and 4 are the block real FFT coefficients.
        The last dimension is pytorches representation of complex values
        """

        (N, C, H, W) = x.shape
        assert C == 1, "FFT is only implemented for a single channel"
        assert H >= self.blocksize, "Input too small for blocksize"
        assert W >= self.blocksize, "Input too small for blocksize"
        assert (H % self.stride == 0) and (
            W % self.stride == 0
        ), "FFT is only for dimensions divisible by the blocksize"

        # unfold to blocks
        x = self.unfold(x)
        # now shape (N, 64, k)
        (N, _, k) = x.shape
        x = x.view(-1, self.blocksize, self.blocksize, k).permute(0, 3, 1, 2)
        # now shape (N, #k, b, b)
        # perform DCT
        coeff = my_rfft(x, signal_ndim=2)

        return coeff / self.blocksize**2

    def inverse(self, coeff, output_shape):
        """
        performs 2D blockwhise inverse rFFT

        Parameters:
        output_shape: Tuple, dimensions of the outpus sample
        """
        if self.interleaving:
            raise Exception(
                "Inverse block FFT is not implemented for interleaving blocks!"
            )

        # perform iRFFT
        # @Mufan Why here needs signal_sizes?
        x = my_irfft(
            coeff, signal_ndim=2, signal_sizes=(self.blocksize, self.blocksize)
        )
        (N, k, _, _) = x.shape
        x = x.permute(0, 2, 3, 1).view(-1, self.blocksize**2, k)
        x = F.fold(
            x,
            output_size=(output_shape[-2], output_shape[-1]),
            kernel_size=self.blocksize,
            padding=0,
            stride=self.blocksize,
        )
        return x * (self.blocksize**2)
