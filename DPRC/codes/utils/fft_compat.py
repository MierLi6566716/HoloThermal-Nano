import torch

def fft_compat(input, signal_ndim, normalized=False):
    """
    Compatibility wrapper for torch.fft (old function) vs torch.fft module (new).
    input: tensor with last dimension size 2 (real, imag)
    signal_ndim: 2 for 2D FFT
    normalized: boolean for ortho normalization
    """
    if callable(torch.fft):
        return torch.fft(input, signal_ndim, normalized)
    else:
        import torch.fft as fft_mod
        # input is [..., 2]
        # In newer torch, we use complex tensors
        input_complex = torch.view_as_complex(input)
        dims = tuple(range(-signal_ndim, 0))
        norm = "ortho" if normalized else None
        output_complex = fft_mod.fftn(input_complex, dim=dims, norm=norm)
        return torch.view_as_real(output_complex)

def ifft_compat(input, signal_ndim, normalized=False):
    """
    Compatibility wrapper for torch.ifft (old function) vs torch.fft.ifftn (new).
    """
    if hasattr(torch, 'ifft'):
        return torch.ifft(input, signal_ndim, normalized)
    else:
        import torch.fft as fft_mod
        input_complex = torch.view_as_complex(input)
        dims = tuple(range(-signal_ndim, 0))
        norm = "ortho" if normalized else None
        output_complex = fft_mod.ifftn(input_complex, dim=dims, norm=norm)
        return torch.view_as_real(output_complex)
