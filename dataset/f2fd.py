import random
import torch
import numpy as np


def get_f2fd_pair(vol, bernoulli_mask_ratio=0.5, bernoulli_mask_patch_size=8, phase_inversion_ratio=0.1, min_mask_radius=0.05, max_mask_radius=0.1):
    vol_fft = np.fft.rfftn(vol)
    vol_fft = np.fft.fftshift(vol_fft, axes=(-3, -2))
    vol_1 = _get_f2fd_vol(
        vol_fft=vol_fft,
        bernoulli_mask_ratio=bernoulli_mask_ratio,
        bernoulli_mask_patch_size=bernoulli_mask_patch_size,
        phase_inversion_ratio=phase_inversion_ratio,
        min_mask_radius=min_mask_radius,
        max_mask_radius=max_mask_radius
    )
    vol_2 = _get_f2fd_vol(
        vol_fft=vol_fft,
        bernoulli_mask_ratio=bernoulli_mask_ratio,
        bernoulli_mask_patch_size=bernoulli_mask_patch_size,
        phase_inversion_ratio=phase_inversion_ratio,
        min_mask_radius=min_mask_radius,
        max_mask_radius=max_mask_radius
    )
    return vol_1, vol_2
    

def _get_f2fd_vol(vol_fft, bernoulli_mask_ratio=0.5, bernoulli_mask_patch_size=8, phase_inversion_ratio=0.1, min_mask_radius=0.05, max_mask_radius=0.1):
    # Patch-based Bernoulli Masking (8x8x8 patches)
    size = vol_fft.shape[0]
    mask_shape = (size // bernoulli_mask_patch_size, size // bernoulli_mask_patch_size, 1 + (size // 2 + 1) // bernoulli_mask_patch_size)
    patch_mask = (np.random.rand(*mask_shape) > bernoulli_mask_ratio).astype(np.float32)
    # Expand mask to match vol_fft shape
    bernoulli_mask = patch_mask.repeat(bernoulli_mask_patch_size, axis=0)
    bernoulli_mask = bernoulli_mask.repeat(bernoulli_mask_patch_size, axis=1)
    bernoulli_mask = bernoulli_mask.repeat(bernoulli_mask_patch_size, axis=2)
    bernoulli_mask_1 = bernoulli_mask[:size, :size, :size // 2 + 1]  # Ensure matching shape
    #bernoulli_mask_2 = 1 - bernoulli_mask_1  # Inverse mask for the second volume
        
    # Random Phase Inversion
    phase_flip = (np.random.rand(*vol_fft.real.shape) < phase_inversion_ratio).astype(np.float32)
    vol_fft_inverted = vol_fft * ((-1) ** phase_flip)
    
    # Spherical Mask to keep low frequencies
    center = size // 2
    radius = random.randint(int(min_mask_radius * size), int(max_mask_radius * size))
    x, y, z = np.meshgrid(torch.arange(size), torch.arange(size), torch.arange(size // 2 + 1), indexing='ij')
    mask = ((x - center) ** 2 + (y - center) ** 2 + (z) ** 2) < (radius ** 2)
    mask = mask.astype(np.float32)
    
    # Combine masks
    overall_mask_1 = np.logical_or(mask, bernoulli_mask_1).astype(np.float32)
    #overall_mask_2 = np.logical_or(mask, bernoulli_mask_2).astype(np.float32)

    vol_1_fft = vol_fft_inverted * overall_mask_1
    #vol_2_fft = vol_fft_inverted * overall_mask_2
    
    vol_1 = np.fft.irfftn(np.fft.ifftshift(vol_1_fft, axes=(-3, -2)), s=vol.shape)
    #vol_2 = np.fft.irfftn(np.fft.ifftshift(vol_2_fft, axes=(-3, -2)), s=vol.shape)    
        
    return vol_1 #, vol_2



