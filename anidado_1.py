from anidado_2 import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def invert_and_fft(img_array, save_prefix="output"):
    """
    Invert an image (negative) and compute its Fourier transform.

    Parameters
    ----------
    image_path : str
        Path to input image.
    save_prefix : str
        Prefix for saving output images.

    Returns
    -------
    dict : results of FFT
    """

    # Invert image (negative)
    inverted = 255 - img_array

    # Compute FFT
    fft_result = compute_fft(inverted)

    # Save images
    Image.fromarray(inverted.astype(np.uint8)).save(f"{save_prefix}_inverted.png")
    plt.imsave(f"{save_prefix}_fft_magnitude.png", np.log1p(fft_result["magnitude"]), cmap="gray")
    plt.imsave(f"{save_prefix}_fft_phase.png", fft_result["phase"], cmap="twilight")
    print("entro a anidado 1")
    return fft_result

