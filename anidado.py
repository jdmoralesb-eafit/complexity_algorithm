from anidado_1 import invert_and_fft
import numpy as np
from PIL import Image

def modified_invert_and_fft(img_array, save_prefix="output"):
    """
    Wrapper around invert_and_fft that modifies the image
    (multiplies by 2 before FFT).

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

    # Multiply by 2 (simple modification)
    modified_array = img_array * 2
    modified_array = np.clip(modified_array, 0, 255).astype(np.uint8)

    # Save modified image (optional)
    Image.fromarray(modified_array).save(f"{save_prefix}_modified.png")

    # Call the existing invert_and_fft but now on modified image
    # To reuse invert_and_fft, we need to temporarily save the modified image
    temp_path = f"{save_prefix}_temp.png"
    Image.fromarray(modified_array).save(temp_path)

    return invert_and_fft(modified_array, save_prefix=save_prefix + "_final")
