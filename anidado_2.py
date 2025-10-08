import numpy as np

def compute_fft(data, dx=1.0, dy=None, shift=True, normalize=True, return_phase=True):
    """
    Compute 1D or 2D discrete Fourier transform with helpful outputs.

    Parameters
    ----------
    data : array_like
        1D or 2D input signal (real or complex).
    dx : float, optional
        Sample spacing along the first (or only) axis. Default 1.0.
    dy : float or None, optional
        Sample spacing along the second axis (only for 2D). If None, dy == dx.
    shift : bool, optional
        If True, apply fftshift to place zero-frequency at center. Default True.
    normalize : bool, optional
        If True, normalize the transform by the number of samples (so Parseval-friendly). Default True.
    return_phase : bool, optional
        If True, include phase (np.angle) in the result. Default True.

    Returns
    -------
    dict
        For 1D:
            {
              'F': complex numpy array (fft, possibly shifted),
              'magnitude': real numpy array,
              'phase': real numpy array (if return_phase=True),
              'freqs': real numpy array (frequency axis, same shape as F)
            }
        For 2D:
            {
              'F': 2D complex array (fft2, possibly shifted),
              'magnitude': 2D real array,
              'phase': 2D real array (if return_phase=True),
              'fx': 1D array (freqs along x / columns),
              'fy': 1D array (freqs along y / rows)
            }

    Raises
    ------
    ValueError
        If data is not 1D or 2D.
    """
    data = np.asarray(data)
    print("entro a anidado 2")
    if data.ndim == 1:
        n = data.shape[0]
        F = np.fft.fft(data)
        if normalize:
            F = F / n
        if shift:
            F = np.fft.fftshift(F)
            freqs = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
        else:
            freqs = np.fft.fftfreq(n, d=dx)
        mag = np.abs(F)
        out = {'F': F, 'magnitude': mag, 'freqs': freqs}
        if return_phase:
            out['phase'] = np.angle(F)
        return out

    elif data.ndim == 2:
        if dy is None:
            dy = dx
        ny, nx = data.shape  # rows (y), cols (x)
        F = np.fft.fft2(data)
        if normalize:
            F = F / (nx * ny)
        if shift:
            F = np.fft.fftshift(F)
            fx = np.fft.fftshift(np.fft.fftfreq(nx, d=dx))  # columns -> x axis
            fy = np.fft.fftshift(np.fft.fftfreq(ny, d=dy))  # rows -> y axis
        else:
            fx = np.fft.fftfreq(nx, d=dx)
            fy = np.fft.fftfreq(ny, d=dy)
        mag = np.abs(F)
        out = {'F': F, 'magnitude': mag, 'fx': fx, 'fy': fy}
        if return_phase:
            out['phase'] = np.angle(F)
        return out

    else:
        raise ValueError("compute_fft only supports 1D or 2D input arrays.")
