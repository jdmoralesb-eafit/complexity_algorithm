"""
Algoritmos de compensacion de fase de Karina Ortega / Rene Restrepo (SOPHIA, EAFIT)
adaptados para medirse desde Main.py.

Fuentes (sensibilidad_slm/slm/algoritmos karina/):
  - PCA_ChaoZuoPaper.m            PCA (Chao Zuo et al.), Restrepo & Ortega 2025
  - phaseVariationOptimization.m  PVM, Shuo Liu et al., DOI 10.1364/OL.43.001870
  - usage.m                       spatial_filter + generateZernikePolynomials

Uso en Main.py: elegir este archivo, luego la funcion `PCA` o `PVM`. Los
parametros son:
    inp        holograma 2-D (el primer parametro lo llena Main con cada imagen)
    wavelength micrometros (default 0.632)
    dx, dy     pitch del pixel en micrometros (default 3.75)
Cada funcion mide el pipeline completo, igual que CFS/vortexLegendre de
pyDHM_methods.py:  recorte cuadrado -> filtro espacial -> onda de referencia
-> compensacion (PCA o PVM).  Devuelven el campo complejo / la fase compensada.

Archivo autocontenido (Main pega el codigo en un script temporal): solo
numpy y scipy.
"""

import numpy as np
from scipy.optimize import minimize


# ─────────────────────────────────────────────
# Pre-compensacion comun (spatial_filter + onda de referencia)
# ─────────────────────────────────────────────
def square_crop(inp):
    """Recorta columnas para dejar el holograma cuadrado (como el .m)."""
    holo = np.asarray(inp, dtype=np.float64)
    if holo.ndim == 3:
        holo = holo[:, :, 0]
    n_rows, n_cols = holo.shape
    if n_cols > n_rows:
        diff = n_cols - n_rows
        lim = diff // 2
        holo = holo[:, lim:n_cols - (diff - lim)]
    elif n_rows > n_cols:
        diff = n_rows - n_cols
        lim = diff // 2
        holo = holo[lim:n_rows - (diff - lim), :]
    return holo


def spatial_filter(holo, factor=3):
    """
    Filtro espacial del orden +1 (usage.m / PCA_ChaoZuoPaper.m), vectorizado.

    Returns holo_filtered (complejo), fx_max, fy_max (1-based como MATLAB).
    """
    N, M = holo.shape
    ft_holo = np.fft.fftshift(np.fft.fft2(np.fft.fftshift(holo)))
    ft_holo[:5, :5] = 0                                   # ft_holo(1:5,1:5)=0

    # filtrar DC: mask1(N/2-20:N/2+20, M/2-20:M/2+20)=0  (1-based inclusivo)
    ft_holo_I = ft_holo.copy()
    ft_holo_I[N // 2 - 21:N // 2 + 20, M // 2 - 21:M // 2 + 20] = 0
    ft_holo_I[0, 0] = 0                                   # mask1(1,1)=0

    # buscar el maximo en la mitad izquierda
    region = np.abs(ft_holo_I[:, :M // 2])
    iy, ix = np.unravel_index(np.argmax(region), region.shape)
    fy_max, fx_max = iy + 1, ix + 1                       # 1-based

    # mascara circular alrededor del maximo
    resc = np.sqrt((fx_max - N / 2) ** 2 + (fy_max - M / 2) ** 2) / factor
    r = np.arange(1, N + 1)[:, None]
    p = np.arange(1, M + 1)[None, :]
    cir_mask = ((r - fy_max) ** 2 + (p - fx_max) ** 2) <= resc ** 2

    holo_filtered = np.fft.fftshift(np.fft.ifft2(np.fft.fftshift(ft_holo * cir_mask)))
    return holo_filtered, fx_max, fy_max


def reference_wave(M, N, wavelength, dxy, fx_max, fy_max):
    """Onda de referencia que centra el orden +1 (usage.m)."""
    k = 2 * np.pi / wavelength
    fx_0, fy_0 = M / 2, N / 2
    theta_x = np.arcsin((fx_0 - fx_max) * wavelength / (M * dxy))
    theta_y = np.arcsin((fy_0 - fy_max) * wavelength / (N * dxy))
    n, m = np.meshgrid(np.arange(-M // 2, M // 2), np.arange(-N // 2, N // 2))
    return np.exp(1j * k * (np.sin(theta_x) * n * dxy + np.sin(theta_y) * m * dxy))


def _precompensate(inp, wavelength, dx, dy):
    holo = square_crop(inp)
    N, M = holo.shape
    dxy = 0.5 * (dx + dy)
    holo_filtered, fx_max, fy_max = spatial_filter(holo, factor=3)
    return holo_filtered * reference_wave(M, N, wavelength, dxy, fx_max, fy_max)


# ─────────────────────────────────────────────
# PCA  (PCA_ChaoZuoPaper.m)
# ─────────────────────────────────────────────
def pca_compensation(psi, limit=16):
    """
    Compensacion por PCA. `limit` = semiancho del recorte central del espectro
    (16 -> region de 32x32, 32 -> 64x64; en el .m: limit_mat = M/2 - limit).

    Pasos: FFT -> recorte -> IFFT (square) -> SVD, 1 componente -> fase ->
    polyfit grado 2 en X y en Y (coordenadas 1-based) -> square*exp(-i*fase)
    -> FFT -> relleno con ceros a MxN -> IFFT.
    """
    M, N = psi.shape
    fft_field = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(psi)))

    # MATLAB (1-based): limit_mat : end-limit_mat-1 con limit_mat = M/2 - limit
    half = int(max(1, min(limit, M // 2 - 1, N // 2 - 1)))
    r0, c0 = M // 2 - half - 1, N // 2 - half - 1
    fft_crop = fft_field[r0:r0 + 2 * half, c0:c0 + 2 * half]
    square = np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(fft_crop)))

    # SVD: componente dominante (numComponentes = 1)
    U, s, Vh = np.linalg.svd(square, full_matrices=False)
    dominant_phase = np.angle(s[0] * np.outer(U[:, 0], Vh[0, :]))

    ny, nx = dominant_phase.shape
    Y, X = np.meshgrid(np.arange(1, ny + 1), np.arange(1, nx + 1), indexing='ij')
    unwrapped = np.unwrap(dominant_phase.ravel())
    cx = np.polyfit(X.ravel(), unwrapped, 2)
    cy = np.polyfit(Y.ravel(), unwrapped, 2)
    phase_correction = (cx[0] * X ** 2 + cx[1] * X + cx[2] +
                        cy[0] * Y ** 2 + cy[1] * Y + cy[2])

    corrected = square * np.exp(-1j * phase_correction)
    sub_corrected = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(corrected)))

    oversampled = np.zeros((M, N), dtype=complex)
    oversampled[r0:r0 + 2 * half, c0:c0 + 2 * half] = sub_corrected
    return np.fft.ifftshift(np.fft.ifft2(np.fft.fftshift(oversampled)))


def PCA(inp, wavelength=0.632, dx=3.75, dy=3.75, limit=16):
    """Pipeline completo de PCA (Karina Ortega / Rene Restrepo)."""
    psi = _precompensate(inp, wavelength, dx, dy)
    return pca_compensation(psi, limit)


# ─────────────────────────────────────────────
# PVM  (phaseVariationOptimization.m + usage.m)
# ─────────────────────────────────────────────
def wrap_to_pi(p):
    return (np.asarray(p) + np.pi) % (2 * np.pi) - np.pi


def generate_zernike_polynomials(Nx, Ny, n_max):
    """Base R^k cos(k*theta), k=1..n_max (generateZernikePolynomials)."""
    X, Y = np.meshgrid(np.linspace(-1, 1, Nx), np.linspace(-1, 1, Ny))
    R = np.sqrt(X ** 2 + Y ** 2)
    Theta = np.arctan2(Y, X)
    return [(R ** k) * np.cos(k * Theta) for k in range(1, n_max + 1)]


def total_variation_cost(a, phi_meas, zlist):
    P = np.zeros_like(phi_meas)
    for k in range(len(a)):
        P = P + a[k] * zlist[k]
    phi_c = wrap_to_pi(phi_meas - P)
    dx = wrap_to_pi(np.concatenate([phi_c[:, 1:], phi_c[:, -1:]], axis=1) - phi_c)
    dy = wrap_to_pi(np.concatenate([phi_c[1:, :], phi_c[-1:, :]], axis=0) - phi_c)
    return float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))


def phase_variation_optimization(psi, zlist, a0=None, maxiter=300):
    """Minimiza la variacion total (BFGS, equivalente a fminunc quasi-newton)."""
    if a0 is None:
        a0 = np.zeros(len(zlist))
    phi_meas = np.angle(psi)
    res = minimize(total_variation_cost, a0, args=(phi_meas, zlist),
                   method='BFGS', options={'maxiter': maxiter, 'gtol': 1e-6})
    P_opt = np.zeros_like(phi_meas)
    for k in range(len(res.x)):
        P_opt = P_opt + res.x[k] * zlist[k]
    return wrap_to_pi(phi_meas - P_opt), res.x


def PVM(inp, wavelength=0.632, dx=3.75, dy=3.75, n_zernike=6):
    """Pipeline completo de PVM (Liu et al.), devuelve la fase corregida."""
    psi = _precompensate(inp, wavelength, dx, dy)
    Ny, Nx = psi.shape
    zlist = generate_zernike_polynomials(Nx, Ny, n_zernike)
    phi_corrected, _ = phase_variation_optimization(psi, zlist, np.zeros(n_zernike))
    return phi_corrected
