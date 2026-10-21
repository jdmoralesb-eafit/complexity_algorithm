import numpy as np


def process_nm(hologram):
    """
    O(N*M) — lineal en total de pixeles.
    Operaciones elemento a elemento.
    """
    img = hologram.astype(np.float64)
    result = np.sin(img) + np.cos(img)
    result += img * 2.5 - np.mean(img)
    result = np.sqrt(np.abs(result) + 1.0)
    result = np.log1p(np.abs(result))
    return result


def process_nm_logn(hologram):
    """
    O(N*M * log(N*M)) — FFT 2D + autocorrelacion + cepstrum.
    """
    img = hologram.astype(np.float64)
    F = np.fft.fft2(img)
    power = F * np.conj(F)
    autocorr = np.fft.ifft2(power).real
    log_spectrum = np.log1p(np.abs(F))
    cepstrum = np.fft.ifft2(log_spectrum).real
    return autocorr + cepstrum


def process_nm2(hologram):
    """
    O(N * M^2) — Matriz de Gram (M x M) via producto img.T @ img.
    Para imagenes cuadradas (N=M), equivale a O(N^3) = O(P^1.5).
    """
    img = hologram.astype(np.float64)
    gram = img.T @ img
    eigenvalues = np.linalg.eigvalsh(gram)
    return np.sum(eigenvalues)


def process_nm3(hologram):
    """
    O(N * M^3) — Para cada fila, multiplica la Gram matrix (M x M)
    con el outer product de la fila. N iteraciones de matmul M x M.
    Para imagenes cuadradas (N=M), equivale a O(N^4) = O(P^2).
    """
    img = hologram.astype(np.float64)
    N, M = img.shape
    gram = img.T @ img
    total = 0.0
    for i in range(N):
        row_outer = np.outer(img[i], img[i])
        product = gram @ row_outer
        total += np.trace(product)
    return total
