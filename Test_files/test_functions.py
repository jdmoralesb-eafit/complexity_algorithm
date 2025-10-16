import numpy as np

def matrix_addition(matrix):
    """
    O(N^2) para matrices NxN
    Suma una matriz consigo misma más una matriz de unos del mismo tamaño.
    """
    ones_matrix = np.ones_like(matrix)
    return matrix + matrix + ones_matrix

def matrix_multiplication(matrix):
    """
    O(N^3) para matrices NxN (multiplicación ingenua)
    Multiplica una matriz por sí misma.
    """
    return np.dot(matrix, matrix)

def triple_matrix_multiplication(matrix):
    """
    O(N^3) - similar a cubic_loops
    Multiplica una matriz por sí misma tres veces.
    """
    return np.dot(np.dot(matrix, matrix), matrix)

def compute_fft_matrix(matrix):
    """
    ~O(N^2 log N) para FFT 2D
    Perform FFT 2D en la matriz de entrada.
    """
    return np.fft.fft2(matrix)

