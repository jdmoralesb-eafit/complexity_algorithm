import numpy as np

def multiply_by_two(arr):
    """
    O(N)
    Multiply each element by 2.
    Linear complexity.
    """
    return arr * 2


def compute_fft(arr):
    """
    ~O(N log N)
    Perform FFT on input array.
    """
    return np.fft.fft(arr)


def quadratic_loops(n):
    """
    O(N^2)
    Example of double nested loop.
    Returns the number of iterations performed.
    """
    count = 0
    for i in range(n):
        for j in range(n):
            count += 1
    return count


def cubic_loops(n):
    """
    O(N^3)
    Example of triple nested loop.
    Returns the number of iterations performed.
    """
    count = 0
    for i in range(n):
        for j in range(n):
            for k in range(n):
                count += 1
    return count
