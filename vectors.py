from typing import Sequence

import numpy as np
from scipy import sparse


def get_vector(dim: int) -> np.ndarray:
    """Create random column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        np.ndarray: column vector.
    """
    return np.random.rand(dim, 1)


def get_sparse_vector(dim: int, density: float = 0.5) -> sparse.coo_matrix: 
    """Create random sparse column vector with dimension dim.

    Args:
        dim (int): vector dimension.

    Returns:
        sparse.coo_matrix: sparse column vector.
    """
    data = np.random.rand(dim)
    mask = np.random.rand(dim) < density
    data[mask] = 0
    
    row = np.arange(dim)
    col = np.zeros(dim)
    
    return sparse.coo_matrix((data, (row, col)), shape=(dim, 1))


def add(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Vector addition. 

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        np.ndarray: vector sum.
    """
    return x + y


def scalar_multiplication(x: np.ndarray, a: float) -> np.ndarray:
    """Vector multiplication by scalar.

    Args:
        x (np.ndarray): vector.
        a (float): scalar.

    Returns:
        np.ndarray: multiplied vector.
    """
    return a * x 


def linear_combination(vectors: Sequence[np.ndarray], coeffs: Sequence[float]) -> np.ndarray:
    """Linear combination of vectors.

    Args:
        vectors (Sequence[np.ndarray]): list of vectors of len N.
        coeffs (Sequence[float]): list of coefficients of len N.

    Returns:
        np.ndarray: linear combination of vectors.
    """
    if len(vectors) != len(coeffs):
      raise ValueError("The number of vectors does not match the number of coefficients.")
      
    result = np.zeros_like(vectors[0])
    for v, c in zip(vectors, coeffs):
        result += c * v
        
    return result


def dot_product(x: np.ndarray, y: np.ndarray) -> float:
    """Vectors dot product.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: dot product.
    """
    return float(np.dot(x, y))


def norm(x: np.ndarray, order: int | float) -> float:
    """Vector norm: Manhattan, Euclidean or Max.

    Args:
        x (np.ndarray): vector
        order (int | float): norm's order: 1, 2 or inf.

    Returns:
        float: vector norm
    """
    if order == 1:
        #  Manhattan: |x1| + |x2| + ... + |xn|
        return float(np.sum(np.abs(x)))
    
    elif order == 2:
        # Euclidean: sqrt(x1² + x2² + ... + xn²)
        return float(np.sqrt(np.sum(x ** 2)))
    
    elif order == np.inf:
        # Max: max(|x1|, |x2|, ..., |xn|)
        return float(np.max(np.abs(x)))
    
    else:
        raise ValueError("Only norms of order 1, 2, or np.inf are supported")


def distance(x: np.ndarray, y: np.ndarray) -> float:
    """L2 distance between vectors.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.

    Returns:
        float: distance.
    """
    return float(sum((x-y) ** 2)) ** 0.5


def cos_between_vectors(x: np.ndarray, y: np.ndarray) -> float:
    """Cosine between vectors in deg.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.


    Returns:
        np.ndarray: angle in deg.
    """
    
    cos_theta = dot_product(x, y) / (norm(x, 2) * norm(y, 2))
    angle_rad = np.arccos(cos_theta)
    
    return float(np.degrees(angle_rad))

def is_orthogonal(x: np.ndarray, y: np.ndarray) -> bool:
    """Check is vectors orthogonal.

    Args:
        x (np.ndarray): 1th vector.
        y (np.ndarray): 2nd vector.


    Returns:
        bool: are vectors orthogonal.
    """
    return float(np.dot(x, y)) == 0


def solves_linear_systems(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Solve system of linear equations.

    Args:
        a (np.ndarray): coefficient matrix.
        b (np.ndarray): ordinate values.

    Returns:
        np.ndarray: sytems solution
    """
    if a.shape[0] != a.shape[1]:
        raise ValueError("Matrix 'a' must be square")

    if a.shape[0] != b.shape[0]:
        raise ValueError("Matrix 'a' and vector 'b' dimensions do not match")

    det = np.linalg.det(a)
    if np.isclose(det, 0.0):
        raise ValueError("Matrix 'a' is singular — the system has no unique solution")

    a_inv = np.linalg.inv(a)
    x = a_inv.dot(b)

    return x
