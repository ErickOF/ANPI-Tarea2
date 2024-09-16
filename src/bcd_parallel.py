import functools
import multiprocessing.managers
import multiprocessing.pool
import matplotlib.pyplot as plt
import multiprocessing
import numpy as np
import psutil
import sympy as sp

from multiprocessing.managers import ListProxy, SyncManager
from scipy.sparse import diags
from sympy.tensor.array import derive_by_array
from sympy.tensor.array.dense_ndim_array import ImmutableDenseNDimArray
from typing import Any, Dict, List, Tuple


def jacobi(
        function: sp.Matrix,
        x: sp.Matrix,
        xk: np.ndarray,
        k: int
    ) -> None:
    """Performs one iteration of the Jacobi method for a given function.

    Args:
        function (sp.Matrix): The symbolic representation of the function to
            be minimized.
        x (sp.Matrix): The symbolic vector of variables.
        xk (np.ndarray): The current numerical approximation of the variables.
        k (int): The index of the variable to update in the current iteration.
    """
    # Substitute all variables except the k-th one with their current values in
    # xk
    values: Dict[str, Any] = {x[i]: xk[i] for i in range(len(x)) if i != k}
    # Evaluate the function with the substituted values
    fx: sp.Matrix = sp.Matrix(function.subs(values))[0]
    # Compute the derivative of the function with respect to the k-th variable
    dfx: sp.Matrix = fx.diff(x[k])
    # Solve for the new value of the k-th variable
    xk[k] = sp.solve(dfx, x[k])[0]

def compute_error(f: sp.Matrix, x: sp.Matrix, values: ListProxy) -> float:
    """Computes the error as the norm of the gradient of the function.

    Args:
        f (sp.Matrix): The symbolic representation of the function to be
            minimized.
        x (sp.Matrix): The symbolic vector of variables.
        values (ListProxy): The current numerical approximation of the
            variables.

    Returns:
        float: The norm of the gradient, representing the error.
    """
    # Compute the gradient of the function
    gradient: ImmutableDenseNDimArray = derive_by_array(f, x)

    # Create a dictionary of variable-value pairs for substitution
    values: Dict[str, Any] = {x[i]: values[i] for i in range(len(values))}

    # Substitute the current values into the gradient
    gradient_value: ImmutableDenseNDimArray = gradient.subs(values)

    # Extract numerical values from the gradient and compute its norm
    nums: np.ndarray = np.array(
        [float(num[0][0][0]) for num in gradient_value.tolist()]
    )

    return np.linalg.norm(nums)

def block_coordinate_descent_method(
        xk: np.ndarray,
        tol: float
    ) -> Tuple[np.ndarray, List[float]]:
    """Performs the Block Coordinate Descent (BCD) algorithm to find the
    minimum of a quadratic function.

    Args:
        xk (np.ndarray): Initial guess for the variables.
        tol (float): Tolerance for the stopping criterion.

    Returns:
        Tuple[np.ndarray, List[float]]: A tuple containing the final values of
            the variables and the error history.
    """
    # Number of variables
    n: int = len(xk)

    # Convert xk to a numpy array of type float64
    xk: np.ndarray = np.array(xk, dtype=np.float64)

    # Create the matrix A using diagonals
    values: List[np.ndarray] = [
        2 * np.ones(n - 1),
        6 * np.ones(n),
        2 * np.ones(n - 1)
    ]

    A: sp.Matrix = sp.Matrix(diags(values, [-1, 0, 1]).toarray())

    # Create the vector b
    b: np.ndarray = np.full((n), 15).T
    b[0] = 12
    b[n - 1] = 12
    b: sp.Matrix = sp.Matrix(b)

    # Create a symbolic vector for the variables
    x: sp.Matrix = sp.Matrix(sp.MatrixSymbol('x', n, 1))

    # Initialize iteration counter
    k: int = 1

    # Define the quadratic function to be minimized
    xtAx: sp.MatMul = 0.5 * sp.MatMul(x.T, A, x)
    btx: sp.MatMul = sp.MatMul(b.T, x)
    f: sp.Matrix = sp.Matrix(sp.MatAdd(xtAx, -btx))

    # Compute initial error
    error: List[float] = [compute_error(f, x, xk)]

    # Multiprocessing
    manager: SyncManager = multiprocessing.Manager()
    _vars: ListProxy = manager.list(xk)

    # Pool with N process, depeding of computer's cpus
    p: multiprocessing.pool.Pool = multiprocessing.Pool(
        psutil.cpu_count(logical=False)
    )

    # Covert all arguments in something that Pool can understand
    mono_arg_func: functools.partial = functools.partial(jacobi, f, x, _vars)

    while (error[-1] > tol):
        # Run in parallel
        p.map(mono_arg_func, range(n))
        k += 1
        error.append(compute_error(f, x, _vars))

    return np.array(_vars), error

def plot_error(k: np.ndarray, error: List[float]) -> None:
    """Plots the error over iterations.

    Args:
        k (np.ndarray): Array of iteration numbers.
        error (List[float]): List of error values for each iteration.
    """
    plt.title('Parallel BCD')
    plt.xlabel('Iteration (k)')
    plt.ylabel('Error')
    plt.grid()
    plt.plot(k, error)
    plt.show()


if __name__ == '__main__':
    # Number of variables
    N: int = 50
    # Tolerance for stopping criterion
    tol: float = 10e-5
    # Initial guess for the variables
    x: np.ndarray = np.ones(N)

    # Perform the Maximum Block Improvement algorithm
    xk, error = block_coordinate_descent_method(x, tol)
    print(f'x value is {xk}')
    
    # Create an array of iteration numbers
    k: np.ndarray = np.array(range(1, len(error) + 1))
    
    # Plot the error over iterations
    plot_error(k, error)
