import functools
import multiprocessing
import numpy as np
import psutil
from scipy.sparse import diags
import sympy as sp
from sympy import MatrixSymbol, Matrix
from sympy.tensor.array import derive_by_array
import matplotlib.pyplot as plt


def jacobi(function, x, xk, k):
	# All values except the current
	values = {x[i]: xk[i] for i in range(len(x)) if i != k}
	# Get the function evaluated in values
	fx = Matrix(function.subs(values))[0]
	# Derivative
	dfx = fx.diff(x[k])
	# Get x value
	xk[k] = sp.solve(dfx, x[k])[0]

def compute_error(f, x, values):
	gradient = derive_by_array(f, x)
	values = {x[i]: values[i] for i in range(len(values))}
	gradient_value = gradient.subs(values)
	nums = np.array([float(num[0][0][0]) for num in gradient_value.tolist()])
	return np.linalg.norm(nums)

def maximum_block_improvement(xk, tol):
	global p
	# x size
	n = len(xk)
	# xk to numpy / Initial values
	xk = np.array(xk, dtype=np.float64)
	# Create matrix A
	values = np.array([2*np.ones(n - 1), 6*np.ones(n), 2*np.ones(n - 1)])
	A = Matrix(diags(values, [-1, 0, 1]).toarray())
	# Create vector b
	b = np.full((n), 15).T
	b[0] = 12
	b[n - 1] = 12
	b = Matrix(b)
	# Array of vars
	x = Matrix(MatrixSymbol('x', n, 1))
	# Current iteration
	k = 1
	# Function
	xtAx = 0.5*sp.MatMul(x.T, A, x)
	btx = sp.MatMul(b.T, x)
	f = Matrix(sp.MatAdd(xtAx, -btx))
	error = [compute_error(f, x, xk)]
	# Multiprocessing
	manager = multiprocessing.Manager()
	_vars = manager.list(xk)
	# Pool with N process, depeding of computer's cpus
	p = multiprocessing.Pool(psutil.cpu_count(logical=False))
	# Covert all arguments in something that Pool can understand
	mono_arg_func = functools.partial(jacobi, f, x, _vars)
	while (error[-1] > tol):
		# Run in parallel
		p.map(mono_arg_func, range(n))
		k += 1
		error.append(compute_error(f, x, _vars))
	return np.array(_vars), error

def plot_error(k, error):
    plt.title('Parallel MBI')
    plt.xlabel('Iteration (k)')
    plt.ylabel('Error')
    plt.grid()
    plt.plot(k, error)
    plt.show()


if __name__ == '__main__':
	N = 50
	tol = 10e-5
	x = [1 for _ in range(N)]
	xk, error = maximum_block_improvement(x, tol)
	print('x value is:', xk)
	k = np.array(range(1, len(error)+1))
	plot_error(k, error)
