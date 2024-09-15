import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import diags
import sympy as sp
from sympy import MatrixSymbol, Matrix
from sympy.tensor.array import derive_by_array


def jacobi(function, x, xk, k):
	# All values except the current
	values = {x[i]: xk[i] for i in range(xk.size) if i != k}
	# Get the function evaluated in values
	fx = Matrix(function.subs(values))[0]
	# Derivative
	dfx = fx.diff(x[k])
	# Get x value
	newX = sp.solve(dfx, x[k])[0]
	return newX

def computeError(f, x, values):
	# Compute gradient
	gradient = derive_by_array(f, x)
	# Values to evaluate gradient
	values = {x[i]: values[i] for i in range(values.size)}
	# Gradient value after evaluation
	gradient_value = gradient.subs(values)
	# Compute norm
	nums = np.array([float(num[0][0][0]) for num in gradient_value.tolist()])
	return np.linalg.norm(nums)

def maximum_block_improvement(xk, tol):
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
	# Compute error
	error = [computeError(f, x, xk)]
	while (error[-1] > tol):
		for j in range(n):
			# Jacobi to update the values
			xk[j] = jacobi(f, x, xk, j)
		k += 1
		# Compute error
		error.append(computeError(f, x, xk))
	return xk, error

def plot_error(k, error):
	plt.title('Sequential MBI')
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
	k = np.array(range(1, len(error) + 1))
	plot_error(k, error)
