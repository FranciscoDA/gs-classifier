#! /usr/bin/env python3

import numpy as np
from itertools import permutations

def index_permutations(shape):
	'Iterate over all the possible index permutations over a specific ndarray shape'
	for a in permutations(range(shape[0])):
		if len(shape) == 1:
			yield (np.array(a),)
		else:
			for b in index_permutations(shape[1:]):
				yield (np.array(a), *b)

def diagonal_dominant(A):
	'Returns True if A is a row-wise diagonal dominant matrix (A must be a numpy ndarray)'
	D = np.abs(A.diagonal())
	S = np.abs(A).sum(axis=1) - D
	return (D > S).all()

def positive_definite(A):
	'Returns True if A is a positive-definite matrix (A must be a numpy ndarray)'
	if not (A==A.T).all():
		return False # not symmetric => can't be positive-definite
	try:
		# cholesky decomposition fails if A is not positive-definite
		np.linalg.cholesky(A)
		return True
	except np.linalg.LinAlgError:
		return False

def gauss_seidel(A, b, x0, epsilon=0.0001, upsilon=100000.0, maxiters=1000):
	def gs_solve(A, b, x0):
		xprev = x0
		iters = 0
		while iters < maxiters:
			xnew = np.zeros_like(xprev)
			for i in range(len(xnew)):
				if A[i,i] != 0:
					S = A[i,0:i] @ xnew[0:i] + A[i,i+1:] @ xprev[i+1:]
					xnew[i] = 1/A[i,i] * (b[i] - S)
			error = np.linalg.norm(xnew-xprev)
			if error is None or error is np.inf or error is np.nan or error > upsilon:
				return None
			if error < epsilon:
				return xnew
			xprev = xnew
			iters += 1

	# Some matrix arrangements may not be positive-definite
	# nor diagonal dominant. However, the method may converge anyways.
	# So we store a queue of those candidates
	queue = []
	for row_mapping, col_mapping in index_permutations(A.shape):
		A_m = A[row_mapping,:][:,col_mapping]
		b_m, x0_m = b[row_mapping], x0[col_mapping]
		if (A_m.diagonal() == 0).all():
			continue
		if not positive_definite(A_m) and not diagonal_dominant(A_m):
			# not PD and not DD => add to queue
			queue.append((row_mapping, col_mapping))
			continue

		solution = gs_solve(A_m, b_m, x0_m)
		if solution is not None:
			return solution[col_mapping[col_mapping]]

	for row_mapping, col_mapping in queue:
		A_m = A[row_mapping,:][:,col_mapping]
		b_m, x0_m = b[row_mapping], x0[col_mapping]
		solution = gs_solve(A_m, b_m, x0_m)
		if solution is not None:
			return solution[col_mapping[col_mapping]]

if __name__ == '__main__':
	# test problems
	# TODO: Add proper testing
	A  = np.array([[1,3,6], [2,1,1], [1,2,3]], dtype='double')
	b  = np.array([20,8,12], dtype='double')
	x0 = np.array([1.5,2,1], dtype='double')

	xf = gauss_seidel(A,b,x0)
	print('x = {0}'.format(xf))
	print('Ax = {0}'.format(A @ xf))
