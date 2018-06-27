#! /usr/bin/env python3

import numpy as np
from gausseidel import gauss_seidel
import math
import time

def gs_classifier_train(A, b, maxloss=0.5, maxiters=None):
	'Train a linear classifier from a matrix of training data (A) and training classes (b)'

	def shuffle(A,b):
		r = np.arange(0, N)
		np.random.shuffle(r)
		return A[r,:], b[r]

	# add a dummy column of all ones to training data for the vector bias
	A = np.append(A, np.ones((A.shape[0],1)), axis=1)
	N,M = A.shape

	# if maxiters is not supplied, calculate maxiters from the number of rows
	maxiters = maxiters if maxiters is not None else N*5

	iterations = 0
	best_solution = np.zeros(M, dtype='double')
	best_loss = math.inf
	while True:
		# select rows for this iteration
		i = np.arange(iterations, iterations+M) % N
		x0 = gauss_seidel(A[i,:], b[i], best_solution)
		iterations += 1
		if x0 is None:
			# if this problem didn't converge, shuffle the training data
			# hopefully we won't get any more subproblems that diverge
			A, b = shuffle(A, b)
			continue

		# make predictions over the training data and calculate loss
		predictions = (A @ x0) * b
		loss = -predictions[predictions<0].sum()
		# if this is a better solution, update the vector
		if loss < best_loss:
			best_solution, best_loss = x0, loss
			# return if all samples were correctly classified
			if (predictions > 0).all():
				break
			# return if the loss is acceptable
			if loss < maxloss:
				break
			iterations = 0
			# reorder the training data so that the best classified
			# samples are at the beginning
			s = np.argsort(predictions)
			A, b = A[-s,:], b[-s]
		if iterations >= maxiters:
			break
	return best_solution, best_loss

if __name__ == '__main__':
	dataset = np.array([
		[0, 0, 1],
		[2, 1, 1],
		[2, 2, 1],
		[3, 3, -1],
		[3, 4, -1],
		[1.5, 2.2, 1],
		[4, 3, -1],
		[1, 2, 1],
		[3.2, 1.5, 1],
		[4, 4, -1]
	])
	t0 = time.time()
	xf = gs_classifier_train(dataset[:,0:2],dataset[:,2])
	t1 = time.time()
	print(xf)
	print(t1-t0)
