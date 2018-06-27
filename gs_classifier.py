#! /usr/bin/env python3

import numpy as np
from gausseidel import gauss_seidel
import random
import math
import time

def gs_classifier_train(A, b, maxloss=0.5, maxiters=None):
	'Train a linear classifier from a matrix of training data (A) and training classes (b)'

	# if maxiters is not supplied, calculate
	# maxiters from the rows-to-columns ratio
	maxiters = maxiters if maxiters is not None else (A.shape[0]//(A.shape[1]+1))*3

	# add a dummy column of all ones to training data for the vector bias
	A0 = np.ones((A.shape[0], A.shape[1]+1), 'double')
	A0[:,:-1] = A
	iterations = 0
	best_solution = np.zeros(A0.shape[1], dtype='double')
	best_loss = math.inf
	while True:
		# select rows for this iteration
		i = np.arange(iterations * A0.shape[1], (iterations+1)*A0.shape[1]) % A0.shape[0]
		x0 = gauss_seidel(A0[i,:], b[i], best_solution)
		iterations += 1
		if x0 is None:
			# if this problem didn't converge, shuffle the training data
			# hopefully we won't get any more subproblems that diverge
			r = np.arange(0, A0.shape[0])
			random.shuffle(r)
			A0, b = A0[r,:], b[r]
			continue

		# make predictions over the training data
		# and calculate loss
		predictions = (A0 @ x0) * b
		hinge_loss = np.maximum(0, 1-predictions).sum()
		# if this is a better solution, update the vector
		if hinge_loss < best_loss:
			best_solution, best_loss = x0, hinge_loss
			# reset the iteration counter,
			# maybe we can still improve the solution
			iterations = 0
			# return if the loss is acceptable or if all
			# samples were correctly classified
			if (predictions > 0).all():
				break
			if hinge_loss < maxloss:
				break
		if iterations >= maxiters:
			break
	return best_solution

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
