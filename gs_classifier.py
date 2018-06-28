#! /usr/bin/env python3

import numpy as np
from gausseidel import gauss_seidel
import math

def gs_classifier_train(S, eta=0.1, maxloss=-math.inf, rounds=3):
	'''
	Train a linear classifier from a matrix of training data (S) where the last column are the classes
	
	Parameters
	----------
	S : array_like
		The training data
	eta : float
		Learning rate (default: 0.1)
	maxloss : float
		Provides an upper bound for the loss to terminate early (default: -inf)
	rounds : int
		After visiting each possible batch, the training data is reshuffled
		and iterated again. This parameter controls how many times to do this
		(default: 3)
	'''

	N,M = S.shape
	# add a dummy column of all ones to training data for the vector bias
	S = np.insert(S, -1, np.ones(N), axis=1)
	A = S[:,:-1]
	b = S[:,-1]

	iterations = 0
	best_solution = np.zeros(M, dtype='double')
	best_loss = math.inf
	while True:
		# select rows for this iteration in a sliding window fashion
		i = np.arange(iterations, iterations+M) % N
		x0 = gauss_seidel(A[i,:], b[i], best_solution)
		iterations += 1
		if x0 is None:
			# if this problem didn't converge, shuffle the training data
			# hopefully we won't get any more subproblems that diverge
			np.random.shuffle(S)
			continue

		# eta is the learning rate
		x0 = best_solution * (1-eta) + eta * x0

		# make predictions over the training data and calculate loss
		predictions = (A @ x0) * b
		loss = np.maximum(0, -predictions).sum()
		# if this is a better solution, update the vector
		if loss < best_loss:
			best_solution, best_loss = x0, loss
			# return if the loss is acceptable
			if loss < maxloss:
				break
			# reset iteration counter and shuffle
			iterations = 0
			np.random.shuffle(S)
		if iterations >= N:
			if rounds <= 0:
				break
			else:
				np.random.shuffle(S)
				iterations = 0
				rounds -= 1
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
	xf = gs_classifier_train(dataset)
	print(xf)
