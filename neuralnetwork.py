#!/usr/bin/env python3
# Brandon Sayers
# Machine Learning A415
# Project 1
#
# This took a lot of will, determination, and time spent staring
# at a computer with an entirely blank, sullen look in the eyes.

import numpy as np
import csv


class Neural_Network(object):
	def __init__(self):
		# Define HyperParameters.
		self.inputLayerSize = 17
		self.hiddenLayerSize = 70
		self.outputLayerSize = 26
		self.learningrate = .001

		# Define memory for accuracy.
		self.epochcount = [0]
		self.correct = [0]

		# Weights.
		self.W1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize)
		self.W2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize)

		# Done: Expected accuracy reached.
		self.successcap = .95
		self.epochcap = 2000
		self.done = False

		# Loop until desired success rate is reached.
		# When reached, export learning statistics.
		# Define 2D gray-scale confusion matrix.
		while(self.done is not True):
			self.matrix = np.zeros((26,26))
			self.epoch()
		else:
			self.outputFile = open('project1data4.csv', 'w')
			writer = csv.writer(self.outputFile, dialect = 'excel')
			for values in self.matrix:
				writer.writerow(values)
			for index in range(len(self.epochcount)):
			 	writer.writerow([self.epochcount[index], self.correct[index]])
			self.outputFile.close()



	def epoch(self):
		# Learn on training set.
		# Iterates through file.
		# Extracts expected output and 16 element feature vector.
		# Begins the forward propagation.
		# Updates 2D gray-scale confusion matrix.
		#
		# X: Inputs.
		# Y: Expected output vector.
		# expected: Expected y output.
		# actual: Actual y output.
		learnfile = open('data_learn.dat', 'r')
		count = 0

		for line in learnfile:
			count += 1
			Y = np.zeros((1,26))
			X = line.strip('\n').split(',')
			expected = float(X[0])
			Y[0,int(expected)-1] = 1
			X[0] = 1
			X = np.float64(X)
			Yhat = self.forward(X)
			actual = Yhat.argmax() + 1
			self.matrix[int(expected-1),actual-1] += 1

			if int(expected) != int(actual):
				self.back(X, Y)

			if count == 10000:
				if (self.test() >= self.successcap or
				   self.epochcount[-1] >= self.epochcap):
					print("DONE")
					self.done = True


	def test(self):
		# Test neural network's classification accuracy by
		# comparing our predicted output to actual output.
		# Returns float, successrate.
		count = 0
		success = 0
		testfile = open('data_test.dat', 'r')
		for line in testfile:
			count += 1

			X = line.strip('\n').split(',')
			expected = float(X[0])
			X[0] = 1
			X = np.float64(X)
			actual = self.forward(X).argmax() + 1

			if int(expected) == int(actual):
				success += 1
		print("count,success:")
		print(count, success)
		successrate = float(success)/float(count)
		self.epochcount.append(max(self.epochcount) + 1)
		self.correct.append(successrate)

		print("Batch %d: %.4f success rate" % (max(self.epochcount),successrate))
		return successrate


	def forward(self, X):
		# Propagate inputs through network.
		# P1: Input into hidden layer
		# A1: Result from hidden layer activation function
		# P2: Input into output layer
		# Y:  Result from output layer softmax function
		self.P1 = np.dot(X, self.W1)
		self.P1 = self.P1.reshape((1,self.hiddenLayerSize))
		self.P1[0,0] = 1
		self.A1 = self.sigmoid(self.P1)
		self.A1[0,0] = 1

		self.P2 = np.dot(self.A1, self.W2)
		Y = self.softmax(self.P2)
		return Y

	def back(self,X,Y):
		# Backpropagate through network.
		dJdW1, dJdW2 = self.costFunctionPrime(X, Y)
		self.W1 = self.W1 - self.learningrate * dJdW1
		self.W2 = self.W2 - self.learningrate * dJdW2

	def softmax(self, P):
		# Apply softmax activation function
		Y = np.exp(P)/(np.sum(np.exp(P)))
		index = np.argmax(Y)
		return Y

	def sigmoid(self, P):
		# Apply sigmoid activation function
		return (1/(1 + np.exp(-P))).reshape((1,self.hiddenLayerSize))

	def sigmoidPrime(self, P):
		# Apply derivative of sigmoid function
		return np.exp(-P)/((1+np.exp(-P))**2)

	def costFunction(self, X, Y):
		# Compute cost for given X, Y.
		self.Y = self.forward(X)
		J = 0.5 * sum((Y - self.Y)**2)
		return J

	def costFunctionPrime(self, X, Y):
		# Apply derivative with respect to Weights
		self.Y = self.forward(X)

		delta2 = -(Y - self.Y)
		dJdW2 = np.dot(self.A1.T, delta2)

		sigp = self.sigmoidPrime(self.P1)
		sigp[0,0] = 0

		delta1 = np.dot(delta2, self.W2.T) * sigp
		dJdW1 = np.dot(X.T.reshape((self.inputLayerSize,1)), delta1)

		return dJdW1, dJdW2


Neural_Network()