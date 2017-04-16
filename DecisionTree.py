from __future__ import division
from collections import Counter
import numpy as np
import pdb
import scipy.io
import math
import random
import sklearn
from sklearn import preprocessing
import itertools
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt 
from Node import Node


class DecisionTree:

	def __init__(self, training_data=None, training_labels=None):
		self.training_data = training_data
		self.training_labels = training_labels
		self.root = Node()
		if training_labels != None:
			rootHist = self.makeHist(training_labels)
			self.root.entropy = self.calculateEntropy(rootHist)


	def calculateEntropy(self, y):
		"""
		Calculates the entropy at one node. y is a hist (dict).
		"""
		total = sum(y.values())
		temp = 0.0
		for key, val in y.items():
			if val == 0:
				continue
			p = val/total
			temp += p*np.log2(p)
		return -1 * temp

	def makeHist(self, y):
		"""
		Makes a histogram as a dictionary out of the labels y.
		"""
		hist = {}
		zeros = 0
		ones = 0
		for i in range(0, y.shape[0]):
			if y[i] == 0:
				zeros += 1
			else:
				ones += 1
		hist[0] = zeros
		hist[1] = ones
		return hist

	def impurity(self, left_label_hist, right_label_hist):
		"""
		Calculates the entropy after a split.

		>>> test = DecisionTree()
		>>> a = {0:1, 1:1}
		>>> b = {0:1, 1:1}
		>>> test.impurity(a, b)
		1.0
		>>> a = {0:7, 1:2}
		>>> b = {0:1, 1:8}
		>>> test.impurity(a, b)
		0.63373142064213306
		"""
		totalLeft = sum(left_label_hist.values())
		totalRight = sum(right_label_hist.values())
		total = totalLeft + totalRight
		leftSum = self.calculateEntropy(left_label_hist)
		rightSum = self.calculateEntropy(right_label_hist)	
		return (totalLeft*leftSum + totalRight*rightSum) / total

	def infoGain(self, y1, y2, entropy):
		"""
		Calculates the improvement in entropy from the root node.
		"""
		left_label_hist = self.makeHist(y1)
		right_label_hist = self.makeHist(y2)
		hAfter = self.impurity(left_label_hist, right_label_hist)
		return entropy - hAfter

	def segmenter(self, training_data, training_labels, entropy, forest=False):
		"""
		Returns the best feature and value and the data splitted.
		"""
		bestFeature = -1
		bestValue = -1
		bestInfoGain = -1
		bestX1 = training_data
		bestX2 = training_data
		bestY1 = training_labels
		bestY2 = training_labels
		d = training_data.shape[1]
		n = training_data.shape[0]
		features = []

		#adds features depending on if we are using forest or not
		if forest == False:
			for i in range(0, d):
				features.append(i)
		else:
			m = round(math.sqrt(d))
			while len(features) != m:
				rand = random.randint(0, d-1)
				if rand not in features:
					features.append(rand)


		for f in features:
			valueList = []
			for sample in training_data:
				if sample[f] not in valueList:
					valueList.append(sample[f])
			for v in valueList:
				x1, y1 = np.copy(training_data), np.copy(training_labels)
				x2, y2 = np.copy(training_data), np.copy(training_labels)
				del1, del2 = [], []
				for i in range(0, n):
					if x2[i][f] <= v:
						del2.append(i)
					if x1[i][f] > v:
						del1.append(i)
				x1, y1 = np.delete(x1, del1, 0), np.delete(y1, del1, 0)
				x2, y2 = np.delete(x2, del2, 0), np.delete(y2, del2, 0)
				temp = self.infoGain(y1, y2, entropy)
				if temp > bestInfoGain and temp > 0:
					bestFeature = f
					bestValue = v
					bestInfoGain = temp
					bestX1, bestX2 = x1, x2
					bestY1, bestY2 = y1, y2
		return bestFeature, bestValue, bestX1, bestX2, bestY1, bestY2

	def train(self, training_data, training_labels, node, forest=False):
		"""
		Trains (builds) this decision tree.
		"""
		f, v, x1, x2, y1, y2 = self.segmenter(training_data, training_labels, node.entropy, forest)
		node.split_rule = (f, v)
		if node == self.root:
			print(node.split_rule)
		if training_data.shape[0] < 1200 or f == -1:
			lastHist = self.makeHist(training_labels)
			if lastHist[0] >= lastHist[1]:
				node.setLabel(0)
			else:
				node.setLabel(1)
			return		
		y1entropy, y2entropy = self.makeHist(y1), self.makeHist(y2)
		y1entropy, y2entropy = self.calculateEntropy(y1entropy), self.calculateEntropy(y2entropy)
		node.left = Node(None, None, None, None, y1entropy)
		node.right = Node(None, None, None, None, y2entropy)
		self.train(x1, y1, node.left)
		self.train(x2, y2, node.right)


	def predict(self, test_data):
		"""
		Predict labels for test data by traversing down this decision tree.
		"""
		pred_labels = np.zeros((test_data.shape[0], 1))
		for i in range(0, test_data.shape[0]):
			currNode = self.root
			while currNode.leaf != True:
				f, v = currNode.split_rule[0], currNode.split_rule[1]
				if test_data[i][f] <= v:
					currNode = currNode.left
				else:
					currNode = currNode.right
			pred_labels[i] = currNode.label
		return pred_labels





















