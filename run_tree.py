from __future__ import division
from collections import Counter
from DecisionTree import DecisionTree
from Node import Node
import numpy as np
import scipy.io
import pdb
import math
import random
import sklearn
from sklearn.feature_extraction import DictVectorizer
import itertools
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt 


def benchmark(pred_labels, true_labels):
    errors = pred_labels != true_labels
    err_rate = np.sum(errors) / float(true_labels.shape[0])
    indices = errors.nonzero()
    return err_rate, indices

def main():
	#spam()
	census()

def spam():

	#load all the spam data
	spam_data = scipy.io.loadmat('spam-dataset/spam_data.mat')
	test_data = spam_data['test_data']
	training_labels = spam_data['training_labels']
	training_data = spam_data['training_data']
	print(training_data.shape[1], 'how many features used')
	training_data, training_labels = sklearn.utils.shuffle(training_data, training_labels)

	#split training data
	#learn_set, learn_labels = training_data[:4000], training_labels[:4000]
	learn_set, learn_labels = training_data, training_labels
	valid_set, valid_labels = training_data[4000:], training_labels[4000:]	

	#train and predict on a single tree
	# spamTree = DecisionTree(learn_set, learn_labels)
	# spamTree.train(learn_set, learn_labels, spamTree.root)
	# pred_labels = spamTree.predict(test_data)
	#print(benchmark(pred_labels, valid_labels)[0])

	#make random forest
	NUM_TREES = 100
	forest = []
	# pred_labels = np.zeros((valid_set.shape[0], 1))
	# sumOfPred = np.zeros((valid_set.shape[0], 1))
	pred_labels = np.zeros((test_data.shape[0], 1))
	sumOfPred = np.zeros((test_data.shape[0], 1))	
	for i in range(0, NUM_TREES):
		print('Now at tree #', i)
		nPrime = np.random.choice(learn_set.shape[0], learn_set.shape[0], True)
		x = learn_set[nPrime]
		y = learn_labels[nPrime]
		tree = DecisionTree(x, y)
		tree.train(x, y, tree.root, True)
		forest.append(tree)
	for tree in forest:
	#	sumOfPred += tree.predict(valid_set)
		sumOfPred += tree.predict(test_data)

	for i in range(0, test_data.shape[0]):
		if sumOfPred[i]/NUM_TREES > .5:
			pred_labels[i] = 1
		elif sumOfPred[i]/NUM_TREES < .5:
			pred_labels[i] = 0
		else:
			pred_labels[i] = random.randint(0, 1)
	#print(benchmark(pred_labels, valid_labels)[0])


	#make csv
	csvList = [['Id,Category']]
	for i in range(1, 5858):
	    csvList.append([i, int(pred_labels[i-1][0])])
	with open('spamForest.csv', 'w', newline='') as fp:
	    a = csv.writer(fp, delimiter=',')
	    a.writerows(csvList)

	return 0

def census():
	train_data = csv.DictReader(open('census_data/train_data.csv'))
	test_data = csv.DictReader(open('census_data/test_data.csv'))
	dataList = []
	testList = []
	train_labels = np.zeros((32724, 1))
	i = 0
	for row in train_data:
		del row['fnlwgt']
		del row['education-num']
		train_labels[i] = row['label']
		del row['label']
		dataList.append(row)
		for key in row.keys():
			try:
				row[key] = float(row[key])
			except:
				pass
		i += 1

	for row in test_data:
		del row['fnlwgt']
		del row['education-num']
		testList.append(row)
		for key in row.keys():
			try:
				row[key] = float(row[key])
			except:
				pass		

	v = DictVectorizer(sparse=False)
	x = v.fit_transform(dataList)
	test = v.transform(testList)

	imputer = sklearn.preprocessing.Imputer(missing_values='NaN', strategy='median', axis=0, verbose=0, copy=True)
	x = imputer.fit_transform(x)
	test = imputer.transform(test)

	censusTree = DecisionTree(x, train_labels)
	censusTree.train(x, train_labels, censusTree.root)
	pred_labels = censusTree.predict(test)

	#make random forest
	# NUM_TREES = 100
	# forest = []
	# # pred_labels = np.zeros((valid_set.shape[0], 1))
	# # sumOfPred = np.zeros((valid_set.shape[0], 1))
	# pred_labels = np.zeros((test.shape[0], 1))
	# sumOfPred = np.zeros((test.shape[0], 1))	
	# for i in range(0, NUM_TREES):
	# 	print('Now at tree #', i)
	# 	nPrime = np.random.choice(x.shape[0], x.shape[0], True)
	# 	x = x[nPrime]
	# 	y = train_labels[nPrime]
	# 	tree = DecisionTree(x, y)
	# 	tree.train(x, y, tree.root, True)
	# 	forest.append(tree)
	# for tree in forest:
	# #	sumOfPred += tree.predict(valid_set)
	# 	sumOfPred += tree.predict(test)

	# for i in range(0, test_data.shape[0]):
	# 	if sumOfPred[i]/NUM_TREES > .5:
	# 		pred_labels[i] = 1
	# 	elif sumOfPred[i]/NUM_TREES < .5:
	# 		pred_labels[i] = 0
	# 	else:
	# 		pred_labels[i] = random.randint(0, 1)

	#make csv
	csvList = [['Id,Category']]
	for i in range(1, 16119):
	    csvList.append([i, int(pred_labels[i-1][0])])
	with open('censusNoForest.csv', 'w', newline='') as fp:
	    a = csv.writer(fp, delimiter=',')
	    a.writerows(csvList)
	return 0












































if __name__ == "__main__":
    main()	
