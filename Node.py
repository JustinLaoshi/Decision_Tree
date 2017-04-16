from __future__ import division
from collections import Counter
import numpy as np
import scipy.io
import math
import pdb
import random
import sklearn
from sklearn import preprocessing
import itertools
import csv
import matplotlib as mpl
import matplotlib.pyplot as plt 


class Node:

	def __init__(self, split_rule=None, left=None, right=None, label=None, entropy=None):
		self.split_rule = split_rule
		self.left = left
		self.right = right
		self.label = label
		self.entropy = entropy
		if label != None:
			self.leaf = True
		else:
			self.leaf = False

	def setLabel(self, label):
		self.label = label
		self.leaf = True
