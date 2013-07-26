# This is a python library of algorithms that perform concept formation written by
# Christopher MacLellan (http://www.christopia.net).  
# 
# Currently only COBWEB
# (http://axon.cs.byu.edu/~martinez/classes/678/Papers/Fisher_Cobweb.pdf) is
# implemented. The system accepts a stream of instances, which are represented as
# hash tables of attributes and values, and learns a concept hierarchy.
# 
# Current plans are to extend this library to include the LABYRINTH
# (http://www.isle.org/~langley/papers/labyrinth.cfb.pdf) algorithm for learning
# structured concepts with relations and components.

import copy
import random
import sys
import math
import heapq
import operator

import csv, sys, gensim, logging, numpy as np, pylab as pl
import time
import pylab
from matplotlib.patches import Rectangle
import matplotlib.pyplot as plt
from P import Parser
from I import Instance
from U import Utility
from V import Visualization
import pickle
import inspect
from COBWEB import COBWEB
from M import Measure

class ConceptTree:
	mergedNodes = []
	splitNodes = []

	def __init__(self, root=None, parent=None, concept_tree=None):
		"""
		The constructor.
		"""
		self.instances = []
		self.viz = Visualization(self)
		self.measure = Measure(self)
		# keep track of the root
		if root:
			self.root = root
		else:
			self.root = self
		self.parent = parent
		# check if the constructor is being used as a copy constructor
		if concept_tree:
			self.utility = copy.deepcopy(concept_tree.utility)
			self.utility.tree = self
			self.children = copy.deepcopy(concept_tree.children)
			for c in self.children:
				c.parent = self
			self.instances = copy.deepcopy(concept_tree.instances)
		else:
			self.utility = Utility(self)
			self.children = []
		# must initialize after utility has been set
		self.CBWB = COBWEB(self)
	
	def makeTree(self, root=None, parent=None, concept_tree=None):
		return ConceptTree(root, parent, concept_tree)
	
	def saveObject(self, filename):
		with open(filename, 'wb') as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
	
	def readObject(self, filename):
		with open(filename, 'rb') as input:
			tree = pickle.load(input)
			attributes = inspect.getmembers(tree)
			for a in attributes:
				setattr(self, a[0], a[1])
	
	def firstCobweb(self, instance):
		ConceptTree.mergedNodes = []
		ConceptTree.splitNodes = []
		self.CBWB.firstCobweb(instance)
	
	def cobweb(self, instance):
		self.CBWB.cobweb(instance)

	def pretty_print(self):
		c.viz.pretty_print()
	
	def toDot(self, fileName, withStrings=True, name_of_latest=None):
		self.viz.toDot(fileName, withStrings, name_of_latest)
	
	def __str__(self):
		"""
		Converts the categorization tree into a string.
		"""
		ret = str(self.utility.av_counts)
		for c in self.children:
			ret += "\n" + str(c)
		return ret

# add an instance to the cluster hierarchy and record how much it changes the tree
def addInc(t, instance):
	if len(t.instances) == 0:
		t.utility.increment_counts(instance)
		return
	t.firstCobweb(instance)
	# Children are used to describe nodes so disapearing addresses are not an issue
	# This is a list of the children of each node which was merged
	mergedNodes = [x for x in ConceptTree.mergedNodes if not (x in ConceptTree.splitNodes)]
	# This is a list of the children of each node which was split
	splitNodes = [x for x in ConceptTree.splitNodes if not (x in ConceptTree.mergedNodes)]
	# The children of a merged node are one step deeper than the node that was merged
	instance.merges = [-1 if nodeList == [] else max([node.measure.getDepth() for node in nodeList])-1 for nodeList in mergedNodes]
	# The children of a split node are not since they moved up after their parent was split
	instance.splits = [-1 if nodeList == [] else max([node.measure.getDepth() for node in nodeList]) for nodeList in splitNodes]
	instance.depth = t.measure.getInstanceDepth(instance)

if __name__ == "__main__":
	# Read in Data
	namecols = [0]
	timecols = [2]
	valcols = [3,4]
	parser = Parser("AllPhoneData_pruned.csv",namecols,timecols,valcols,normalize=True)
	
	index = 0
	t = ConceptTree()

	while not parser.atEnd():
		addInc(t, parser.getNext())
	t.viz.plotClusters(Instance.properties[0], Instance.properties[1], depth=1)
	
	t.saveObject('test.cbwb')
