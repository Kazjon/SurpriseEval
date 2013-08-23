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
import sys
import numpy as np
import pickle
import os

from P import Parser
from I import Instance
from U import Utility
from V import Visualization
from COBWEB import COBWEB
from M import Measure

class Node:
	mergedNodes = []
	splitNodes = []

	def __init__(self, root=None, parent=None, concept_tree=None, filename=None):
		"""
		The constructor.
		"""
		self.instances = []
		self.viz = Visualization(self)
		self.measure = Measure(self)
		self.filename = filename
		self.old_count = 0
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
		return Node(root, parent, concept_tree)
	
	def saveObject(self, filename, remove='tmp.cbwb'):
		with open(filename, 'wb') as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
		if os.path.isfile(remove):
			os.remove(remove)
		print "Saved", filename
	
	def firstCobweb(self, instance):
		Node.mergedNodes = []
		Node.splitNodes = []
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
	instance.before_delta = t.measure.delta()
	if len(t.instances) == 0:
		t.utility.increment_counts(instance)
	else:
		t.firstCobweb(instance)
	setMeasures(t, instance)

def setMeasures(t, instance):
	mergedNodes = [x for x in Node.mergedNodes if not (x in Node.splitNodes)]
	splitNodes = [x for x in Node.splitNodes if not (x in Node.mergedNodes)]
	instance.merges = [-1 if nodeList == [] else max([node.measure.getDepth() for node in nodeList])-1 for nodeList in mergedNodes]
	instance.splits = [-1 if nodeList == [] else max([node.measure.getDepth() for node in nodeList]) for nodeList in splitNodes]
	instance.depth = float(t.measure.getInstanceDepth(instance))
	instance.after_delta = t.measure.delta()

def readObject(filename):
	with open(filename, 'rb') as input:
		return pickle.load(input)

def findFile(namestart, parser, prefix='.'):
	for f in os.listdir(prefix):
		fn, ext = os.path.splitext(f)
		if ext == ".cbwb" and fn.startswith(namestart):
			index = int(fn.strip(namestart))
			print "Found",f,"and setting index to", index
			parser.index = index + 1
			return readObject(prefix+'/'+f)
	return Node()
	
if __name__ == "__main__":
	# Read in Data
	namecols = [0]
	timecols = [2]
	valcols = [9]
	parser = Parser("AllPhoneData_pruned.csv",namecols,timecols,valcols,normalize=True)
	
	#t = readObject('test.cbwb')
	t = findFile('test_', parser)
	old_index = parser.index
	
	while not parser.atEnd():
		addInc(t, parser.getNext())
		if parser.index >= old_index * 2:
			t.saveObject('test_'+str(parser.index)+'.cbwb', remove='test_'+str(old_index)+'.cbwb')
			old_index = parser.index
	t.viz.plotClusters('time', Instance.properties[0], depth=2)
	
	t.saveObject('test.cbwb')
