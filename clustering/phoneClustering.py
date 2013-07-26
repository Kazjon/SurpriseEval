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
from Instance import Instance

class ConceptTree:
	mergedNodes = []
	splitNodes = []
	updatedNodes = []

	def __init__(self, root=None, parent=None, concept_tree=None):
		"""
		The constructor.
		"""
		self.labels = []
		# keep track of the root
		if root:
			self.root = root
		else:
			self.root = self
		self.parent = parent
		# check if the constructor is being used as a copy constructor
		if concept_tree:
			self.count = concept_tree.count
			self.av_counts = copy.deepcopy(concept_tree.av_counts)
			self.children = copy.deepcopy(concept_tree.children)
			self.labels = copy.deepcopy(concept_tree.labels)
			self.measures = copy.deepcopy(concept_tree.measures)
		else:
			self.measures = {}
			self.count = 0.0
			self.av_counts = {} # attribute_value_vector
			self.children = []
	
	def getDepth(self):
		if self.root == self or self.parent == None:
			return 0
		return 1 + self.parent.getDepth()
	
	def deepestChild(self):
		if self.children == []:
			return self.getDepth()
		maxDepth = 0
		for c in self.children:
			dc = c.deepestChild()
			if dc > maxDepth:
				maxDepth = dc
		return maxDepth
	
	def calc_measures(self, a, values, append=True):
		self.measures[a] = self.measures.setdefault(a,{})
		if isinstance(values, ConceptTree):
			self.measures[a]['list'] = (self.measures[a].get('list',[]) + values.measures[a]['list'])
		elif type(values) == dict:
			self.measures[a]['list'] = self.measures[a].setdefault('list',[])
			if append:
				self.measures[a]['list'].append(values[a])
			else:
				self.measures[a]['list'].remove(values[a])
		else:
			raise Exception('"values" must be either a node or an values')
		if len(self.measures[a]['list']) == 0:
			self.measures[a]['avg'] = 0.0
			return
		self.measures[a]['avg'] = np.average(self.measures[a]['list'])

	def increment_counts(self, instance, name=""):
		"""
		Increment the counts at the current node according to the specified
		instance.

		input:
			instance: {a1: v1, a2: v2, ...} - a hashtable of attr and values. 
		"""
		ConceptTree.updatedNodes.append(self)
		self.labels.append(name)
		self.count += 1.0 
		for a in instance:
			self.av_counts[a] = self.av_counts.setdefault(a,{})
			index = instance[a]
			amount = 1.0
			if type(instance[a]) == int or type(instance[a]) == float:
				index = 'numerically_valued_attribute'
				amount = instance[a]
				self.calc_measures(a, instance, True)
			self.av_counts[a][index] = (self.av_counts[a].get(index, 0) + amount)

	def decrement_counts(self, instance, name=""):
		"""
		Decrement the counts at the current node according to the specified
		instance.
		
		input:
			instance: {a1: v1, a2: v2, ...} - a hashtable of attr and values. 
		"""
		ConceptTree.updatedNodes.remove(self)
		self.labels.remove(name)
		self.count -= 1.0 
		for a in instance:
			self.av_counts[a] = self.av_counts.setdefault(a,{})
			index = instance[a]
			amount = 1.0
			if type(instance[a]) == int or type(instance[a]) == float:
				index = 'numerically_valued_attribute'
				amount = instance[a]
				self.calc_measures(a, instance, False)
			self.av_counts[a][index] = (self.av_counts[a].get(index, 0) - amount)
			# for clarity in printing we remove the values and attributes
			if self.av_counts[a][index] == 0:
				del self.av_counts[a][index]
			if self.av_counts[a] == {}:
				del self.av_counts[a]
	
	def update_counts_from_node(self, node):
		"""
		Increments the counts of the current node by the amount in the specified
		node.
		"""
		self.count += node.count
		for l in node.labels:
			self.labels.append(l)
		for a in node.av_counts:
			for v in node.av_counts[a]:
				if v == 'numerically_valued_attribute':
					self.calc_measures(a, node)
				self.av_counts[a] = self.av_counts.setdefault(a,{})
				self.av_counts[a][v] = (self.av_counts[a].get(v,0) + node.av_counts[a][v])

	def create_new_child(self,instance,name=""):
		"""
		Creates a new child (to the current node) with the counts initialized by
		the given instance. 
		"""
		new_child = ConceptTree(self.root, self)
		new_child.increment_counts(instance,name)
		self.children.append(new_child)

	def create_child_with_current_counts(self):
		"""
		Creates a new child (to the current node) with the counts initialized by
		the current node's counts.
		"""
		self.children.append(ConceptTree(self.root, self, self))

	def two_best_children(self,instance,name=""):
		"""
		Returns the indices of the two best children to incorporate the instance
		into in terms of category utility.

		input:
			instance: {a1: v1, a2: v2,...} - a hashtable of attr. and values. 
		output:
			(0.2,2),(0.1,3) - the category utility and indices for the two best
			children (the second tuple will be None if there is only 1 child).
		"""
		if len(self.children) == 0:
			raise Exception("No children!")
		
		self.increment_counts(instance,name)
		children_cu = []
		for i in range(len(self.children)):
			self.children[i].increment_counts(instance,name)
			children_cu.append((self.category_utility(),i))
			self.children[i].decrement_counts(instance,name)
		self.decrement_counts(instance,name)
		children_cu.sort(reverse=True)

		if len(self.children) == 1:
			return children_cu[0], None 

		return children_cu[0], children_cu[1]

	def new_child(self,instance,name=""):
		"""
		Updates root count and adds child -- permenant.
		"""
		return self.cu_for_new_child(instance,name,False)

	def cu_for_new_child(self,instance,name="",undo=True):
		"""
		Returns the category utility for creating a new child using the
		particular instance.
		"""
		self.increment_counts(instance,name)
		self.create_new_child(instance,name)
		cu = self.category_utility()
		if undo:
			self.children.pop()
			self.decrement_counts(instance,name)
		return cu

	def merge(self,best1,best2):
		"""
		A version of merge that is permenant.
		"""
		return self.cu_for_merge(best1,best2,False)

	def cu_for_merge(self, best1, best2, undo=True):
		"""
		Returns the category utility for merging the two best children.
		NOTE! - I decided that testing a merge does not incorporate the latest
		instance, but waits for a second call of cobweb on the root. The
		original paper says that I should incorporate the instance into the
		merged node, but since we don't do something like this for split I
		didn't do it here. This gives the option to merge multiple nodes before
		incorporating the instance. 

		input:
			best1: 1 - an index for a child in the children array.
			best2: 2 - an index for a child in the children array.
		output:
			0.02 - the category utility for the merge of best1 and best2.
		"""
		#TODO - Might want to consider adding the instance to the merged node.
		first = best1
		second = best2

		if second < first:
			temp = first 
			first = second 
			second = temp

		first_c = self.children[first]
		second_c = self.children[second]

		new_c = ConceptTree(self.root, self)
		new_c.update_counts_from_node(first_c)
		new_c.update_counts_from_node(second_c)

		self.children.pop(second)
		self.children.pop(first)
		self.children.append(new_c)

		cu = self.category_utility()

		if undo:
			self.children.pop()
			self.children.insert(first,first_c)
			self.children.insert(second,second_c)
		else:
			# If we aren't undoing the merge then we have to add the leaves
			new_c.children.append(first_c)
			first_c.parnet = new_c
			new_c.children.append(second_c)
			second_c.parnet = new_c
			ConceptTree.mergedNodes.append([first_c, second_c])
			for m in ConceptTree.mergedNodes:
				if len(m) > 2 and first_c in m and second_c in m:
					m.remove(first_c)
					m.remove(second_c)
					m.append(new_c)
			for s in ConceptTree.splitNodes:
				if len(s) > 2 and first_c in s and second_c in s:
					s.remove(first_c)
					s.remove(second_c)
					s.append(new_c)

		return cu

	def split(self,best):
		"""
		Permemantly split the best.
		"""
		return self.cu_for_split(best,False)

	def cu_for_split(self,best,undo=True):
		"""
		Return the category utility for splitting the best child.
		
		input:
			best1: 0 - an index for a child in the children array.
		output:
			0.03 - the category utility for the split of best1.
		"""
		oldChildren = self.children[0:len(self.children)]
		best_c = self.children.pop(best)
		for child in best_c.children:
			self.children.append(child)
		cu = self.category_utility()

		if undo:
			for i in range(len(best_c.children)):
				self.children.pop()
			self.children.insert(best,best_c)
		else:
			ConceptTree.splitNodes.append(best_c.children)
			for child in best_c.children:
				child.parent = self
			for m in ConceptTree.mergedNodes:
				if best_c in m:
					m.remove(best_c)
					for c in best_c.children:
						m.append(c)
			for s in ConceptTree.splitNodes:
				if best_c in s:
					s.remove(best_c)
					for c in best_c.children:
						s.append(c)

		return cu

	def check_children_eq_parent(self):
		"""
		Checks the property that the counts of the children sum to the same
		count as the parent. This is/was useful when debugging.
		"""
		if len(self.children) == 0:
			return

		child_count = 0.0
		for child in self.children:
			child_count += child.count
		assert self.count == child_count

	def is_instance(self,instance):
		"""
		Checks to see if the current node perfectly represents the instance (all
		of the attribute values the instance has are probability 1.0 and here
		are no extra attribute values).
		"""
		for attribute in self.av_counts.keys():
			if attribute not in instance:
				return False
			if type(instance[attribute]) == dict:
				for value in self.av_counts[attribute]:
					if (self.av_counts[attribute][value] / self.count) != 1.0:
						return False
					if instance[attribute] != value:
						return False
			else:
					if instance[attribute] != self.av_counts[attribute]['numerically_valued_attribute'] / self.count:
						return False
		
		for attribute in instance:
			if attribute not in self.av_counts:
				return False
			if type(instance[attribute]) == dict:
				if instance[attribute] not in self.av_counts[attribute]:
					return False
				if ((self.av_counts[attribute][instance[attribute]] / self.count) != 1.0):
					return False
			else:
				if len(self.av_counts[attribute].keys()) != 1 or self.av_counts[attribute].get('numerically_valued_attribute', 0) == 0:
					return False
		
		return True

	def closest_matching_child(self,instance,name=""):
		"""
		Returns the child that is the best match for the instance in terms of
		difference between attribute value probabilites (note the instance has
		probability 1 of all attribute values it possesses). This function is
		used when the category utility of all actions is 0. It is a secondary
		heuristic for deciding the best node to add to.
		"""
		best = 0
		smallest_diff = float('inf')
		for i in range(len(self.children)):
			child = self.children[i]
			sum_diff = 0.0
			count = 0.0
			for attribute in child.av_counts:
				for value in self.av_counts[attribute]:
					count += 1
					if value == 'numerically_valued_attribute':
						sum_diff += instance[attribute] - (self.av_counts[attribute][value] / self.count)
					else:
						if attribute in instance and instance[attribute] == value:
							sum_diff += 1.0 - (self.av_counts[attribute][value] / self.count)
						else:
							sum_diff += 1.0

			if count > 0:
				sum_diff /= count
			else:
				sum_diff = float('inf')

			if sum_diff < smallest_diff:
				best = i
				smallest_diff = sum_diff
		
		return best
	
	def firstCobweb(self, instance,name=""):
		#print "start "+self.nodeString()
		ConceptTree.mergedNodes = []
		ConceptTree.splitNodes = []
		ConceptTree.updatedNodes = []
		self.cobweb(instance,name)
	
	def cobweb(self, instance,name=""):
		"""
		Incrementally integrates an instance into the categorization tree
		defined by the current node. This function operates recursively to
		integrate this instance and uses category utility as the heuristic to
		make decisions.
		"""
		#if not self.children and self.is_instance(instance): 
		#	self.increment_counts(instance,name)

		if not self.children:
			self.create_child_with_current_counts()
			self.increment_counts(instance,name)
			self.create_new_child(instance,name)
			
		else:
			best1, best2 = self.two_best_children(instance,name)
			operations = []
			operations.append((best1[0],"best"))
			operations.append((self.cu_for_new_child(instance,name),'new'))
			# a nodes only two children want to merge and split when they are exactly the same, leading to problems
			if best2 and len(self.children) > 2 and not [self.children[best1[1]], self.children[best2[1]]] in ConceptTree.mergedNodes:
				operations.append((self.cu_for_merge(best1[1],best2[1]),'merge'))
			if len(self.children[best1[1]].children) and not self.children[best1[1]].children in ConceptTree.splitNodes:
				operations.append((self.cu_for_split(best1[1]),'split'))
			operations.sort(reverse=True)

			best_action = operations[0][1]
			action_cu = operations[0][0]
			if action_cu == 0.0:
				self.increment_counts(instance,name)
				self.children[self.closest_matching_child(instance,name)].cobweb(instance,name)
			elif best_action == 'best':
				#print "best "+self.children[best1[1]].nodeString()
				self.increment_counts(instance,name)
				self.children[best1[1]].cobweb(instance,name)
			elif best_action == 'new':
				self.new_child(instance,name)
			elif best_action == 'merge':
				#print "merge "+self.children[best1[1]].nodeString()+" "+self.children[best2[1]].nodeString()
				self.merge(best1[1],best2[1])
				while len(ConceptTree.updatedNodes) > 0:
					ConceptTree.updatedNodes[0].decrement_counts(instance,name)
				self.root.cobweb(instance,name)
			elif best_action == 'split':
				#print "split "+self.children[best1[1]].nodeString()
				self.split(best1[1])
				while len(ConceptTree.updatedNodes) > 0:
					ConceptTree.updatedNodes[0].decrement_counts(instance,name)
				self.root.cobweb(instance,name)
			else:
				raise Exception("Should never get here.")


	def category_utility(self):
		"""
		The category utility is a local heuristic calculation to determine if
		the split of instances across the children increases the ability to
		guess from the parent node. 
		"""
		if len(self.children) == 0:
			return 0.0

		category_utility = 0.0

		exp_parent_guesses = self.expected_correct_guesses()

		for child in self.children:
			p_of_child = child.count / self.count
			exp_child_guesses = child.expected_correct_guesses()
			category_utility += p_of_child * (exp_child_guesses - exp_parent_guesses)

		# return the category utility normalized by the number of children.
		return category_utility / (1.0 * len(self.children))

	def expected_correct_guesses(self):
		"""
		The number of attribute value guesses we would be expected to get
		correct using the current concept.
		"""
		exp_count = 0.0
		#attributeList = self.getAllAVCounts()
		for attribute in self.av_counts:
			for value in self.av_counts[attribute]:
				if value == 'numerically_valued_attribute':
					exp_count += (self.measures[attribute]['avg'])**2
				else:
					exp_count += (self.av_counts[attribute][value] / self.count)**2
		return exp_count
	
	def getAllAVCounts(self):
		if self.children == []:
			return self.scaleAVCounts()
		listOfCounts = {}
		for c in self.children:
			childCounts = c.getAllAVCounts()
			for a in childCounts:
				if listOfCounts.get(a,0) == 0:
					listOfCounts[a] = {}
				for v in childCounts[a]:
					if listOfCounts.get(v,0) == 0:
						listOfCounts[a][v] = []
					if type(childCounts[a][v]) == list:
						listOfCounts[a][v] = listOfCounts[a][v] + childCounts[a][v]
					else:
						listOfCounts[a][v].append(childCounts[a][v])
		return listOfCounts

	def num_concepts(self):
		"""
		Return the number of concepts contained in the tree defined by the
		current node. 
		"""
		children_count = 0
		for c in self.children:
			children_count += c.num_concepts() 
		return 1 + children_count 

	def pretty_print(self,depth=0):
		"""
		Prints the categorization tree.
		"""
		for i in range(depth):
			print "\t",
				
		print self.nodeString()
		
		for c in self.children:
			c.pretty_print(depth+1)
	
	def nodeString(self, justPhone=True):
		if not self.children and self.labels[0] != "" and justPhone:
			return "|- "+str(self.labels[0])
		nodeString = "|- {"
		addComma = False
		for a in self.av_counts:
			if addComma:
				nodeString += ", "
			else:
				addComma = True
			nodeString += "'"+a+"': "
			if len(self.av_counts[a].keys()) == 1 and not self.av_counts[a].get('numerically_valued_attribute', 0) == 0:
				nodeString += str(round(self.av_counts[a].get('numerically_valued_attribute', 0)/self.count, 2))
			else:
				nodeString += "{"
				addCommaAgain = False
				for v in self.av_counts[a].keys():
					if addCommaAgain:
						nodeString += ", "
					else:
						addCommaAgain = True
					nodeString += "'"+str(v)+"' : "+str(self.av_counts[a][v])
				nodeString += "}"
		nodeString += "}"
		return nodeString + ":" + str(self.count)
	
	# find the maximum and minimum values for each attribute in the graph
	def getMinMax(self):
		if not self.children:
			return self.scaleAVCounts(), self.scaleAVCounts()
		minCounts = self.scaleAVCounts()
		maxCounts = self.scaleAVCounts()
		for c in self.children:
			minC, maxC = c.getMinMax()
			for a in minCounts.keys():
				if minC.get(a, 0) == 0:
					for v in minCounts[a].keys():
						minCounts[a][v] = 0
				else:
					for v in minCounts[a].keys():
						if minC[a].get(v,0) < minCounts[a][v]:
							minCounts[a][v] = minC[a].get(v,0)
						if not maxC.get(a,0) == 0 and maxC[a].get(v,0) > maxCounts[a][v]:
							maxCounts[a][v] = maxC[a].get(v,0)
		return minCounts, maxCounts
	
	# get a deep copy of the average attribute values
	def scaleAVCounts(self):
		scaled_av_counts = copy.deepcopy(self.av_counts)
		for a in scaled_av_counts.keys():
			for v in scaled_av_counts[a].keys():
				scaled_av_counts[a][v] = scaled_av_counts[a][v]/self.count
		return scaled_av_counts
	
	# get a string to represent subtrees on a tree map
	def treeString(self, averages=False):
		avgCounts = []
		if averages:
			avgCounts = self.scaleAVCounts()
		minCounts, maxCounts = self.getMinMax()
		treeString = ""
		for a in minCounts:
			treeString += a+": "
			if len(minCounts[a].keys()) == 1 and minCounts[a].get('numerically_valued_attribute', -1) > -1:
				treeString += "("+str(round(minCounts[a].get('numerically_valued_attribute', 0), 2))+","
				if averages:
					treeString += str(round(avgCounts[a].get('numerically_valued_attribute', 0), 2))+","
				treeString += str(round(maxCounts[a].get('numerically_valued_attribute', 0), 2))+")"
			else:
				treeString += "{"
				addComma = False
				for v in minCounts[a].keys():
					if addComma:
						treeString += ", "
					else:
						addComma = True
					treeString += "'"+str(v)+"' : "
					treeString += "("+str(minCounts[a][v])+","
					if averages:
						treeString += str(avgCounts[a][v])+","
					treeString += str(maxCounts[a][v])+")"
				treeString += "}"
			treeString += "\n"
		treeString += "Count: "+str(self.count)
		return treeString
	
	# draw a tree map of this tree
	def drawTreeMap(self, depth, averages=False, fileName="", valueDictionary={}):
		fig = pylab.figure(figsize=(7, 7), linewidth=0.001)
		ax = fig.add_subplot(111,aspect='equal')
		blocks = self.buildTreeMap(depth, averages, valueDictionary)
		for i in range(len(blocks)):
			r = Rectangle(blocks[i][0], blocks[i][1], blocks[i][2], facecolor=(1,1-blocks[i][5],1-blocks[i][5]), edgecolor='k')
			text = blocks[i][3].replace("|- ", "")
			text = text.replace("{", "")
			text = text.replace(", ", "\n")
			text = text.replace("}", "\ncount")
			ax.add_patch(r)
			ax.text(blocks[i][0][0]+.001, blocks[i][0][1]+.001, text, size=blocks[i][4])
		if fileName:
			extension = fileName[len(fileName)-3:len(fileName)]
			pylab.savefig(fileName, format=extension)
		else:
			pylab.show()
	
	# build a list of parameters for the rectangles on the tree map
	def buildTreeMap(self, depth, averages=False, valueDictionary={}, start=[0,0], width=1, height=1, vertSplit=True):
		# if we aren't going deeper than this subtree
		if depth == 0 or not self.children:
			# find the correct font size
			fontSize = 12
			minWidth = .4
			minHeight = .3
			if width < minWidth or height < minHeight: 
				fontSize = 8*min(width/minWidth, height/minHeight)
			# find the number of merges and splits for instances in this subtree
			value = 0
			for l in self.labels:
				if valueDictionary.get(l,-1) > 0:
					value += 1
			# the rectangle to represent this subtree
			return [[start, width, height, self.treeString(averages), fontSize, value/self.count]]
		# if we are going deeper than this subtree
		blocks = []
		cumPerc = 0
		# build the rectangles to represent each child
		for c in self.children:
			perc = c.count/self.count
			newBlocks = []
			if vertSplit:
				newStart = [start[0] + cumPerc*width, start[1]]
				newWidth = perc*width
				newBlocks = c.buildTreeMap(depth-1, averages, valueDictionary, newStart, newWidth, height, not vertSplit)
			else:
				newStart = [start[0], start[1] + cumPerc*height]
				newHeight = perc*height
				newBlocks = c.buildTreeMap(depth-1, averages, valueDictionary, newStart, width, newHeight, not vertSplit)
			for b in newBlocks:
				blocks.append(b)
			cumPerc += perc
		return blocks

	# make a dot file to represent this tree
	def toDot(self, fileName,withStrings=True, name_of_latest=None):
		f = open('./'+fileName, 'w+')
		number, dotString = self.toDotString(0,withStrings,name_of_latest)
		f.write("digraph ConceptTree {\n\tnode [fontsize=8];\n\tnull0 [shape=box, label=\""+str(self.delta_prod())+"\"];\n"+dotString+"}")
		f.close()

	# build the string for the interior of the dot file
	def toDotString(self, number,withStrings=True, name_of_latest=None):
		splitColor = ", width=.5, height=.5, color=red"
		mergeColor = ", width=.5, height=.5, color=yellow"
		bothColor = ", width=.5, height=.5, color=blue"
		nullColor = ""
		latestEdge = " [color=green]"
		nullEdge = ""
		dotString = ""
		idString = "null"+str(number)
		number += 1
		for i in range(len(self.children)):
			c = self.children[i]
			# find the color of each node
			colorString = nullColor
			for m_list in ConceptTree.mergedNodes:
				if c in m_list:
					colorString = mergeColor
			for s_list in ConceptTree.splitNodes:
				if c in s_list:
					if colorString == mergeColor:
						colorString = bothColor
					else:
						colorString = splitColor
			# find edge color
			edgeString = nullEdge
			if name_of_latest and name_of_latest in c.labels:
				edgeString = latestEdge
			# build the structure if we aren't at the leaves yet
			if c.children:
				dotString += "\tnull"+str(number)+" [shape=point"+colorString+"];\n"
				dotString += "\t"+idString+" -> null"+str(number)+" "+edgeString+";\n"
				number, childDot = c.toDotString(number, withStrings, name_of_latest)
				dotString += childDot
			# make leaves
			else:
				if withStrings:
					dotString += "\tstring"+str(number)+" [label=\""+c.labels[0].replace(" ","\\n")+"\""+colorString+"];\n"
					dotString += "\t"+idString+" -> string"+str(number)+" "+edgeString+";\n"
				else:
					dotString += "\tnull"+str(number)+" [shape=point"+colorString+"];\n"
					dotString += "\t"+idString+" -> null"+str(number)+" "+edgeString+";\n"
				number += 1
		return number, dotString
	
	def delta(self):
		if self.children == []:
			return {0:[0,1]}
		delta_dict = {0:[1,0]}
		for c in self.children:
			delta_c = c.delta()
			for k in delta_c.keys():
				delta_dict.setdefault(k+1, [0,0])
				delta_dict[k+1][0] += delta_c[k][0]
				delta_dict[k+1][1] += delta_c[k][1]
		return delta_dict
	
	def delta_prod(self):
		delta_dict = self.delta()
		total = [0, 0]
		for k in delta_dict.keys():
			total[0] += k*delta_dict[k][0]
			total[1] += k*delta_dict[k][1]
		return {'both':sum(total), 'nl':total[0], 'l':total[1]}
	
	def __str__(self):
		"""
		Converts the categorization tree into a string.
		"""
		ret = str(self.av_counts)
		for c in self.children:
			ret += "\n" + str(c)

		return ret

# add an instance to the cluster hierarchy and record how much it changes the tree
def addInc(t,index,instances):
	if t.count == 0:
		index = seedTree(t,index,instances)
		return index
	t.firstCobweb(instances[index].getAttributes(), instances[index].pretty_print(False))
	# Children are used to describe nodes so disapearing addresses are not an issue
	# This is a list of the children of each node which was merged
	mergedNodes = [x for x in ConceptTree.mergedNodes if not (x in ConceptTree.splitNodes)]
	# This is a list of the children of each node which was split
	splitNodes = [x for x in ConceptTree.splitNodes if not (x in ConceptTree.mergedNodes)]
	# The children of a merged node are one step deeper than the node that was merged
	instances[index].merges = [-1 if nodeList == [] else max([node.getDepth() for node in nodeList])-1 for nodeList in mergedNodes]
	# The children of a split node are not since they moved up after their parent was split
	instances[index].splits = [-1 if nodeList == [] else max([node.getDepth() for node in nodeList]) for nodeList in splitNodes]
	index += 1
	return index

# If this is the first instance of the tree, leave it as a single node
def seedTree(t,index,instances):
	t.increment_counts(instances[0].getAttributes(), instances[0].pretty_print(False))
	instances[0].merges = instances[0].splits = []
	index += 1
	return index

if __name__ == "__main__":
	t = ConceptTree()

	# Read in Data
	# The columns we are interested in (weight.oz, Volume.in3, Talk.hr, Standby.hr)
	startcol = 3
	endcol = 9

	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	names = []
	years = []
	instances = []

	reload(sys)
	sys.setdefaultencoding("utf-8")
	with open('prunedPhones.csv','r') as csvfile:
		reader = csv.reader(csvfile, delimiter=',', quotechar='"')
		Instance.properties = reader.next()
		Instance.properties = Instance.properties[startcol:endcol]
		for row in reader:
			i = Instance(np.array(row[startcol:endcol]).astype('float').tolist(), float(row[11]), str(row[1])+" "+str(row[2]))
			instances.append(i)
	
	minVals = instances[0].attributes[0:len(Instance.properties)]
	maxVals = instances[0].attributes[0:len(Instance.properties)]
	for inst in instances:
		i = 0
		for a in inst.attributes:
			if a < minVals[i]:
				minVals[i] = a
			if a > maxVals[i]:
				maxVals[i] = a
			i += 1
	
	minVals = np.array(minVals)
	maxVals = np.array(maxVals)
	
	for i in instances:
		i.attributes = ((np.array(i.attributes)-minVals)/(maxVals - minVals)).tolist()
	
	index = 0
	t = ConceptTree()

	end = 100
	while index < end:
		index = addInc(t,index,instances)
		t.toDot(str(index)+'.dot', name_of_latest=instances[index-1].pretty_print(False))
	
#	for prop in Instance.properties:
#		fig = pl.figure()
#		X = []
#		Y = []
#		S = []
#		for inst in instances:
#			X.append(inst.time)
#			Y.append(inst.getAttribute(prop))
#			# Put the splits and merges on an inverse exponential scale and find the maximum
#			splits = [math.exp(-(x ** 2) * .5) for x in inst.splits if not x == -1]
#			splits = max(splits) if not splits == [] else 0
#			merges = [math.exp(-(x ** 2) * .5) for x in inst.merges if not x == -1]
#			merges = max(merges) if not merges == [] else 0
#			S.append(max(splits, merges) * 10000)
#		S = np.sqrt(S).tolist()
#		pl.scatter(X,Y, s=1, color='gray', alpha=.5)
#		pl.scatter(X,Y, s=S, color='r')
#	pl.ylabel('$'+prop+'$')
#	pl.xlabel('$Year$')
#	pl.show()
#	# In case you want to save children to a file
#	#pl.savefig(prop+"_shallowestChange.png", format="png")
