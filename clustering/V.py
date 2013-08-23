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

import numpy as np
from matplotlib import pyplot as pl
import colorsys
from I import Instance

# Strings for creating a .dot file
splitColor = ", width=.5, height=.5, color=red"
mergeColor = ", width=.5, height=.5, color=yellow"
bothColor = ", width=.5, height=.5, color=blue"
nullColor = ""
latestEdge = " [color=green]"
nullEdge = ""

class Visualization:
	def __init__(self, tree):
		"""
		The constructor.
		"""
		self.tree = tree
	
	def pretty_print(self,depth=0):
		"""
		Prints the categorization tree.
		"""
		for i in range(depth):
			print "\t",
				
		print self.__str__()
		
		for c in self.tree.children:
			c.viz.pretty_print(depth+1)
	
	def plotClusters(self, x, y, depth=1):
		parent_queue = [self.tree]
		cluster_list = []
		while len(parent_queue) > 0:
			node = parent_queue.pop()
			if node.measure.getDepth() == depth or len(node.children) == 0:
				cluster_list.append(node)
			else:
				for c in node.children:
					parent_queue.append(c)
		colors = self.getColors(len(cluster_list))
		for i in range(len(cluster_list)):
			cluster = cluster_list[i]
			X = []
			Y = []
			S = []
			for inst in cluster.instances:
				attributes = inst.getAttributes()
				if x in Instance.properties:
					X.append(attributes[x])
				else:
					X.append(inst.time)
				if y in Instance.properties:
					Y.append(attributes[y])
				else:
					Y.append(inst.time)
				S.append(inst.depth)
			S = (((max(S)-np.array(S)) ** 2) * 30).tolist()
			pl.scatter(X, Y, color=colors[i], s=S)
		pl.xlabel(x)
		pl.ylabel(y)
		pl.show()
	
	def getColors(self, number):
		colors = []
		for i in range(number):
			new_col = colorsys.hsv_to_rgb(float(i)/number, 1, 1)
			colors.append(new_col)
		return colors
	
	# make a dot file to represent this tree
	def toDot(self, fileName,withStrings=True, latest=None):
		f = open('./'+fileName, 'w+')
		number, dotString = self.toDotString(0,withStrings,latest)
		f.write("digraph ConceptTree {\n\tnode [fontsize=8];\n\tnull0 [shape=box, label=\""+str(self.tree.measure.delta_prod())+"\"];\n"+dotString+"}")
		f.close()
	
	# build the string for the interior of the dot file
	def toDotString(self, number,withStrings=True, latest=None):
		dotString = ""
		idString = "null"+str(number)
		number += 1
		for i in range(len(self.tree.children)):
			c = self.tree.children[i]
			# find the color of each node
			colorString = nullColor
			for m_list in self.tree.mergedNodes:
				if c in m_list:
					colorString = mergeColor
			for s_list in self.tree.splitNodes:
				if c in s_list:
					if colorString == mergeColor:
						colorString = bothColor
					else:
						colorString = splitColor
			# find edge color
			edgeString = nullEdge
			if latest and latest in c.instances:
				edgeString = latestEdge
			# build the structure if we aren't at the leaves yet
			if c.children:
				dotString += "\tnull"+str(number)+" [shape=point"+colorString+"];\n"
				dotString += "\t"+idString+" -> null"+str(number)+" "+edgeString+";\n"
				number, childDot = c.viz.toDotString(number, withStrings, latest)
				dotString += childDot
			# make leaves
			else:
				if withStrings:
					dotString += "\tstring"+str(number)+" [label=\""+c.instances[0].pretty_print(False).replace(" ","\\n")+"\""+colorString+"];\n"
					dotString += "\t"+idString+" -> string"+str(number)+" "+edgeString+";\n"
				else:
					dotString += "\tnull"+str(number)+" [shape=point"+colorString+"];\n"
					dotString += "\t"+idString+" -> null"+str(number)+" "+edgeString+";\n"
				number += 1
		return number, dotString
	
	def __str__(self, justPhone=True):
		if not self.tree.children and justPhone:
			return "|- "+str(self.tree.instances[0].pretty_print(False))
		return str(self.tree.utility)
