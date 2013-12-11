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
import heapq
from matplotlib import pyplot as pl
from matplotlib import widgets
import colorsys
import inspect
import csv
from I import Instance

# Strings for creating a .dot file
splitColor = ", width=.5, height=.5, color=red"
mergeColor = ", width=.5, height=.5, color=yellow"
bothColor = ", width=.5, height=.5, color=blue"
nullColor = ""
latestEdge = " [color=green]"
nullEdge = ""
descriptiveString = "\tmergeBox [label=\"Merge\" "+mergeColor+"];\n\tsplitBox [label=\"Split\" "+splitColor+"];\n\tbothBox [label=\"Both\" "+bothColor+"];\n"

class Visualization:
	plotted_viz = None
	
	def __init__(self, tree, directory="."):
		"""
		The constructor.
		"""
		self.tree = tree
		self.directory = directory+"/"
		# used for interactive surprise plot
		self.last_tag = None
		self.surprise_list = []
		self.graph = None
	
	def pretty_print(self,depth=0):
		"""
		Prints the categorization tree.
		"""
		for i in range(depth):
			print "\t",
				
		print self.__str__()
		
		for c in self.tree.children:
			c.viz.pretty_print(depth+1)
	
	# make a csv file of all surprise calculations
	def toCSV(self, filename):
		inst_time_list = [inst.time for inst in self.tree.instances]
		time_list = list(set(inst_time_list))
		time_list.sort()
		surprise_list = range(len(time_list))
		for i in range(len(time_list)):
			weights = self.weightFunction(time_list[i], inst_time_list, 0.15)
			numerator = 0
			for j in range(len(self.tree.instances)):
				if weights[j] > 0:
					numerator += self.tree.instances[j].surprise_num * weights[j]
			surprise_list[i] = numerator/sum(weights)
		
		with open(filename, 'wb') as csvfile:
			writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
			writer.writerow(['Name', 'Release Date', 'Surprise'])
			for inst in self.tree.instances:
				index = time_list.index(inst.time)
				writer.writerow([inst.name, inst.time, inst.surprise_num, inst.surprise_num/surprise_list[index]])
	
	# A line showing average surprise over time
	def surpriseLine(self):
		inst_time_list = [inst.time for inst in self.tree.instances]
		time_list = list(set(inst_time_list))
		time_list.sort()
		surprise_list = range(len(time_list))
		for i in range(len(time_list)):
			weights = self.weightFunction(time_list[i], inst_time_list, 0.15)
			numerator = 0
			for j in range(len(self.tree.instances)):
				if weights[j] > 0:
					numerator += self.tree.instances[j].surprise_num * weights[j]
			surprise_list[i] = numerator/sum(weights)
		
		fig = pl.figure()
		ax = fig.add_subplot(111)
		self.graph = ax
		ax.plot(time_list, surprise_list)
		pl.ylabel('Surprise')
		pl.xlabel("Time")
		pl.show()
	
	# A helper function for surpriseLine
	def weightFunction(self, value, comparisonList, weightFactor):
		return np.exp(weightFactor*(-(abs(np.array(comparisonList)-value) ** 2)))
	
	# A simple plot of the clusters along two dimensions made by cutting off the tree at some depth
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
	
	# A helper function for plotClusters
	def getColors(self, number):
		colors = []
		for i in range(number):
			new_col = colorsys.hsv_to_rgb(float(i)/number, 1, 1)
			colors.append(new_col)
		return colors
	
	# A short description to identify the node in the dot file
	def shortDescription(self, number=0, numChildren=True):
		if self.tree in self.tree.util_values:
			utility = round(self.tree.util_values[self.tree],4)
		else:
			utility = None
		if numChildren:
			return (number, self.tree.measure.numberLeaves(), utility)
		return (number, utility)
	
	# A helper function for toDot
	def describeNode(self, instance=None, numDescriptors=3, largest=True, numChildren=True):
		description = {'parent':[], 'instance':[], 'desc':""}
		if self.tree in self.tree.util_values:
			description['desc'] = str(len(self.tree.instances))+" elements\\n"+"("+str(round(self.tree.util_values[self.tree],4))+")\\n"
		if not self.tree.parent is None:
			# define difference in relation to parent
			diff_av = {}
			parent_av = self.tree.parent.utility.get_av_counts()
			child_av = self.tree.utility.get_av_counts()
			for a in child_av.keys():
				diff_av[round(parent_av[a]-child_av[a],2)] = a
			absVal = [abs(v) for v in diff_av]
			toSort = zip(absVal, diff_av, child_av.keys())
			toSort.sort(reverse=True)
			description['parent'] = [a+":"+str(d) for (_,d,a) in toSort]
		if not instance is None:
			if self.tree.instances == [instance]:
				description['instance'] = [instance.pretty_print(False).replace(" ","\\n")]
			else:
				# define difference in relation to latest instance
				diff_av = {}
				child_av = self.tree.utility.get_av_counts()
				for a in child_av:
					diff_av[round(instance.getAttribute(a)-child_av[a],2)] = a
				absVal = [abs(v) for v in diff_av]
				toSort = zip(absVal, diff_av, child_av.keys())
				toSort.sort(reverse=True)
				description['instance'] = [a+":"+str(d) for (_,d,a) in toSort]
		return description
	
	# make a dot file to represent this tree
	def toDot(self, fileName, withStrings=True, latest=None):
		if not latest is None:
			latest.dot_descriptions = []
		node_id, dotString = self.toDotString(0, withStrings,latest)
		stringToWrite = "digraph ConceptTree {\n\tnode [fontsize=8];\n\tnull0 [shape=box, label=\""
		stringToWrite += str(round(self.tree.measure.delta_num(latest),2))+"\"];\n"
		if len(self.tree.mergedNodes) + len(self.tree.splitNodes) > 0:
			stringToWrite += descriptiveString
		stringToWrite += dotString+"}"
		if not latest is None:
			latest.dot_string = stringToWrite
			return
	
	# build the string for the interior of the dot file
	def toDotString(self, number, withStrings=True, latest=None):
		dotString = ""
		idString = "null"+str(number)
		for i in range(len(self.tree.children)):
			# increment the number so each child gets a new id
			number += 1
			c = self.tree.children[i]
			# find the color of each node
			colorString = nullColor
			for m_list in self.tree.mergedNodes:
				if c in m_list:
					colorString = mergeColor
					break
			for s_list in self.tree.splitNodes:
				if c in s_list:
					if colorString == mergeColor:
						colorString = bothColor
						break
					else:
						colorString = splitColor
						break
			# find edge color
			edgeString = nullEdge
			if latest and latest in c.instances:
				edgeString = latestEdge
			# build the structure if we aren't at the leaves yet
			if c.children and (not latest or latest in c.instances):
				description = c.viz.describeNode(instance=latest)
				if not latest is None:
					latest.dot_descriptions.append(description)
				dotString += "\tnull"+str(number)+" [label=\"REPLACE_STRING\""+colorString+"];\n"
				dotString += "\t"+idString+" -> null"+str(number)+" "+edgeString+";\n"
				number, childDot = c.viz.toDotString(number, withStrings, latest)
				dotString += childDot
			# make leaves
			else:
				if not c.children:
					if withStrings or (latest and latest in c.instances):
						description = c.viz.describeNode(instance=latest)
						if not latest is None:
							latest.dot_descriptions.append(description)
						dotString += "\tstring"+str(number)+" [label=\"REPLACE_STRING\""+colorString+"];\n"
						dotString += "\t"+idString+" -> string"+str(number)+" "+edgeString+";\n"
					else:
						description = c.viz.describeNode(instance=latest)
						if not latest is None:
							latest.dot_descriptions.append(description)
						dotString += "\tstring"+str(number)+" [label=\"REPLACE_STRING\""+colorString+"];\n"
						dotString += "\t"+idString+" -> string"+str(number)+" "+edgeString+";\n"
				else:
					description = c.viz.describeNode(instance=latest)
					if not latest is None:
						latest.dot_descriptions.append(description)
					dotString += "\tstring"+str(number)+" [label=\"REPLACE_STRING\""+colorString+"];\n"
					dotString += "\t"+idString+" -> string"+str(number)+" "+edgeString+";\n"
		return number, dotString
	
	# An interactive plot for Examinining Surprise
	def plotSurprise(self, time=True):
		self.surprise_list = [inst.surprise_num for inst in self.tree.instances]
		if time:
			self.inst_time_list = [inst.time for inst in self.tree.instances]
		else:
			self.inst_time_list = range(len(self.surprise_list))
		Visualization.plotted_viz = self
		
		fig = pl.figure()
		ax = fig.add_subplot(111)
		self.graph = ax
		ax.scatter(self.inst_time_list, self.surprise_list)
		#fig.subtitle("Click a dot to see it's explanation.  Press a number to change the number of attributes shown and press 'p' to toggle the measure.")
		pl.ylabel('Surprise')
		pl.xlabel("number attributes shown:3 measured from instance")
		
		callback = PlotManager()
		
		fig.canvas.mpl_connect('button_press_event', callback.onclick)
		fig.canvas.mpl_connect('key_press_event', callback.onpress)
		
		pl.show()
	
	def __str__(self, justPhone=True):
		if not self.tree.children and justPhone:
			return "|- "+str(self.tree.instances[0].pretty_print(False))
		return str(self.tree.utility)

class PlotManager:
	parent = False
	number = 3
	
	def onpress(self, event):
		try:
			num = int(event.key)
			self.number = num
		except ValueError:
			if event.key == "p":
				self.parent = not self.parent
		labelString = "number attributes shown:"+str(self.number)+" measured from "
		if self.parent:
			labelString += "parent"
		else:
			labelString += "instance"
		pl.xlabel(labelString)
		pl.draw()
	
	def onclick(self, event):
		# check for right click
		if event.button == 3:
			self.closestElement(event.xdata, event.ydata)

	def closestElement(self, x, y):
		if not Visualization.plotted_viz.last_tag is None:
			Visualization.plotted_viz.last_tag.remove()
			Visualization.plotted_viz.last_tag = None
		surprise = Visualization.plotted_viz.surprise_list
		time = Visualization.plotted_viz.inst_time_list
		if len(surprise) == 0 or x is None or y is None:
			return
		range_x = max(time) - min(time)
		range_y = max(surprise) - min(surprise)
		dist = [float(abs(x - time[i]))/range_x + float(abs(y-surprise[i]))/range_y for i in range(len(surprise))]
		index = dist.index(min(dist))
		Visualization.plotted_viz.last_tag = Visualization.plotted_viz.graph.annotate(
			Visualization.plotted_viz.tree.instances[index].pretty_print(False),
			xy = (time[index], surprise[index]), xytext = (-20, 20),
			textcoords = 'offset points', ha = 'right', va = 'bottom',
			bbox = dict(boxstyle = 'round,pad=0.5', fc = 'white', alpha = 1.0),
			arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
		pl.draw()
		Visualization.plotted_viz.tree.instances[index].render(self.parent, self.number)
