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
from Instance import Instance

numerical_key = 'numerically_valued_attribute'
class Utility:
	updatedNodes = []

	def __init__(self, tree):
		"""
		The constructor.
		"""
		self.tree = tree
		self.av_counts = {}
		self.count = 0
	
	def increment_counts(self, instance):
		"""
		Increment the counts at the current node according to the specified
		instance.

		input:
			instance: {a1: v1, a2: v2, ...} - a hashtable of attr and values. 
		"""
		Utility.updatedNodes.append(self.tree)
		self.tree.instances.append(instance)
		self.count += 1.0 
		attributes = instance.getAttributes()
		for a in attributes:
			self.av_counts[a] = self.av_counts.setdefault(a,{})
			index = attributes[a]
			amount = 1.0
			amount_2 = 1.0
			if type(attributes[a]) == int or type(attributes[a]) == float:
				index = numerical_key
				amount = float(attributes[a])
				amount_2 = float(attributes[a] ** 2)
			self.av_counts[a][index] = (self.av_counts[a].get(index, np.array([0,0])) + [amount, amount_2])

	def decrement_counts(self, instance):
		"""
		Decrement the counts at the current node according to the specified
		instance.
		
		input:
			instance: {a1: v1, a2: v2, ...} - a hashtable of attr and values. 
		"""
		Utility.updatedNodes.remove(self.tree)
		self.tree.instances.remove(instance)
		self.count -= 1.0
		attributes = instance.getAttributes()
		for a in attributes:
			self.av_counts[a] = self.av_counts.setdefault(a,{})
			index = attributes[a]
			amount = 1.0
			amount_2 = 1.0
			if type(attributes[a]) == int or type(attributes[a]) == float:
				index = numerical_key
				amount = attributes[a]
				amount_2 = float(attributes[a] ** 2)
			self.av_counts[a][index] = (self.av_counts[a].get(index, np.array([0,0])) - [amount, amount_2])
			# for clarity in printing we remove the values and attributes
			if not index == numerical_key and (self.av_counts[a][index] == np.array([0,0])).all():
				del self.av_counts[a][index]
			if self.av_counts[a] == {}:
				del self.av_counts[a]
	
	def update_counts_from_node(self, node):
		"""
		Increments the counts of the current node by the amount in the specified
		node.
		"""
		self.count += node.utility.count
		for inst in node.instances:
			self.tree.instances.append(inst)
		for a in node.utility.av_counts:
			for v in node.utility.av_counts[a]:
				self.av_counts[a] = self.av_counts.setdefault(a,{})
				self.av_counts[a][v] = (self.av_counts[a].get(v,np.array([0,0])) + node.utility.av_counts[a][v])
	
	def mean_std(self, attribute, value=numerical_key, scaled=True):
		sum_1 = self.av_counts[attribute][value][0]
		sum_2 = self.av_counts[attribute][value][0]
		mean = sum_1 / self.count
		var = sum_2/self.count - (mean ** 2)
		std = np.sqrt(var)
		if scaled:
			r_mean, r_std = self.tree.root.utility.mean_std(attribute, value, False)
			mean -= r_mean
			std /= r_std
		return mean, std
	
	def category_utility(self):
		"""
		The category utility is a local heuristic calculation to determine if
		the split of instances across the children increases the ability to
		guess from the parent node. 
		"""
		if len(self.tree.children) == 0:
			return 0.0

		category_utility = 0.0

		exp_parent_guesses = self.expected_correct_guesses()

		for child in self.tree.children:
			p_of_child = child.utility.count / self.count
			exp_child_guesses = child.utility.expected_correct_guesses()
			category_utility += p_of_child * (exp_child_guesses - exp_parent_guesses)

		# return the category utility normalized by the number of children.
		return category_utility / (1.0 * len(self.tree.children))
	
	def mean_ecg(self):
		exp_count = 0.0
		for attribute in self.av_counts:
			for value in self.av_counts[attribute]:
				mean, std = self.mean_std(attribute, value)
				exp_count += mean ** 2
		return exp_count
	
	def expected_correct_guesses(self):
		"""
		The number of attribute value guesses we would be expected to get
		correct using the current concept.
		"""
		return self.mean_ecg()
	
	def __str__(self):
		avString = "|- {"
		addComma = False
		for a in self.av_counts:
			if addComma:
				avString += ", "
			else:
				addComma = True
			avString += "'"+a+"': "
			if len(self.av_counts[a].keys()) == 1 and not self.av_counts[a].get(numerical_key, 0) == 0:
				avString += str(round(self.av_counts[a].get(numerical_key, 0)/self.count, 2))
			else:
				avString += "{"
				addCommaAgain = False
				for v in self.av_counts[a].keys():
					if addCommaAgain:
						avString += ", "
					else:
						addCommaAgain = True
					avString += "'"+str(v)+"' : "+str(self.av_counts[a][v])
				avString += "}"
		avString += "}"
		return avString + ":" + str(self.count)
