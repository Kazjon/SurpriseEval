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
import copy

numerical_key = 'numerically_valued_attribute'
class Utility:
	updatedNodes = []

	def __init__(self, tree, constructor=None):
		"""
		The constructor.
		"""
		self.tree = tree
		self.av_counts = {}
		self.count = 0
		self.sq_count = 0
		if not constructor is None:
			self.av_counts = copy.deepcopy(constructor.av_counts)
			self.count = constructor.count
			self.sq_count = constructor.sq_count
	
	def get_av_counts(self, unscale_with_parser=None):
		simple_av = {}
		for a in self.av_counts:
			if numerical_key in self.av_counts[a]:
				simple_av[a] = self.av_counts[a][numerical_key][0]/self.count
				if not unscale_with_parser is None:
					parser = unscale_with_parser
					simple_av[a] * parser.pastCalc[a]['std']
					simple_av[a] + parser.pastCalc[a]['avg']
		return simple_av
	
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
	
	def category_utility(self, recordToNode=None):
		answer = self.leoski_utility_function()
		if recordToNode:
			self.tree.util_values[recordToNode] = answer
		return answer
	
	""" Leoski Utility Functions """
	def leoski_utility_function(self):
		if len(self.tree.children) == 0:
			return 0.0

		category_utility = 0.0

		for child in self.tree.children:
			p_of_child = child.utility.count / self.count
			exp_child_separation = child.utility.leoski_attribute_separation()
			category_utility += p_of_child * exp_child_separation

		# return the category utility normalized by the number of children.
		return category_utility / (1.0 * len(self.tree.children))

	def leoski_attribute_separation(self):
		exp_count = 0.0
		for attribute in self.av_counts:
			for value in self.av_counts[attribute]:
				parent_mean, parent_std = list(self.tree.parent.utility.av_counts[attribute][numerical_key]/self.tree.parent.utility.count)
				parent_std -= parent_mean ** 2
				child_mean, child_std = list(self.av_counts[attribute][numerical_key]/self.count)
				child_std -= child_mean ** 2
				if value == numerical_key:
					mean_diff = abs(parent_mean - child_mean)
					std_diff = parent_std - child_std
					exp_count += (mean_diff) * (std_diff)
				else:
					exp_count += (child_mean)**2 - (parent_mean)**2
		return exp_count
	""" END: Leoski Utility Functions """
	
	""" STD Utility Functions """
	def std_utility_function(self):
		if len(self.tree.children) == 0:
			return 0.0

		category_utility = 0.0

		for child in self.tree.children:
			p_of_child = child.utility.count / self.count
			exp_child_separation = child.utility.std_attribute_separation()
			category_utility += p_of_child * exp_child_separation

		# return the category utility normalized by the number of children.
		return category_utility / (1.0 * len(self.tree.children))

	def std_attribute_separation(self):
		exp_count = 0.0
		for attribute in self.av_counts:
			for value in self.av_counts[attribute]:
				parent_mean, parent_std = list(self.tree.parent.utility.av_counts[attribute][numerical_key]/self.tree.parent.utility.count)
				parent_std -= parent_mean ** 2
				child_mean, child_std = list(self.av_counts[attribute][numerical_key]/self.count)
				child_std -= child_mean ** 2
				if value == numerical_key:
					std_diff = parent_std - child_std
					exp_count += (std_diff)
				else:
					exp_count += (child_mean)**2 - (parent_mean)**2
		return exp_count
	""" END: STD Utility Functions """
	
	""" Mean Utility Functions """
	def mean_utility_function(self):
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
		return self.mean_ecg()
	
	""" END: Mean Utility Functions """
	
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
