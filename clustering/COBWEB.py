from U import Utility

class COBWEB:
	fileNumber = 0
	
	def __init__(self, tree):
		self.tree = tree
		self.utility = tree.utility
	
	def create_new_child(self,instance):
		"""
		Creates a new child (to the current node) with the counts initialized by
		the given instance. 
		"""
		new_child = self.tree.makeTree(self.tree.root, self.tree)
		new_child.utility.increment_counts(instance)
		self.tree.children.append(new_child)

	def create_child_with_current_counts(self):
		"""
		Creates a new child (to the current node) with the counts initialized by
		the current node's counts.
		"""
		self.tree.children.append(self.tree.makeTree(self.tree.root, self.tree, self.tree))

	def two_best_children(self,instance):
		"""
		Returns the indices of the two best children to incorporate the instance
		into in terms of category utility.

		input:
			instance: {a1: v1, a2: v2,...} - a hashtable of attr. and values. 
		output:
			(0.2,2),(0.1,3) - the category utility and indices for the two best
			children (the second tuple will be None if there is only 1 child).
		"""
		if len(self.tree.children) == 0:
			raise Exception("No children!")
		
		self.utility.increment_counts(instance)
		children_cu = []
		for i in range(len(self.tree.children)):
			self.tree.children[i].utility.increment_counts(instance)
			children_cu.append((self.utility.category_utility(self.tree.children[i]),i))
			self.tree.children[i].utility.decrement_counts(instance)
		self.utility.decrement_counts(instance)
		children_cu.sort(reverse=True)

		if len(self.tree.children) == 1:
			return children_cu[0], None 

		return children_cu[0], children_cu[1]

	def new_child(self,instance):
		"""
		Updates root count and adds child -- permenant.
		"""
		return self.cu_for_new_child(instance,False)

	def cu_for_new_child(self,instance,undo=True):
		"""
		Returns the category utility for creating a new child using the
		particular instance.
		"""
		self.utility.increment_counts(instance)
		self.create_new_child(instance)
		cu = self.utility.category_utility()
		if undo:
			self.tree.children.pop()
			self.utility.decrement_counts(instance)
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

		first_c = self.tree.children[first]
		second_c = self.tree.children[second]

		new_c = self.tree.makeTree(self.tree.root, self.tree)
		new_c.utility.update_counts_from_node(first_c)
		new_c.utility.update_counts_from_node(second_c)

		self.tree.children.pop(second)
		self.tree.children.pop(first)
		self.tree.children.append(new_c)

		cu = self.utility.category_utility()

		if undo:
			self.tree.children.pop()
			self.tree.children.insert(first,first_c)
			self.tree.children.insert(second,second_c)
		else:
			# If we aren't undoing the merge then we have to add the leaves
			new_c.children.append(first_c)
			first_c.parent = new_c
			new_c.children.append(second_c)
			second_c.parent = new_c
			self.tree.mergedNodes.append([first_c, second_c])
			self.tree.splitMergeOrder.append("m")
			for m in self.tree.mergedNodes:
				if len(m) > 2 and first_c in m and second_c in m:
					m.remove(first_c)
					m.remove(second_c)
					m.append(new_c)
			for s in self.tree.splitNodes:
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
		oldChildren = self.tree.children[0:len(self.tree.children)]
		best_c = self.tree.children.pop(best)
		for child in best_c.children:
			self.tree.children.append(child)
		cu = self.utility.category_utility()

		if undo:
			for i in range(len(best_c.children)):
				self.tree.children.pop()
			self.tree.children.insert(best,best_c)
		else:
			self.tree.splitNodes.append(best_c.children)
			self.tree.splitMergeOrder.append("s")
			for child in best_c.children:
				child.parent = self.tree
			for m in self.tree.mergedNodes:
				if best_c in m:
					m.remove(best_c)
					for c in best_c.children:
						m.append(c)
			for s in self.tree.splitNodes:
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
		if len(self.tree.children) == 0:
			return

		child_count = 0.0
		for child in self.tree.children:
			child_count += child.utility.count
		assert self.utility.count == child_count

	def is_instance(self,instance):
		"""
		Checks to see if the current node perfectly represents the instance (all
		of the attribute values the instance has are probability 1.0 and here
		are no extra attribute values).
		"""
		inst_attributes = instance.getAttributes()
		for attribute in self.utility.av_counts.keys():
			if attribute not in inst_attributes:
				return False
			if type(inst_attributes[attribute]) == dict:
				for value in self.utility.av_counts[attribute]:
					if (self.utility.av_counts[attribute][value] / self.utility.count) != 1.0:
						return False
					if inst_attributes[attribute] != value:
						return False
			else:
					if inst_attributes[attribute] != self.utility.av_counts[attribute]['numerically_valued_attribute'] / self.utility.count:
						return False
		
		for attribute in instance:
			if attribute not in self.utility.av_counts:
				return False
			if type(inst_attributes[attribute]) == dict:
				if inst_attributes[attribute] not in self.utility.av_counts[attribute]:
					return False
				if ((self.utility.av_counts[attribute][inst_attributes[attribute]] / self.utility.count) != 1.0):
					return False
			else:
				if len(self.utility.av_counts[attribute].keys()) != 1 or self.utility.av_counts[attribute].get('numerically_valued_attribute', 0) == 0:
					return False
		
		return True

	def closest_matching_child(self,instance):
		"""
		Returns the child that is the best match for the instance in terms of
		difference between attribute value probabilites (note the instance has
		probability 1 of all attribute values it possesses). This function is
		used when the category utility of all actions is 0. It is a secondary
		heuristic for deciding the best node to add to.
		"""
		best = 0
		smallest_diff = float('inf')
		inst_attributes = instance.getAttributes()
		for i in range(len(self.tree.children)):
			child = self.tree.children[i]
			sum_diff = 0.0
			count = 0.0
			for attribute in child.utility.av_counts:
				for value in self.utility.av_counts[attribute]:
					count += 1
					if value == 'numerically_valued_attribute':
						sum_diff += inst_attributes[attribute] - (self.utility.av_counts[attribute][value][0] / self.utility.count)
					else:
						if attribute in instance and inst_attributes[attribute] == value:
							sum_diff += 1.0 - (self.utility.av_counts[attribute][value][0] / self.utility.count)
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
	
	def firstCobweb(self, instance, display=False):
		Utility.updatedNodes = []
		self.cobweb(instance, display)
	
	def cobweb(self, instance, display=False):
		"""
		Incrementally integrates an instance into the categorization tree
		defined by the current node. This function operates recursively to
		integrate this instance and uses category utility as the heuristic to
		make decisions.
		"""
		#if not self.children and self.is_instance(instance): 
		#	self.utility.increment_counts(instance)

		if not self.tree.children:
			self.create_child_with_current_counts()
			self.utility.increment_counts(instance)
			self.create_new_child(instance)
			if display:
				# print status
				self.tree.root.viz.toDot(str(COBWEB.fileNumber)+'.dot', latest=instance)
				COBWEB.fileNumber += 1
			
		else:
			best1, best2 = self.two_best_children(instance)
			operations = []
			operations.append((best1[0],"best"))
			operations.append((self.cu_for_new_child(instance),'new'))
			# a nodes only two children want to merge and split when they are exactly the same, leading to problems
			if best2 and len(self.tree.children) > 2 and not [self.tree.children[best1[1]], self.tree.children[best2[1]]] in self.tree.mergedNodes:
				operations.append((self.cu_for_merge(best1[1],best2[1]),'merge'))
			if len(self.tree.children[best1[1]].children) and not self.tree.children[best1[1]].children in self.tree.splitNodes:
				operations.append((self.cu_for_split(best1[1]),'split'))
			operations.sort(reverse=True)
			
			if display:
				# print status
				self.tree.instances.append(instance)
				self.tree.root.viz.toDot(str(COBWEB.fileNumber)+'.dot', latest=instance)
				COBWEB.fileNumber += 1
				self.tree.instances.pop()
			
			best_action = operations[0][1]
			action_cu = operations[0][0]
			if action_cu == 0.0:
				self.utility.increment_counts(instance)
				self.tree.children[self.closest_matching_child(instance)].cobweb(instance)
			elif best_action == 'best':
				self.utility.increment_counts(instance)
				self.tree.children[best1[1]].cobweb(instance)
			elif best_action == 'new':
				self.new_child(instance)
			elif best_action == 'merge':
				self.merge(best1[1],best2[1])
				while len(Utility.updatedNodes) > 0:
					Utility.updatedNodes[0].utility.decrement_counts(instance)
				self.tree.root.cobweb(instance)
			elif best_action == 'split':
				self.split(best1[1])
				while len(Utility.updatedNodes) > 0:
					Utility.updatedNodes[0].utility.decrement_counts(instance)
				self.tree.root.cobweb(instance)
			else:
				raise Exception("Should never get here.")
