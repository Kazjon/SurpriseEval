
class Measure:
	def __init__(self, tree):
		self.tree = tree
	
	def delta(self):
		if self.tree.children == []:
			return {0:[0,1]}
		delta_dict = {0:[1,0]}
		for c in self.tree.children:
			delta_c = c.measure.delta()
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
	
	def delta_num(self, latest):
		depth = self.getInstanceDepth(latest)
		if depth == 0:
			return 0.0
		number_of_changes = len(self.tree.mergedNodes) + len(self.tree.splitNodes)
		return float(number_of_changes + 1)/depth
	
	def depth_only(self, latest):
		depth = self.getInstanceDepth(latest)
		if depth == 0:
			return 0.0
		return 1.0/depth
	
	def getDepth(self):
		if self.tree.root == self.tree or self.tree.parent == None:
			return 0
		return 1 + self.tree.parent.measure.getDepth()
	
	def getInstanceDepth(self, instance):
		for c in self.tree.children:
			if instance in c.instances:
				return c.measure.getInstanceDepth(instance)
		return self.getDepth()
	
	def deepestChild(self):
		if self.tree.children == []:
			return self.getDepth()
		maxDepth = 0
		for c in self.tree.children:
			dc = c.measure.deepestChild()
			if dc > maxDepth:
				maxDepth = dc
		return maxDepth
	
	def numberLeaves(self):
		return len(self.tree.instances)
