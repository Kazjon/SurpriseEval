import pydot, pylab, scipy.misc

class Instance:
	properties = []
	pastCalc = {}
	fig = None
	
	def __init__(self, a, t, n):
		self.attributes = a
		self.time = t
		self.name = n
		# Information maintained to manage surprise
		self.merges = []
		self.splits = []
		self.splitMergeStory = []
		self.depth = 0
		self.before_delta = {}
		self.after_delta = {}
		self.dot_string = ""
		self.surprise_num = 0
		self.dot_descriptions = []
		self.properties = Instance.properties
		self.pastCalc = Instance.pastCalc
	
	def __repr__(self):
		return self.pretty_print(False)
	
	def __str__(self):
		return self.pretty_print(True)
	
	def __lt__(self, inst):
		return self.time < inst.time
	
	def changes(self, numElements):
		num_changes = 0
		for change in self.splitMergeStory:
			num_changes += float(self.numElementsInChange(change))/numElements
		return num_changes
	
	def numElementsInChange(self, change):
		story = change[4:-2]
		story_list = story.split("), (")
		num_elements = 0
		for s in story_list:
			num_elements += int(s.split(", ")[1])
		return num_elements
	
	def render(self, mode='instance', number=3, unscale=False):
		if self.dot_string == "":
			return
		# Print out the merges and splits for this node
		print self.pretty_print(False), self.surprise_num
		self.describe_merges_and_splits()
		# make picture and display
		# Render dot file and save
		dot_list = self.dot_string.split("REPLACE_STRING")
		string = dot_list[0]
		for i in range(len(dot_list)-1):
			string += self.dot_descriptions[i]['desc']
			if number == 0:
				number = 1000
			for j in range(number):
				if j >= len(self.dot_descriptions[i][mode]):
					break
				if unscale:
					attribute = self.dot_descriptions[i][mode][j]
					att_name = attribute.split(":")[0]
					try:
						att_num = float(attribute.split(":")[1])
						att_num *= self.pastCalc[att_name]['std']
						if mode == "no_compare":
							att_num += self.pastCalc[att_name]['avg']
						att_num = round(att_num,2)
						string += att_name+":"+str(att_num)+"\\n"
					except IndexError:
						string += self.dot_descriptions[i][mode][j]+"\\n"
				else:
					string += self.dot_descriptions[i][mode][j]+"\\n"
			string += dot_list[i+1]
		dot_file = open('tmp.dot', 'w+')
		dot_file.write(string)
		dot_file.close()
		# Read dot to graph and render
		graph = pydot.graph_from_dot_file('tmp.dot')
		graph.write_png('tmp.png')
		graph_image = scipy.misc.imread('tmp.png')
		if not Instance.fig is None:
			pylab.close(Instance.fig)
			Instance.fig = None
		Instance.fig = pylab.figure(frameon=False)
		ax_size = [0,0,1,1]
		Instance.fig.add_axes(ax_size)
		pylab.imshow(graph_image, vmin=1, vmax=99, origin='upper')
		pylab.axis('off')
		pylab.show()
	
	def save_merges_and_splits(self, merges, splits, order):
		counter = 0
		mapping = {}
		for changes in [merges, splits]:
			for sub_list in changes:
				for node in sub_list:
					if not node in mapping:
						mapping[node] = counter
						counter += 1
		self.merges = [[n.viz.shortDescription(number=mapping[n]) for n in l] for l in merges]
		self.splits = [[n.viz.shortDescription(number=mapping[n]) for n in l] for l in splits]
		self.splitMergeStory = []
		mergeIndex = 0
		splitIndex = 0
		for change in order:
			if change == "s":
				self.splitMergeStory.append(change + ":"+str(self.splits[splitIndex]))
				splitIndex += 1
			else:
				self.splitMergeStory.append(change + ":"+str(self.merges[mergeIndex]))
				mergeIndex += 1
	
	def describe_merges_and_splits(self):
		print '\n'.join(self.splitMergeStory)
	
	def pretty_print(self, showAttributes=True):
		printString = self.name+" "+str(round(self.time, 2))
		if showAttributes:
			printString += ":"+str(self.getAttributes())
		return printString
	
	def __lt__(self, other):
		return self.time < other.time
	
	def getAttribute(self, a):
		if type(a) == str:
			return self.attributes[self.properties.index(a)]
		if type(a) == int:
			return self.attributes[a]
	
	def getAttributes(self):
		if not (len(self.attributes) == len(self.properties)):
			print "ERROR!: "+str(len(self.attributes))+", "+str(len(self.properties))
			return None
		a = {}
		for i in range(len(self.properties)):
			a[self.properties[i]] = self.attributes[i]
		return a
