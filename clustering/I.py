class Instance:
	properties = []
	
	def __init__(self, a, t, n):
		self.attributes = a
		self.time = t
		self.name = n
		self.merges = []
		self.splits = []
		self.depth = 0
	
	def __repr__(self):
		return self.pretty_print(False)
	
	def __str__(self):
		return self.pretty_print(True)
	
	def pretty_print(self, showAttributes=True):
		printString = self.name+" "+str(round(self.time, 2))
		if showAttributes:
			printString += ": "+str(self.getAttributes())
		return printString
	
	def __lt__(self, other):
		return self.time < other.time
	
	def getAttribute(self, a):
		if type(a) == str:
			return self.attributes[Instance.properties.index(a)]
		if type(a) == int:
			return self.attributes[a]
	
	def getAttributes(self):
		if not (len(self.attributes) == len(Instance.properties)):
			print "ERROR!: "+str(len(self.attributes))+", "+str(len(Instance.properties))
			return None
		a = {}
		for i in range(len(Instance.properties)):
			a[Instance.properties[i]] = self.attributes[i]
		return a
