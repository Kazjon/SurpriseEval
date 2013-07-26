import numpy as np
import sys, csv
from I import Instance

class Parser:
	# The columns we are interested in
	
	def __init__(self, filename, namec, timec, valc, condition=None):
		self.namecols = namec
		self.timecols = timec
		self.valcols = valc
		self.instances = []
		self.pastCalc = {}
		with open(filename,'r') as csvfile:
			reader = csv.reader(csvfile, delimiter=',', quotechar='"')
			propertyNames = reader.next()
			for v in self.valcols:
				Instance.properties.append(propertyNames[v])
			for row in reader:
				# Compute instance name
				name = ""
				addSpace = False
				for n in self.namecols:
					if addSpace:
						name += " "
					else:
						addSpace = True
					name += str(row[n])
				# Compute instance time
				time = 0
				for t in self.timecols:
					time = np.float(row[t])
				# Compute instance attributes
				attributes = []
				for v in self.valcols:
					attributes.append(np.float(row[v]))
				inst = Instance(attributes, time, name)
				if condition is None or condition(inst):
					self.instances.append(inst)
	
	def getProperties(self):
		return Instance.properties[0:len(Instance.properties)]
	
	def getNames(self):
		if self.pastCalc.get('names', 0):
			return self.pastCalc['names']
		names = []
		for inst in self.instances:
			names.append(inst.name)
		self.pastCalc['names'] = names
		return names
	
	def getList(self, prop="", scaled=True):
		isTime = not (prop in Instance.properties)
		
		# If previously calculated return old work
		answer = self.pastCalc.get(prop, 0)
		if answer:
			if scaled and not isTime:
				return answer['scaledList']
			return answer['list']
		
		# Start calculation
		List = []
		index = 0
		if not isTime:
			index = Instance.properties.index(prop)
		for inst in self.instances:
			if isTime:
				List.append(inst.time)
			else:
				List.append(inst.attributes[index])
		
		# Scale so average is zero and std is 1
		avg = np.average(List)
		std = np.std(List)
		scaledList = ((np.array(List)-avg)/std).tolist()
		
		# Save calculations so they don't have to be redone in future
		self.pastCalc[prop] = {}
		self.pastCalc[prop]['list'] = List
		self.pastCalc[prop]['scaledList'] = scaledList
		self.pastCalc[prop]['avg'] = avg
		self.pastCalc[prop]['std'] = std
		
		# Return the requested information
		if scaled and not isTime:
			return scaledList
		return List
		
	def getScale(self,prop):
		return self.pastCalc[prop]['std']
		
	def getTranslate(self,prop):
		return self.pastCalc[prop]['avg']
