# Author: Kazjon Grace <k.grace@uncc.edu>
# Author: Katherine Brady <katherine.a.brady@vanderbilt.edu>

import numpy as np
from matplotlib import pyplot as pl
from P import Parser
from DU import DistanceUncertainty
from joblib import Parallel, delayed
import time, pickle, os, inspect, sys, types

def updateBins(value, sorted_ind_list, sorted_dep_list, new_sorted_ind_list, weightFunction, findBins, findNewBins, findResults, weightFactor, bins, old_weights, indecies):
	indecies.sort()
	if len(old_weights) == 0:
		return findBins(value, sorted_ind_list, sorted_dep_list, weightFunction, weightFactor, bins, findResults)
	else:
		return findBins(value, sorted_ind_list, sorted_dep_list, weightFunction, weightFactor, bins, findResults)
		#return findNewBins(value, new_sorted_ind_list, sorted_dep_list, weightFunction, weightFactor, bins, indecies, old_weights, findResults)

# function to find the bin boundaries for a given value)
def findBins(value, sorted_ind_list, sorted_dep_list, weightFunction, weightFactor, bins, findResults):
	weights = weightFunction(value, sorted_ind_list, weightFactor)
	for i in range(2, len(weights)+1):
		weights[-i+1:len(weights)] += weights[-i]
	# find each contour
	results = []
	for b in bins:
		results.append(findResults(b, weights, sorted_dep_list))
	return results, weights

def findNewBins(value, new_sorted_ind_list, sorted_dep_list, weightFunction, weightFactor, bins, indecies, old_weights, findResults):
	new_weights = weightFunction(value, new_sorted_ind_list, weightFactor).tolist()
	if value == 1996:
		print indecies
		print new_weights
		print old_weights
	for i in indecies:
		nextWeight = new_weights.pop(0)
		if i == 0:
			toInsert = nextWeight
		else:
			toInsert = nextWeight + old_weights[i-1]
		old_weights = np.insert(old_weights, i, toInsert)
		old_weights[i+1:len(old_weights)] += nextWeight
	if value == 1996:
		print old_weights
	# find each contour
	results = []
	for b in bins:
		results.append(findResults(b, old_weights, sorted_dep_list))
	return results, old_weights

def findResults(bin_number, weights, sorted_dep_list):
	lower_bound = 0
	upper_bound = len(weights)
	goal = bin_number * weights[-1]
	while upper_bound - lower_bound > 1:
		middle = (upper_bound + lower_bound)/2
		if weights[middle] > goal:
			upper_bound = middle
		else:
			lower_bound = middle
	return (bin_number, sorted_dep_list[lower_bound])

def weightFunction(value, comparisonList, weightFactor):
	return np.exp(weightFactor*(-(abs(np.array(comparisonList)-value) ** 2)))

class ObservedDistribution:

	def __init__(self, parser, ind_attr, contours , dep_attr, weight_std_ratio=None, retrain=False, prefix='.'):
		# If no weight is provided then expect to find one in a filename
		if weight_std_ratio is None:
			namestart = ind_attr+' '+str(contours)+' '+dep_attr+' '
			namestart = namestart.replace(" ","_")
			for f in os.listdir(prefix):
				fn,ext = os.path.splitext(f)
				if ext == ".od" and fn.startswith(namestart):
					weight_std_ratio = float(fn.strip(namestart))
					print "Found",f,"and setting weight ratio to",weight_std_ratio
		if weight_std_ratio is None:
			print "No suitable weight factor found in provided OD files, using 0.15."
			weight_std_ratio = 0.15
		self.weight_std_ratio = weight_std_ratio
		self.contours = contours
		self.prefix = prefix
		filename = getFileName(ind_attr, contours, dep_attr, weight_std_ratio)
		self.path = os.path.join(prefix,filename)
		# if this od has already been computed read in the file it was saved to and copy the attributes from that version
		if os.path.isfile(self.path) and not retrain:
			od = readObject(self.path)
			attributes = inspect.getmembers(od)
			for a in attributes:
				if not type(a[1]) is types.MethodType:
					setattr(self, a[0], a[1])
			print "read in",self.path
		# otherwise build the od as normal
		else:
			self.retrain(parser, ind_attr, dep_attr)
	
	def refresh(self, ind_val=None):
		# set attributes
		self.ind_list = self.Data.getList(self.ind_attr, False)
		self.dep_list = self.Data.getList(self.dep_attr, True)
		self.dep = {}
		self.ind = list(set(self.ind_list))
		self.listToContours = [self.ind.index(i) for i in self.ind_list]
		self.DU = DistanceUncertainty(self)
		
		# Variables
		self.std = np.std(self.ind_list)
		self.weightFactor = 1.0 / (2 * ((float(self.weight_std_ratio) * self.std) ** 2))
		
		# The points ordered by their dependent attribute
		orderedPoints = zip(self.dep_list, self.ind_list)
		orderedPoints.sort()
		self.sorted_dep_list, self.sorted_ind_list = zip(*orderedPoints)
		if not ind_val is None:
			new_indecies = [i for i in range(len(orderedPoints)) if self.sorted_ind_list[i] == ind_val]
			return new_indecies
	
	def retrain(self, parser, ind_attr, dep_attr):
		self.bins = [0.5]
		# Expand the bins to the number of contours selected (zero contours = just predict the median)
		for i in xrange(self.contours):
			self.bins = np.concatenate([[self.bins[0]/2],self.bins,[1-self.bins[0]/2]])
		# convert self.bins back into a list
		if not type(self.bins) == list:
			self.bins = self.bins.tolist()
		self.bins.sort()
		
		# set parser
		self.Data = parser
		# set attributes
		self.ind_attr = ind_attr
		self.dep_attr = dep_attr
		self.refresh()
		
		# Parallel computation:
		results, self.weights = zip(*Parallel(n_jobs=-1)(delayed(findBins)(value, self.sorted_ind_list, self.sorted_dep_list, weightFunction, self.weightFactor, self.bins, findResults) for value in self.ind))
		
		self.finishTraining(results)
		
		if not os.path.isfile(self.path):
			self.saveObject(self.path)
	
	def finishTraining(self, results):
		self.weights = list(self.weights)
		
		# Get Training Dots
		# Make a list of the input values at each bin boundary
		for b in self.bins:
			self.dep[b] = []
		
		for r in results:
			for pair in r:
				self.dep[pair[0]].append(pair[1])
		
		# set self.ind and self.dep (these are what ED will train on)
		self.ind = np.array(self.ind).T
		for b in self.bins:
			self.dep[b] = np.array(self.dep[b])
			#This section avoids a crash that occurs when all of a self.ind[b] have the same value.
			if np.allclose(self.dep[b],self.dep[b][0]):
				self.dep[b][0] *= 1.01
				print "Ran into a Y-axis contour (",b,") that has no variance in any of the point weights."
				print "This probably means the weighting factors are inappropriately large."
				print "Avoiding a crash in the SVR by falsely editing the first datapoint by 1% and then proceeding."
	
	# function to find the weights for each point around a single point on the independent axis
	def weightFunction(self, value):
		return np.exp(self.weightFactor*(-(abs(np.array(self.ind_list)-value) ** 2)))
	
	# getter functions
	def distanceUncertainty(self, values):
		return self.DU.distanceUncertainty(values)
	
	def indAttrName(self):
		return self.ind_attr
	
	def indAttr(self):
		return self.ind
	
	def indAttrList(self):
		return self.ind_list
	
	def depAttrName(self):
		return self.dep_attr
	
	def observedContours(self):
		return self.dep
	
	def scaledDepAttr(self):
		return self.Data.getList(self.dep_attr, True)
	
	def unscaledDepAttr(self):
		return self.Data.getList(self.dep_attr, False)
	
	# scale and unscale points trained on the scaled dots
	def unscalePoints(self, vals):
		return vals*np.array(self.Data.getScale(self.dep_attr))+np.array(self.Data.getTranslate(self.dep_attr))
		
	def scalePoints(self,vals):
		return vals/np.array(self.Data.getScale(self.dep_attr))-np.array(self.Data.getTranslate(self.dep_attr))
	
	def plotArtefacts(self,stroke=None,fill='black',plot=None,alpha=1):
		if plot is None:
			plot = pl.figure().add_subplot(1,1,1)
		plot.scatter(self.ind_list, self.unscalePoints(self.dep_list), edgecolor=stroke,facecolor=fill,s=5,lw=0.25,alpha=alpha)
		return plot

	def plotArtefact(self,x=None,y=None,plot=None,alpha=1,ED=None):
		if plot is None:
			plot = pl.figure().add_subplot(1,1,1)
		if x is None:
			minx = min(self.ind_list)
			maxx = max(self.ind_list)
			x = minx+np.random.random()*np.ptp([minx, maxx])
		if y is None:
			y_pred = ED.getExpectationsAt(x,False)
			error_scale = np.mean(y_pred[self.bins[-1]]-y_pred[self.bins[0]])
			y = ED.getExpectationsAt(x,False,medianOnly=True)+((np.random.random()*error_scale)-(0.5*error_scale))
		
		text_pos = pl.ylim()[0]+np.ptp(pl.ylim())*0.025
		
		surprise,raw_surprise = ED.surpriseCalc(x,y,None,False)
		plot.axvline(x)
		plot.scatter(x,y,s=500,c='r',marker='*')
		text = "".join(["  Hypothetical phone: S=",str(round(surprise,3)),' (raw: ',str(round(abs(raw_surprise),3)),')'])
		plot.annotate(text,[x,text_pos],color='b')
		return plot
	
	# plot the results
	def plotObservedContours(self, title="", plot=None, alpha=1):
		if plot is None:
			plot = pl.figure().add_subplot(1,1,1)
		# median dot size (for Kaz)
		med_S = 2
		# regular dot size
		reg_S = .5
		# data size
		data_S = 5
		
		centralIndex = self.bins.index(0.5)
		for i,b in enumerate(self.bins):
			dist_from_med = float(abs(i-centralIndex))/(len(self.bins) *.5)
			color = (1-dist_from_med, dist_from_med, 0)
			if b == 0.5:
				S = med_S
			else:
				S = reg_S
			zipped = zip(self.ind, self.unscalePoints(self.dep[b]))
			zipped.sort()
			x, y = zip(*zipped)
			list(x)
			list(y)
			plot.plot(x, y, color=color, lw=S, alpha=alpha)
		if self.ind_attr is None:
			plot.set_xlabel('$Year$')
		else:
			plot.set_xlabel(self.ind_attr)
		plot.set_ylabel(self.dep_attr)
		if len(title) > 0:
			plot.set_title(title)
		return plot
	
	def plotWeights(self, value, plot=None, alpha=1):
		if plot is None:
			plot = pl.figure().add_subplot(1,1,1)
		weights = self.weightFunction(value)
		weights *= 10000.0/sum(weights)
		weights = np.sqrt(weights)
		plot.scatter(self.ind_list, self.unscalePoints(self.dep_list), color='r', s=weights, alpha=alpha)
		return plot
	
	#Show the current plot(s).
	def show(self):
		pl.show()
	
	#Save the current plot to a given filename.
	def save(self,filename):
		if os.path.isfile(self.prefix+'/'+filename):
			version = 1
			while os.path.isfile(self.prefix+'/'+str(version)+'_'+filename):
				version += 1
			filename = str(version)+'_'+filename
		pl.savefig(self.prefix+'/'+filename)
		print 'Saved',self.prefix+'/'+filename
	
	def saveObject(self, filename):
		with open(filename, 'wb') as output:
			pickle.dump(self, output, pickle.HIGHEST_PROTOCOL)
	
def readObject(filename):
	with open(filename, 'rb') as input:
		return pickle.load(input)

def getFileName(ind_attr, contours, dep_attr, w):
	filename = ind_attr+' '+str(contours)+' '+dep_attr+' '+str(w)+'.od'
	return filename.replace(' ','_')

if __name__ == "__main__":
	weightFactors = [0.15]
	
	#Old phone data import
	#namecols = [0]
	#timecols = [2]
	#valcols = [2,3,4,5,6,7,8,9,10,11,12,13,14]
	#parser = Parser("AllPhoneData_pruned.csv",namecols,timecols,valcols)

	namecols = [0]
	timecols = [2]
	valcols = range(2,15)
	contours = 4
	
	parser = Parser("AllPhoneData_pruned.csv",namecols,timecols,valcols)

	
	properties = parser.getProperties()[0:2] 
	for ind_attr in properties:
		for dep_attr in properties:
			if ind_attr == dep_attr:
				continue
			ods = []
			for w in weightFactors:
				ods.append(ObservedDistribution(parser, ind_attr, contours, dep_attr, w))
				# Plotting code
				#fig = OD.plotArtefacts()
				#fig = OD.plotObservedContours(plot=fig, alpha=.9)
				#fig.set_title(str(w))
				#fig.set_xlabel(ind_attr)
				#fig.set_ylabel(dep_attr)
				#OD.save(ind_attr+'-'+dep_attr+'-'+str(w)+'.png')
				#OD.save("".join([ind_attr,dep_attr,'_David.pdf']))
