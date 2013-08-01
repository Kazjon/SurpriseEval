import OD
from OD import ObservedDistribution
from P import Parser
from matplotlib import pyplot as pl
import numpy as np
from joblib import Parallel, delayed

class Updater(ObservedDistribution):
	
	def __init__(self, parser_train, ind_attr, contours , dep_attr, weight_std_ratio=1, parser_test=None, retrain=True, prefix=None):
		ObservedDistribution.__init__(self, parser_train, ind_attr, contours , dep_attr, weight_std_ratio, retrain, prefix, save=False)
		self.addTestData(parser_test)
	
	def addTestData(self, parser_test):
		self.test_parser = parser_test
		ind_vals = self.test_parser.getList(self.ind_attr, False)
		dep_vals = self.test_parser.getList(self.dep_attr, False)
		names = self.test_parser.getNames()
		self.test_list = zip(ind_vals, dep_vals, names)
		self.test_list.sort()
		self.toBeginning()
	
	def update(self, ind_val):
		all_ind_vals = self.test_parser.getList(self.ind_attr, False)
		zipped = zip(all_ind_vals, self.test_parser.instances)
		instances = [inst for (ind, inst) in zipped if ind == ind_val]
		for inst in instances:
			self.test_parser.instances.remove(inst)
			self.Data.instances.append(inst)
			self.test_parser.pastCalc = {}
			self.Data.pastCalc = {}
		indices = self.refresh(ind_val)
		for i in indices:
			self.weights.insert(i, np.array([]))
		new_sorted_ind_list = [self.sorted_ind_list[i] for i in indices]
		
		# Parallel computation:
		results, self.weights = zip(*Parallel(n_jobs=-1)(delayed(OD.updateBins)(self.ind[i], self.sorted_ind_list, self.sorted_dep_list, new_sorted_ind_list, OD.weightFunction, OD.findBins, OD.findNewBins, OD.findResults, self.weightFactor, self.bins, self.weights[i], indices) for i in range(len(self.ind))))
		#print np.array(self.weights)[indices]
		self.finishTraining(results)
	
	def toBeginning(self):
		self.test_counter = 0
	
	def getNext(self, scale=False, remove=False):
		ind_vals, dep_vals, names = zip(*[(i,d,n) for (i,d,n) in self.test_list if i == self.test_list[self.test_counter][0]])
		ind_vals = list(ind_vals)
		dep_vals = list(dep_vals)
		names = list(names)
		if scale:
			dep_vals = self.scalePoints(dep_vals).tolist()
		if not remove:
			self.test_counter += len(ind_vals)
		else:
			self.test_list = self.test_list[0:self.test_counter] + self.test_list[self.test_count + len(ind_vals):len(self.test_list)]
		return zip(ind_vals, dep_vals, names)
	
	def getList(self, scale=False):
		ind_vals, dep_vals, names = zip(*self.test_list)
		ind_vals = list(ind_vals)
		dep_vals = list(dep_vals)
		names = list(names)
		if scale:
			dep_vals = self.scalePoints(dep_vals).tolist()
		return zip(ind_vals, dep_vals, names)
	
	def atEnd(self):
		return self.test_counter == len(self.test_parser.getList(self.ind_attr, False))
	
	def plotArtefacts(self,stroke=None,fill='black',plot=None,alpha=1, trainOnly=False):
		ind_list = self.ind_list
		dep_list = self.unscaledDepAttr()
		if not trainOnly:
			ind_list += self.test_parser.getList(self.ind_attr, False)
			dep_list += self.test_parser.getList(self.dep_attr, False)
		if plot is None:
			plot = pl.figure().add_subplot(1,1,1)
		plot.scatter(ind_list, dep_list, edgecolor=stroke,facecolor=fill,s=5,lw=0.25,alpha=alpha)
		return plot
		
	def allLimits(self, projection=[0,0]):
		limits = self.limits()
		ind = self.test_parser.getList(self.ind_attr, False)
		dep = self.test_parser.getList(self.dep_attr, False)
		indrange = max(ind) - min(ind)
		deprange = max(dep) - min(dep)
		limits[0] = [min(limits[0][0],min(ind)) - (0.05*indrange),max(limits[0][1],max(ind)) + (0.05*indrange)]
		limits[1] = [min(limits[1][0],min(dep)) - (0.05*deprange),max(limits[1][1],max(dep)) + (0.05*deprange)]
		return limits

def plotThings(updater):
	fig = updater.plotArtefacts()
	fig = updater.plotObservedContours(plot=fig, alpha=.9)
	fig.set_title(str(w))
	fig.set_xlabel(ind_attr)
	fig.set_ylabel(dep_attr)

if __name__ == "__main__":
	weightFactors = [.15]
	
	condition_train = lambda inst: inst.time < 2001
	condition_test = lambda inst: inst.time >= 2001
	namecols = [0]
	timecols = [2]
	valcols = [2,3,4,5,6,7,8,9,10,11,12,13,14]
	parser_train = Parser("data/AllPhoneData_pruned.csv",namecols,timecols,valcols,condition_train)
	parser_test = Parser("data/AllPhoneData_pruned.csv",namecols,timecols,valcols,condition_test)
	contours = 6
	
	properties = parser_train.getProperties()
	for ind_attr in [properties[0]]:
		for dep_attr in [properties[-1]]:
			if ind_attr == dep_attr:
				continue
			for w in weightFactors:
				updater = Updater(parser_train, ind_attr, contours, dep_attr, w, parser_test=parser_test, prefix="ods/")
				plotThings(updater)
				updater.update(2001)
				plotThings(updater)
				updater.show()
