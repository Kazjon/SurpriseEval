import numpy as np
from P import Parser
from ED import ExpectedDistribution
from EDV import ExpectedDistributionVisualiser
from GE import Log
from U import Updater
from matplotlib import pyplot as pl
import matplotlib as mpl
import scipy.interpolate as interp
import pickle
import os,subprocess, sys
from joblib import Parallel, delayed

class Surprise:
	
	def __init__(self, updater=None, ed=None, params={'C':1,'gamma':0.01}, lims=None, spprefix = "sps/", plotprefix="plots/"):
		self.params = params
		self.updater = updater
		self.spprefix = spprefix
		self.plotprefix = plotprefix
		self.ed = ed
		self.edv = None
		self.xlims = lims[0]
		self.ylims = lims[1]
		self.prevFunctions = {}
		if ed is None:
			self.ed = ExpectedDistribution(self.updater, self.params)
			
	def createVisualiser(self, xres=100, yres=100):
		self.edv = ExpectedDistributionVisualiser(self.ed, self.updater, self, xres, yres, self.plotprefix)
		return self.edv
	
	def filename(self, index=-1):
		indexString = ' '+str(index)
		if index == -1:
			indexString = ""
		filename = self.updater.ind_attr+' '+self.updater.dep_attr+indexString+'.sp'
		return os.path.join(self.spprefix,filename.replace(' ','_'))
	
	def saveList(self, surprise_list, index=-1, remove=-1):
		filename = self.filename(index)
		if remove >= 0:
			old_filename = self.filename(remove)
			if os.path.isfile(old_filename):
				os.remove(old_filename)
		with open(filename, 'wb') as output:
			pickle.dump(surprise_list, output, pickle.HIGHEST_PROTOCOL)
	
	def readList(self, index=-1):
		filename = self.filename(index)
		if os.path.isfile(filename):
			with open(filename, 'rb') as input:
				return pickle.load(input)
		else:
			return None
	
	def runUpdaterAndCalcSurprise(self, recompute=False, catchUp=True, update=True, plotAtUpdate=False):
		if self.edv is None:
			self.createVisualiser()
		surprise_list = self.readList()
		if surprise_list is None or recompute:
			test_list = self.updater.getList(False)
			max_ind = max(self.updater.indAttrList())
			surprise_list = []
			find_list = None
			# catch up
			if catchUp:
				for start_index in range(len(test_list)):
					find_list = self.readList(start_index)
					if find_list:
						break
			if find_list:
				surprise_list = find_list
				old_index = start_index
				start_index += 1
			else:
				old_index = -1
				start_index = 0
				if test_list[start_index][0] > max_ind:
					max_ind = test_list[start_index][0]
			imageCounter = 0;
			for i in range(start_index, len(test_list)):
				ind, dep, name = test_list[i]
				surprise_list.append(self.surpriseCalc(ind, dep)[0])
				if ind > max_ind:
					max_ind = ind
					if plotAtUpdate:
						count = "%05d" % imageCounter
						imageCounter += 1
						#number_to_print = 5
						#print 'least', surprise_list[0:number_to_print]
						#print 'most', surprise_list[-number_to_print:-1] + [surprise_list[-1]]
						fn = "Surprise "+self.updater.indAttrName(True)+" "+self.updater.depAttrName(True)+" "+count
						fn = fn.replace(" ","_")
						self.makePlot(fn+'.jpg', surprise_list, test_list[0:i+1])
					print "updating",i
					self.updater.update(max_ind)
					self.ed = ExpectedDistribution(self.updater, self.params)
					self.edv = self.createVisualiser()
				if i >= old_index * 2:
					self.saveList(surprise_list, i, old_index)
					old_index = i
			surprise_list = zip(surprise_list, test_list)
			surprise_list.sort()
			self.saveList(surprise_list, remove=old_index)
		return surprise_list
	
	def makePlot(self, filename, surprise_list, test_list):
		if self.edv is None:
			self.createVisualiser()
		ind_vals, dep_vals, names = zip(*test_list)
		fig = self.edv.plotExpectationContours(showDU=True,showMU=True)
		#fig = self.updater.plotArtefacts(plot=fig, fill='green')
		#fig = self.updater.plotObservedContours(plot=fig)
		fig = self.plotArtefacts(surprise_list=zip(surprise_list, test_list), plot=fig)
		fig.set_title(str(self.updater.weight_std_ratio))
		fig.set_xlabel(self.updater.indAttrName())
		fig.set_ylabel(self.updater.depAttrName())
		ind_buffer = (max(ind_vals) - min(ind_vals))*.05
		dep_buffer = (max(dep_vals) - min(dep_vals))*.05
		if self.xlims == None:
			fig.set_xlim(min(ind_vals)-ind_buffer, max(ind_vals)+ind_buffer)
		else:
			fig.set_xlim(self.xlims[0],self.xlims[1])
		if self.ylims == None:
			fig.set_ylim(min(dep_vals)-dep_buffer, max(dep_vals)+dep_buffer)
		else:
			fig.set_ylim(self.ylims[0],self.ylims[1])
		self.updater.saveFig(os.path.join(self.plotprefix,filename))

	def surpriseFunction(self, indval):
		if self.prevFunctions.get(indval, False):
			return self.prevFunctions[indval]
		#Unlike most of these functions, surpriseCalc only works on a single (x,y) pair.
		#Calculate the predictions for each bin at this time.
		predictions = self.ed.getExpectationsAt(np.atleast_2d(indval).T)		
		predictedDists = self.updater.unscalePoints(np.concatenate(predictions.values())) 
		#concat because getPredictionBins returns an array for each bin, we just want an array of all bins
		
		predictedDists.sort()
		valrange = np.ptp(predictedDists)
		
		freqs= predictions.keys()
		freqs.sort()		
		# Adjust the frequency axis to [-1,1]
		freqs = np.array(freqs)
		freqs = freqs - 0.5
		freqs = freqs * 2
		
		#Calculate the error percentage
		distUncert = self.updater.distanceUncertainty(indval)[0]
		errUncert = self.ed.MU.misclassUncertainty(indval)[0]
		uncertainty = min(1,distUncert+errUncert)
		
		# Add in the fake end values
		range_extension = 2
		predictedDists = np.concatenate([[predictedDists[0]-range_extension*(5*valrange)],predictedDists,[predictedDists[-1]+range_extension*(5*valrange)]])
		freqs = np.concatenate([[-1],freqs,[1]])
		
		f = interp.PchipInterpolator(predictedDists,freqs)
		self.prevFunctions[indval] = (f, uncertainty)
		return f,uncertainty

	#surpriseCalc only works on a single value of x
	def surpriseCalc(self,indval,depval,dep_scaled=False):
		if dep_scaled:
			depval = self.updater.unscalePoints(depval)
		f,uncertainty = self.surpriseFunction(indval)
		raw_surprise = f(depval)
		signed_surprise = raw_surprise*(1-uncertainty)
		surprise = abs(signed_surprise)
		return surprise,raw_surprise
	
	def surpriseFig(self, indval, depval, fig, alpha=1):
		if self.edv is None:
			self.createVisualiser()
		f,uncertainty = self.surpriseFunction(indval)
		mpl.rcParams['lines.linewidth'] = 1
		pl.ylim(-1,1)
		# Add in the fake end values
		range_extension = 2
		predictedDists = np.concatenate([[predictedDists[0]-range_extension*(5*valrange)],predictedDists,[predictedDists[-1]+range_extension*(5*valrange)]])
		freqs = np.concatenate([[-1],freqs,[1]])
		xlimits=[predictedDists[1]-(valrange*0.5),predictedDists[-2]+(valrange*0.5)]
		pl.xlim(xlimits)
		pl.scatter(predictedDists,freqs,alpha=alpha)
		interpi = np.linspace(predictedDists[0],predictedDists[-1],10000)
		interpd = f(interpi)
		pl.plot(interpi,interpd,alpha=alpha)
		pl.scatter(depval,f(depval),s=500,c='r',marker='*',alpha=alpha)
		pl.axhline(signed_surprise,alpha=alpha)
		pl.axhline(0,alpha=alpha*0.25,ls='--')
		fig.set_xlabel(self.OD.indAttrName())
		fig.set_ylabel("Surprise (ignore sign)")
	
	def plotArtefacts(self,surprise_list=None,stroke=None,fill='black',plot=None,alpha=1):
		if self.edv is None:
			self.createVisualiser()
		if plot is None:
			plot = pl.figure().add_subplot(1,1,1)
		
		if surprise_list is None:
			surprise_list = self.surpriseList()
		s_list, test_list = zip(*surprise_list)
		s_list = list(s_list)
		colors = []
		for i in range(len(s_list)):
			if s_list[i] > 1:
				s_list[i] = 1
				colors.append('red')
			else:
				colors.append(fill)
		S = [max((s ** 4)*30,1) for s in s_list]
		x = [s[1][0] for s in surprise_list]
		y = [s[1][1] for s in surprise_list]
		
		plot.scatter(x, y, edgecolor=stroke,facecolor=colors,s=S,lw=0.25,alpha=alpha)
#		for (ind, dep, name) in test_list:
#			fig.annotate(
#				name, 
#				xy = (ind, dep), xytext = (-20, 20),
#				textcoords = 'offset points', ha = 'right', va = 'bottom',
#				bbox = dict(boxstyle = 'round,pad=0.5', fc = 'yellow', alpha = 0.5),
#				arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))
		return plot

if __name__ == "__main__":
	mpl.rc('figure',figsize=[12, 8]) 
	mpl.rc('figure.subplot',left=0.05,right=0.995,top=0.995,bottom=0.05)
	
	condition_train = lambda inst: inst.time < 1991.4
	condition_test = lambda inst: True
	namecols = [0]
	timecols = [2]
	valcols = [2,3,4,5,6,7,8,9,10,11,12,13,14]
	parser_train = Parser("data/AllPhoneData_pruned.csv",namecols,timecols,valcols,condition_train)
	parser_test = Parser("data/AllPhoneData_pruned.csv",namecols,timecols,valcols,condition_test)
#	namecols = [1,2]
#	timecols = [11]
#	valcols = [3,4,5,6,7,8,9,11]
#	parser_train = Parser("prunedPhones.csv",namecols,timecols,valcols,condition_train)
#	parser_test = Parser("prunedPhones.csv",namecols,timecols,valcols,condition_test)
	contours = 4
	number_to_print = 5
	
	properties = parser_train.getProperties()
	gridlog = Log(sys.argv[1]) 
	for ind_attr in [properties[0]]:
		for dep_attr in properties[:]:
			if ind_attr == dep_attr:
				continue
			updater = Updater(parser_train, ind_attr, contours, dep_attr, None, parser_test=parser_test)
			surprise = None
			params = gridlog.getBestParams(ind_attr,dep_attr)
			if params is not None:
				print "Found",params,"for independent:",ind_attr,"dependent:",dep_attr
				surprise = Surprise(updater, params=params,lims=updater.allLimits([5,5]))
			else:
				print "Found no params in log for independent:",ind_attr,"dependent:",dep_attr,", using defaults"
				surprise = Surprise(updater, params={'C':1,'gamma':0.01},lims=updater.allLimits([5,5]))
			edv = surprise.createVisualiser(250,250)
			surprise_list = surprise.runUpdaterAndCalcSurprise(recompute=True, catchUp=False, plotAtUpdate=True)
			print 'least', surprise_list[0:number_to_print]
			print 'most', surprise_list[-number_to_print:-1] + [surprise_list[-1]]
			fn = "plots/Surprise_"+updater.indAttrName(True)+"_"+updater.depAttrName(True)
			subprocess.call("ffmpeg -b 5000 -f image2 -r 8 -i "+fn+"_%05d.jpg -vcodec libx264 -y "+fn+".mp4",shell=True)
			parser_train = Parser("data/AllPhoneData_pruned.csv",namecols,timecols,valcols,condition_train)
			parser_test = Parser("data/AllPhoneData_pruned.csv",namecols,timecols,valcols,condition_test)