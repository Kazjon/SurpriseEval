import numpy as np
import matplotlib as mpl
import sys
import scipy.interpolate as interp
from matplotlib import pyplot as pl
from P import Parser
from sklearn.svm import SVR
from EDV import ExpectedDistributionVisualiser
from MU import MisclassUncert
from OD import ObservedDistribution
from joblib import Parallel, delayed
import time

def trainBin(params, ind, dep):
	return SVR(cache_size=1000,kernel='rbf', C=params['C'], gamma=params['gamma']).fit(ind, dep)

def predictBin(svr, vals):
	return svr.predict(vals)

class ExpectedDistribution:
	
	def __init__(self, _OD, _paramsets={'C':100,'gamma':0.1}, parallel = True, train=True):
		self.OD = _OD
		self.parallel = parallel
		self.params = {}
		for b in self.OD.bins:
			if _paramsets.has_key(b):
				self.params[b] = _paramsets[b]
			else:
				self.params[b] = _paramsets
		self.ind = self.OD.indAttr()
		self.dep = self.OD.observedContours()
		self.svr = {}
		if train:
			self.train()
		self.MU = MisclassUncert(self,self.OD)
	
	def train(self):
		regressors = []
		if self.parallel:
			regressors = Parallel(n_jobs=-1)(delayed(trainBin)(self.params[b], np.atleast_2d(self.ind).T, self.dep[b]) for b in self.OD.bins)
		else:
			for b in self.OD.bins:
				regressors.append(trainBin(self.params[b],np.atleast_2d(self.ind).T, self.dep[b]))
				#self.svr[b] = SVR(cache_size=1000,kernel='rbf', C=self.params[b]['C'], gamma=self.params[b]['gamma'])
				#self.svr[b].fit(np.array([self.ind]).T,self.dep[b])
		for i,model in enumerate(regressors):
			self.svr[self.OD.bins[i]] = model	
	
	def misclassUncertainty(self, values, ignore_eps=True):
		return self.MU.misclassUncertainty(values, ignore_eps)
		
	# Get the bin distribution predicted by the SVR model
	def getExpectationsAt(self, vals, returnScaled=True, medianOnly=False):
		if medianOnly:
			results = self.svr[0.5].predict(vals)
			if not returnScaled:
				results = self.OD.unscalePoints(results)
			return results

		# get the result predicted for each input value
		results = {}
		if self.parallel and len(vals) > 1:
			predlist = Parallel(n_jobs=-1)(delayed(predictBin)(self.svr[b],vals) for b in self.OD.bins)
			for i,pred in enumerate(predlist):
				results[self.OD.bins[i]] = pred
		else:
			for b in self.OD.bins:
				results[b] = self.svr[b].predict(vals)
	
		# Flatten lines which cross
		# if b < 0.5 flatten b to its larger neighbor
		reverse_bins = self.OD.bins[::-1]
		for i,b in enumerate(reverse_bins):
			if b < 0.5:
				results[b] = np.minimum(results[b],results[reverse_bins[i-1]])
		# if b > 0.5 flatten b to its smaller neighbor
		for i,b in enumerate(self.OD.bins):
			if b > 0.5:
				results[b] = np.maximum(results[b],results[self.OD.bins[i-1]])
	
		# undo scaling to display points corresponding to the original values
		if not returnScaled:
			for b in self.OD.bins:
				results[b] = self.OD.unscalePoints(results[b])
		return results
	
	def getParams(self):
		return self.svr[0.5].get_params() 
	
	'''SurpriseCalc has been moved to S.py and split up
	#surpriseCalc only works on a single (x,y) pair.
	def surpriseCalc(self,indval,depval,fig=None,dep_scaled=True, returnFunction=False, alpha=1, ignore_eps = False):
		#Unlike most of these functions, surpriseCalc only works on a single (x,y) pair.
		#Calculate the predictions for each bin at this time.
		predictions = self.getExpectationsAt(np.atleast_2d(indval).T)		
		predictedDists = self.OD.unscalePoints(np.concatenate(predictions.values())) 
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
		distUncert = self.OD.distanceUncertainty(indval)[0]
		errUncert = self.MU.misclassUncertainty(indval, ignore_eps=ignore_eps)[0]
		uncertainty = min(1,distUncert+errUncert)
		
		if dep_scaled:
			depval = self.OD.unscalePoints(depval)
		
		# Add in the fake end values
		range_extension = 2
		predictedDists = np.concatenate([[predictedDists[0]-range_extension*(5*valrange)],predictedDists,[predictedDists[-1]+range_extension*(5*valrange)]])
		freqs = np.concatenate([[-1],freqs,[1]])

		f = interp.PchipInterpolator(predictedDists,freqs)
		if returnFunction:
			return f,uncertainty
		raw_surprise = f(depval)
		signed_surprise = raw_surprise*(1-uncertainty)
		surprise = abs(signed_surprise)
		if fig is not None:
			mpl.rcParams['lines.linewidth'] = 1
			pl.ylim(-1,1)
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
			#print 'Surprisingness = ',round(surprise,3),', raw = ',abs(round(raw_surprise,3))
		return surprise,raw_surprise
	'''
	
#This main function has not been maintained and is probably crashy and unreliable, it's here for legacy only.
if __name__ == "__main__":
	mpl.rc('figure',figsize=[9, 6]) 
	mpl.rc('figure.subplot',left=0.05,right=0.995,top=0.995,bottom=0.05)
	contours = 6
	
	namecols = [1,2]
	timecols = [11]
	valcols = [3,4,5,6,7,8,9,11]
	
	parser = Parser("prunedPhones.csv",namecols,timecols,valcols)
	
	#Get either every dimension or a single dimension
	val1s = [parser.getProperties()[0]]
	val2s = [parser.getProperties()[1]]
	
	weightFactors = [0.2]
	Cs =			[20]
	gammas =		[0.2]


	#'''
	od = ObservedDistribution(parser, val1s[0], contours, val2s[0], weightFactors[0])
	ed = ExpectedDistribution(od,parallel=False)
	edv = ExpectedDistributionVisualiser(ed,od,50,50)
	fig = od.plotObservedContours(title='', alpha=0.25)
	edv.save('dummy.pdf')
	#'''
	
	print "---------------------Parallel=True---------------------"
	for val1 in val1s:
		for val2 in val2s:
			if val1 is not val2:
				start_time = time.time()
				#print "Modelling",val1,"(independent) against",val2,"(dependent)."
				od = ObservedDistribution(parser, val1, contours, val2, weightFactors[0])
				ed = ExpectedDistribution(od,{'C':Cs[0],'gamma':gammas[0]},parallel=True)
				edv = ExpectedDistributionVisualiser(ed,od,50,50)
				#fig = od.plotObservedContours(title=val1+' '+val2, alpha=0.25)
				fig=od.plotArtefacts(stroke='black',fill='black')
				edv.plotExpectationContours(plot=fig,showDU=True,showMU=True)
				#fakeIndex = int(np.random.random()*len(od.indAttr()))
				#fakeX = od.indAttr()[fakeIndex]
				#fakeY = od.unscaledDepAttr()[fakeIndex]
				#edv.plotArtefact(fakeX,fakeY, plot=fig,ED=ed)
				#edv.plotUncertaintyChannel(onMedian=False,plot=pl.gcf().add_subplot(14,1,14))
				#edv.plotSurpriseDistribution(fakeX,fakeY,plot=pl.gcf().add_subplot(5,5,5))
				fn = "".join(['-'.join([val1,val2,str(weightFactors[0]),str(Cs[0]),str(gammas[0])]),'-P=True.pdf'])
				edv.save(fn,False)
				print time.time() - start_time, "seconds"
	
	print "---------------------Parallel=False---------------------"
	for val1 in val1s:
		for val2 in val2s:
			if val1 is not val2:
				start_time = time.time()
				#print "Modelling",val1,"(independent) against",val2,"(dependent)."
				od = ObservedDistribution(parser, val1, contours, val2, weightFactors[0])
				ed = ExpectedDistribution(od,{'C':Cs[0],'gamma':gammas[0]},parallel=False)
				edv = ExpectedDistributionVisualiser(ed,od,50,50)
				#fig = od.plotObservedContours(title=val1+' '+val2, alpha=0.25)
				fig=od.plotArtefacts(stroke='black',fill='white')
				edv.plotExpectationContours(plot=fig,showDU=True,showMU=True)
				#fakeIndex = int(np.random.random()*len(od.indAttr()))
				#fakeX = od.indAttr()[fakeIndex]
				#fakeY = od.unscaledDepAttr()[fakeIndex]
				#edv.plotArtefact(fakeX,fakeY, plot=fig,ED=ed)
				#edv.plotUncertaintyChannel(onMedian=False,plot=pl.gcf().add_subplot(14,1,14))
				#edv.plotSurpriseDistribution(fakeX,fakeY,plot=pl.gcf().add_subplot(5,5,5))

				fn = "".join(['-'.join([val1,val2,str(weightFactors[0]),str(Cs[0]),str(gammas[0])]),'-P=False.pdf'])
				edv.save(fn,False)
				print time.time() - start_time, "seconds"
				
#'''