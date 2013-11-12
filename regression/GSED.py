import numpy as np
from sklearn.svm import SVR
from MU import MisclassUncert
from OD import ObservedDistribution
from sklearn.grid_search import IterGrid
from ED import ExpectedDistribution
from joblib import Parallel, delayed
from elementtree.SimpleXMLWriter import XMLWriter
import types,pprint

def gridSquare(params,OD,verbose):
	if verbose:
		print '--Started',str(params)
	if params.keys().count('OD'):
		OD = params['OD']
	ed = ExpectedDistribution(OD,params,parallel=False)
	e = sum(ed.misclassUncertainty(OD.indAttr(),ignore_eps=True))*0.01
	if verbose:
		print '--Error for',params,'was',e
	return ed,e

class GridSearchED(ExpectedDistribution):
	
	def __init__(self, _OD, _paramsets={'C':100,'gamma':0.1}, grid=[0.1,1,10], parallel = True, train=True, verbose=True, log=None):
		g={}
		if type(_OD) is types.ListType:
			ExpectedDistribution.__init__(self, _OD[0], _paramsets, parallel, train=False)
			g['OD'] = _OD
		else:
			ExpectedDistribution.__init__(self, _OD, _paramsets, parallel, train=False)
		self.verbose = verbose
		self.log = log
		if type(grid) is dict:
			for param in self.params[0.5]: #gridsearch currently only supports EDs with paramsets uniform across contours
				g[param] = np.atleast_1d(grid[param])*self.params[0.5][param]
		else:
			grid = np.atleast_1d(grid)
			for param in self.params[0.5]: #gridsearch currently only supports EDs with paramsets uniform across contours
				g[param] = grid*self.params[0.5][param]
		
		self.grid = IterGrid(g)
		if train:
			self.train()
	
	def train(self):
		eds = []
		errors = []
		best = 0
		if self.parallel:
			results = Parallel(n_jobs=-1)(delayed(gridSquare)(params,self.OD,self.verbose) for params in self.grid)
			for i,t in enumerate(results):
				eds.append(t[0])
				errors.append(t[1])
				if errors[i] < errors[best]:
					best = i
				if self.log is not None:
					params = list(self.grid)[i]
					strparams = {}
					strparams['error'] = str(errors[i])
					for key in params:
						strparams[key] = str(params[key])
					self.log.element("test",**strparams)
		else:
			params_index = 0
			for params in self.grid:
				if self.verbose:
					print '--Started',params
				eds.append(ExpectedDistribution(self.OD,params,parallel=False))
				errors.append(sum(eds[params_index].misclassUncertainty(self.OD.indAttr(),ignore_eps=True))*0.01)
				if self.verbose:
					print '--Error for',params,'was',errors[params_index]
				if self.log is not None:
					strparams = {}
					strparams['error'] = str(errors[params_index])
					for key in params:
						strparams[key] = str(params[key])
					self.log.element("test",**strparams)
				if errors[params_index] < errors[best]:
					best = params_index
				params_index += 1
		if self.verbose:
			print '--Best was',list(self.grid)[best],'with',errors[best]
		if self.log is not None:
			params = list(self.grid)[best]
			strparams = {}
			strparams['error'] = str(errors[best])
			for key in params:
				strparams[key] = str(params[key])
			self.log.element("best",**strparams)
		self.svr = eds[best].svr
		self.params = eds[best].params
		for k in self.params.keys():
			if 'OD' in self.params[k]:
				del self.params[k]['OD']
		self.OD = eds[best].OD		
		