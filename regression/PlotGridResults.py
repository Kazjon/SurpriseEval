import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as pl
import sys
from P import Parser
from ED import ExpectedDistribution
from EDV import ExpectedDistributionVisualiser
from OD import ObservedDistribution
from U import Updater
from S import Surprise
from GSED import GridSearchED
from GE import Log
from joblib import Parallel, delayed
import time
import os

if __name__ == "__main__":
	mpl.rc('figure',figsize=[18, 12]) 
	mpl.rc('figure.subplot',left=0.05,right=0.995,top=0.995,bottom=0.05)
	
	namecols = [0]
	timecols = [2]
	valcols = [2,3,4,5,6,7,8,9,10,11,12,13,14]
	parser = Parser("data/AllPhoneData_pruned.csv",namecols,timecols,valcols)
	contours = 4
	
	properties = parser.getProperties()
	gridlog = Log(sys.argv[1]) 
	for ind_attr in properties:
		for dep_attr in properties:
			if ind_attr == dep_attr:
				continue
			updater = Updater(parser, ind_attr, contours, dep_attr, None)
			surprise = None
			params = gridlog.getBestParams(ind_attr,dep_attr)
			if params is not None:
				print "Found",params,"for independent:",ind_attr,"dependent:",dep_attr
				surprise = Surprise(updater, params=params,lims=updater.allLimits([5,5]))
			else:
				print "Found no params in log for independent:",ind_attr,"dependent:",dep_attr,", using defaults"
				surprise = Surprise(updater, params={'C':1,'gamma':0.01},lims=updater.allLimits([5,5]))
			edv = surprise.createVisualiser(250,100)
			fig=edv.plotSurpriseGradient()
			updater.plotArtefacts(plot=fig,stroke='black',fill='white')
			updater.plotObservedContours(plot=fig)
			edv.plotExpectationContours(plot=fig,showDU=True,showMU=True)
			fn = "".join(['-'.join([ind_attr,dep_attr,str(updater.weight_std_ratio),str(surprise.ed.params[0.5]['C']),str(surprise.ed.params[0.5]['gamma'])])+'_hires.png'])
			edv.save(fn)