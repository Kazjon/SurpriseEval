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
from joblib import Parallel, delayed
import time

if __name__ == "__main__":
	mpl.rc('figure',figsize=[18, 12]) 
	mpl.rc('figure.subplot',left=0.05,right=0.995,top=0.995,bottom=0.05)
	contours = 4
	
	namecols = [0]
	timecols = [2]
	valcols = range(2,15)
	
	parser = Parser("data/AllPhoneData_pruned.csv",namecols,timecols,valcols)
	
	# Values that we've found through previous executions of this grid-search
	found = {}
	found[('Release Year','RAM Capacity (Mb)')] = {'C':1,'gamma':0.1}
	found[('Release Year','Pixel Density (per inch)')] = {'C':0.1,'gamma':0.01}

	
	defaults = {'C': 3,'gamma': 0.1}
	
	val1 = ['Release Year']
	val2 = ['Width (mm)']
	
	for v1 in val1:
		for v2 in val2:
			if v1 is not v2:
				start_time = time.time()
				print "Modelling",v1,"(independent) against",v2,"(dependent)."
				od = Updater(parser, v1, contours, v2, None)
				#fig = od.plotObservedContours()
				if (v1,v2) in found.keys():
					#ed = ExpectedDistribution(od,found[(v1,v2)])
					s = Surprise(od,params=found[(v1,v2)])
				else:
					#ed = ExpectedDistribution(od,defaults)
					s = Surprise(od,params=defaults)
				edv = s.createVisualiser(100,100) #ExpectedDistributionVisualiser(ed,od,None,50,20)
				edv.setLimits()
				#fig=edv.plotSurpriseGradient()
				fig=od.plotArtefacts(stroke='black',fill='black')
				#od.plotObservedContours(plot=fig)
				edv.plotExpectationContours(plot=fig,showDU=True,showMU=True)
				#edv.plotUncertaintyChannel(onMedian=False,plot=pl.gcf().add_subplot(14,1,14))
				fn = "".join(['-'.join([v1,v2,str(od.weight_std_ratio),str(s.params['C']),str(s.params['gamma']),'plotone.png'])])
				edv.save(fn)
				#edv.show()
				print time.time() - start_time, "seconds"