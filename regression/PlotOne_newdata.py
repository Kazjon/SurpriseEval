import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as pl
import sys
from P import Parser
from ED import ExpectedDistribution
from EDV import ExpectedDistributionVisualiser
from OD import ObservedDistribution
from GridSearchED import GridSearchED
from joblib import Parallel, delayed
import time

if __name__ == "__main__":
	mpl.rc('figure',figsize=[12, 8]) 
	mpl.rc('figure.subplot',left=0.05,right=0.995,top=0.995,bottom=0.05)
	contours = 4
	
	namecols = [0]
	timecols = [2]
	valcols = range(2,15)
	
	parser = Parser("AllPhoneData_pruned.csv",namecols,timecols,valcols)
	
	# Values that we've found through previous executions of this grid-search
	found = {}
	found[('Release Year','RAM Capacity (Mb)')] = {'C':1,'gamma':0.1}
	found[('Release Year','Pixel Density (per inch)')] = {'C':0.1,'gamma':0.01}

	
	defaults = {'C': 10,'gamma': 0.1}
	
	val1 = ['Release Year']
	val2 = ['ROM Capacity (Mb)']
	
	for v1 in val1:
		for v2 in val2:
			if v1 is not v2:
				start_time = time.time()
				print "Modelling",v1,"(independent) against",v2,"(dependent)."
				od = ObservedDistribution(parser, v1, contours, v2, None, prefix="ods/"+v1.replace(" ","_").replace("(","").replace(")",""))
				if (v1,v2) in found.keys():
					ed = ExpectedDistribution(od,found[(v1,v2)])
				else:
					ed = ExpectedDistribution(od,defaults)
				#testvals = [2012.923401,2013.170034,2007.497475, 2007.744108, 2007.990741, 1994.919192, 1995.165825, 1995.412458, 1991.712963, 1990.233165, 1990.479798, 1990.726431]
				#ed.MU.misclassUncertainty(testvals,ignore_eps=True)
				#sys.exit()
				edv = ExpectedDistributionVisualiser(ed,od,100,20)
				#fig=edv.plotSurpriseGradient()
				fig = od.plotArtefacts(stroke='black',fill='white')
				#od.plotObservedContours(plot=fig)
				edv.plotExpectationContours(plot=fig,showDU=True,showMU=True, ignore_eps=False)
				edv.plotUncertaintyChannel(onMedian=False,plot=pl.gcf().add_subplot(14,1,14), ignore_eps=False)
				fn = "".join(['-'.join([v1,v2,str(od.weightFactor),str(ed.params[0.5]['C']),str(ed.params[0.5]['gamma']),'plotone_oldMU2.jpg'])])
				edv.save(fn)

				#fig2=edv.plotSurpriseGradient()
				fig2 = od.plotArtefacts(stroke='black',fill='white')
				#od.plotObservedContours(plot=fig)
				edv.plotExpectationContours(plot=fig2,showDU=True,showMU=True, ignore_eps=True)
				edv.plotUncertaintyChannel(onMedian=False,plot=pl.gcf().add_subplot(14,1,14), ignore_eps=True)
				fn = "".join(['-'.join([v1,v2,str(od.weightFactor),str(ed.params[0.5]['C']),str(ed.params[0.5]['gamma']),'plotone_newMU2.jpg'])])
				edv.save(fn)
				print time.time() - start_time, "seconds"