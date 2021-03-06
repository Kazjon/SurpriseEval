import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as pl
import sys,os
from P import Parser
from ED import ExpectedDistribution
from EDV import ExpectedDistributionVisualiser
from OD import ObservedDistribution
from GSED import GridSearchED
from GE import Log
from joblib import Parallel, delayed
from S import Surprise
import time
from elementtree.SimpleXMLWriter import XMLWriter


def plotAndSave(od,ed,fn):
	surprise = Surprise(od, plotprefix="")
	edv = surprise.createVisualiser(50,50)
	#fig=edv.plotSurpriseGradient()
	fig = od.plotArtefacts(stroke='black',fill='white')
	edv.plotExpectationContours(plot=fig,showDU=True,showMU=True)
	#edv.plotUncertaintyChannel(onMedian=False,plot=pl.gcf().add_subplot(14,1,14))
	edv.save(fn)

if __name__ == "__main__":
	mpl.rc('figure',figsize=[18, 12]) 
	mpl.rc('figure.subplot',left=0.075,right=0.995,top=0.925,bottom=0.075)
	contours = 3
	
	namecols = [0]
	timecols = [2]
	valcols = range(2,15)
	
	parser = Parser("data/AllPhoneData_pruned.csv",namecols,timecols,valcols)
	
	# Values to start the search with
	
	Cs = {}
	Cs['Release Year'] = 1
	Cs['RAM Capacity (Mb)'] = 1
	Cs['ROM Capacity (Mb)'] = 1
	Cs['Storage (Mb)'] = 1
	Cs['CPU Clock (MHz)'] = 1
	Cs['Display Diagonal (in)'] = 1
	Cs['Display Width(px)'] = 1
	Cs['Display Length(px)'] = 1
	Cs['Width (mm)'] = 1
	Cs['Length (mm)'] = 1
	Cs['Depth (mm)'] = 1
	Cs['Volume (cubic cm)'] = 1
	Cs['Mass (grams)'] = 1
	Cs['Pixel Density (per inch)'] = 1
	
	gammas = {}
	gammas['Release Year'] = 1e-2
	gammas['RAM Capacity (Mb)'] = 1e-2
	gammas['ROM Capacity (Mb)'] = 1e-2
	gammas['Storage (Mb)'] = 1
	gammas['CPU Clock (MHz)'] = 1e-2
	gammas['Display Diagonal (in)'] = 1e-2
	gammas['Display Width(px)'] = 1e-2
	gammas['Display Length(px)'] = 1e-2
	gammas['Width (mm)'] = 1e-2
	gammas['Length (mm)'] = 1e-2
	gammas['Depth (mm)'] = 1e-2
	gammas['Volume (cubic cm)'] = 1e-2
	gammas['Mass (grams)'] = 1e-2
	gammas['Pixel Density (per inch)'] = 1e-2
	
	#Get either every dimension or a single dimension
	val1s = parser.getProperties()
	val2s = parser.getProperties()
	
	# Values that we've found through previous executions of this grid-search
	printFound = True
	found = {}
	#found[('Release Year','Width (mm)')] = {'C':1000,'gamma':0.1}
	
	prefix = "gridoutput/"
	if not os.path.exists(prefix):
	    os.makedirs(prefix)
	t = time.localtime(time.time())
	logfn = os.path.join(prefix,"gridlog_"+str(t[0])+'.'+str(t[1])+'.'+str(t[2])+'_'+str(t[3])+'.'+str(t[4])+'.'+str(t[5])+".xml")
	with open(logfn,'w') as f:
		writer = XMLWriter(f)
		rootnode = writer.start("root")
		for val1 in val1s:
			for val2 in val2s:
				if val1 is not val2:
					start_time = time.time()
					print "Modelling",val1,"(independent) against",val2,"(dependent)."
					ed = None
					if (val1,val2) not in found.keys():
						writer.start("model",ind=val1,dep=val2)
						ods = []
						weights = [0.4,0.3,0.2]
						for w in weights:
							ods.append(ObservedDistribution(parser, val1, contours, val2, w, retrain=True)) 
						#[0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100]
						#[0.01, 0.033, 0.1, 0.33, 1, 3, 10, 33, 100] 
						#[0.01, 0.1, 1, 10, 100]
						#[0.1,1,10]
						ed = GridSearchED(ods,{'C':Cs[val1],'gamma':gammas[val1]},grid={'gamma':[0.1,1],'C':[3]},parallel=True, log=writer)
						fn = os.path.join(prefix,"".join(['-'.join([val1,val2,str(ed.OD.weight_std_ratio),str(ed.params[0.5]['C']),str(ed.params[0.5]['gamma']),'.pdf'])]))
						plotAndSave(ed.OD,ed,fn)
						writer.end("model")
					elif printFound==False:
						print '--skipping as already found.'
					else:
						od = ObservedDistribution(parser, val1, contours, val2, None, prefix=odpath, retrain=True)
						ed = ExpectedDistribution(od,found[(val1,val2)])
						fn = os.path.join(prefix,"".join(['-'.join([val1,val2,str(od.weight_std_ratio),str(ed.params[0.5]['C']),str(ed.params[0.5]['gamma'])])+'.pdf']))
						plotAndSave(od,ed,fn)
					print time.time() - start_time, "seconds"
		writer.close(rootnode)
	Log(logfn).plotGridErrors()