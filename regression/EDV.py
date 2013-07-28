import sys, os
import numpy as np
import scipy as sp
import matplotlib as mpl
from matplotlib import pyplot as pl
from matplotlib.collections import LineCollection 


class ExpectedDistributionVisualiser:
	
	#Create a new EDV object, passing it the meshing resolution on the X and Y axes, which are used for plotting.
	def __init__(self, ED, OD, surprise, xres=100, yres=100, prefix="plots/"):
		self.ED = ED
		self.OD = OD
		self.surprise = surprise
		self.xres = xres
		self.yres = yres
		self.prefix = ""

	#Plot the contours of the expected distribution in the space defined by the dependent and independent variables.
			#projection: a tuple for how far to the left and right of the plot to extend.
			#showDU: Whether to show the distance uncertainty as blue/red colouration.
			#showMU: Whether to show the misclassification uncertainty as blue/red colouration.
			#plot: where to plot (None saves to a file instead)
			#alpha: transparency of everything to be drawn
	def plotExpectationContours(self, projection = [0,0], showDU=False, showMU=False, plot=None, alpha=1):
		if plot is None:
			plot = pl.figure().add_subplot(1,1,1)
		self.setLimits(xprojection=projection)
		self.labelAxes(plot)
		mpl.rcParams['lines.linewidth']=1
		
		minx = min(self.OD.indAttr())-projection[0]
		maxx = max(self.OD.indAttr())+projection[1]
		x = np.array([np.linspace(minx, maxx, self.xres)]).T
		y_pred = self.ED.getExpectationsAt(x,False)
		
		DUvals=np.zeros(len(x))
		MUvals=np.zeros(len(x))
		if showDU:
			DUvals = self.OD.distanceUncertainty(x)
		if showMU:
			MUvals = self.ED.misclassUncertainty(x)
		scaling = 1-np.minimum(np.ones(len(x)),DUvals+MUvals)	
		for i,b in enumerate(self.OD.bins):
			alpha = 1-(abs(i-len(self.OD.bins)/2.0)/float(len(self.OD.bins)/2.0)) # Weight the colour of the contour to the distance from the median
			points = np.array([x[:,0], y_pred[b]]).T.reshape(-1, 1, 2) 
			segments = np.concatenate([points[:-1], points[1:]], axis=1)
			# create a (r,g,b,a) color description for each line segment 
			cmap = [] 
			for a in segments: 
				# the x-value for this segment is: 
				x0 = a.mean(0)[0]
				# so it has a scaling value of: 
				s0 = np.interp(x0,x[:,0],scaling) 
				cmap.append([1-s0,0,s0,alpha])
			# Create the line collection object, and set the color parameters. 
			lw = 1
			if b== 0.5:
				lw = 3
			lc = LineCollection(segments, linewidths=lw) 
			lc.set_color(cmap) 
			plot.add_collection(lc)
		return plot

	#Plot the uncertainty at each point in the space defined by the dependent and independent variables, either as a channel around the median or an area plot.
		#projection: a tuple for how far to the left and right of the plot to extend.
		#showDU: Whether to show the distance uncertainty
		#showMU: Whether to show the misclassification uncertainty.
		#onMedian: Whether to plot as a channel around the median, or as a separate plot.
		#plot: where to plot (None saves to a file instead)
		#alpha: transparency of everything to be drawn
	def plotUncertaintyChannel(self, projection = [0,0], showDU=True, showMU=True, onMedian=True, plot=None, alpha=0.5):
		if plot is None:
			plot = pl.figure().add_subplot(1,1,1)
		self.setLimits()
		self.labelAxes(plot)
		mpl.rcParams['lines.linewidth'] = 1
		
		minx = min(self.OD.indAttr())-projection[0]
		maxx = max(self.OD.indAttr())+projection[1]
		x = np.array([np.linspace(minx, maxx, self.xres)]).T
		y_pred = self.ED.getExpectationsAt(x,False)
		
		DUvals=np.zeros(len(x))
		MUvals=np.zeros(len(x))
		if showDU:
			DUvals = self.OD.distanceUncertainty(x)
		if showMU:
			MUvals = self.ED.misclassUncertainty(x)
		if onMedian:
			error_scale = np.mean(y_pred[self.OD.bins[-1]]-y_pred[self.OD.bins[0]])
			MUvals *= error_scale
			DUvals *= error_scale
			plot.plot(x, y_pred[0.5]-error_scale*0.5,'r--',alpha=alpha)
			plot.plot(x, y_pred[0.5]+error_scale*0.5,'r--',alpha=alpha)
			plot.fill(np.concatenate([x, x[::-1]]),np.concatenate([y_pred[0.5] - (0.5 * MUvals),(y_pred[0.5] + (0.5 * MUvals))[::-1]]),alpha=alpha*0.5, fc='r', ec='None')
			plot.fill(np.concatenate([x, x[::-1]]),np.concatenate([np.maximum(y_pred[0.5]-error_scale*0.5,y_pred[0.5] - (0.5 * MUvals) - (0.5 * DUvals)),np.minimum((y_pred[0.5]+error_scale*0.5)[::-1],(y_pred[0.5] + (0.5 * MUvals) + (0.5 * DUvals))[::-1])]),alpha=alpha*0.5, fc='r', ec='None')
		else:
			pl.ylim([0,2])
			plot.plot(x, np.ones(len(x)),'r--',alpha=alpha)
			plot.fill_between(x[:,0],MUvals,alpha=alpha*0.5, facecolor='r', edgecolor='None')
			plot.fill_between(x[:,0],MUvals+DUvals,alpha=alpha*0.5, facecolor='r', edgecolor='None')
			
		return plot
	
	#Plot a gradient showing the surprisingness of each point in the space defined by the dependent and independent variables.
		#projection: a tuple for how far to the left and right of the plot to extend.
		#plot: where to plot (None saves to a file instead)
		#alpha: transparency of everything to be drawn
	def plotSurpriseGradient(self, projection = [0,0], plot=None, alpha=1):
		if plot is None:
			plot = pl.figure().add_subplot(1,1,1)
		self.setLimits(xprojection=projection)
		self.labelAxes(plot)
		mpl.rcParams['lines.linewidth'] = 1
		
		minx = min(self.OD.indAttr())-projection[0]
		maxx = max(self.OD.indAttr())+projection[1]
		x = np.array([np.linspace(minx, maxx, self.xres)]).T
		y_pred = self.ED.getExpectationsAt(x,False)
		
		valspace = np.linspace(pl.ylim()[0],pl.ylim()[1],self.yres)
		xx,yy = np.meshgrid(x,valspace)
		surpriseFuncs = []
		uncerts = []
		for v in x:
			if hasattr(v, '__len__'):
				v = v[0]
			sF,u = self.surprise.surpriseFunction(v)
			surpriseFuncs.append(sF)
			uncerts.append(u)
		surpriseVals = []
		for i,v in enumerate(x):
			surpriseVals.append(1-((1-uncerts[i])*abs(surpriseFuncs[i](valspace))))
		surpriseVals = np.array(surpriseVals)
		levels = np.linspace(0,1,self.yres)
		cmap = mpl.cm.gray
		cmap.set_over('black')
		cs = pl.contourf(xx,yy,surpriseVals.T,levels=levels,cmap=cmap, alpha=alpha)
		for c in cs.collections: 
			c.set_antialiased(False)
		return plot
	
	#Plot a cumulative distribution function aligning the likelihood of an observation at a given value with its surprisingness.
		#x: The x axis value to calculate the distribution at
		#y: The y axis value to evaluate surprise at.
		#plot: where to plot (None saves to a file instead)
		#alpha: transparency of everything to be drawn
	def plotSurpriseDistribution(self, x, y, plot=None, alpha=1, scaled=False):
		if scaled:
			y = self.OD.unscalePoints(y)
		if plot is None:
			plot = pl.figure().add_subplot(1,1,1)
		self.surprise.surpriseFig(x, y, plot,alpha=alpha) 
	
	#Plot a single artefact along with its surprisingness in the space defined by the dependent and independent variables.
		#x: The x axis value of the artefact, or None for a random one.
		#y: The y axis value of the artefact, or None for a random one.
		#plot: where to plot (None saves to a file instead)
		#alpha: transparency of everything to be drawn
	def plotArtefact(self, x=None, y=None, plot=None, alpha=1,ED=None):
		return self.OD.plotArtefact(x,y,plot,alpha,ED)
	
	#Plot all the known artefacts in the space defined by the dependent and independent variables.
		#stroke: the colour of the artefact borders.
		#fill: the colour of the artefacts.
		#plot: where to plot (None saves to a file instead)
		#alpha: transparency of everything to be drawn
	def plotArtefacts(self, stroke=None,fill='black', plot=None, alpha=1):
		return self.OD.plotArtefacts(stroke,fill,plot,alpha)
	
	#Add the names of the dependent and independent attributes to the x and y labels respectively.
	def labelAxes(self, plot):
		plot.set_xlabel(self.OD.indAttrName())
		plot.set_ylabel(self.OD.depAttrName())
	
	#Sets the limits of the current figure to some default values based on the range of known artefacts
		#xprojection: a tuple for how far to the left and right of the plot to extend.
		#ypadding: How much further than the most extreme Y values to plot, as a percentage of the known data's y range.
	def setLimits(self, xprojection=[0,0], ypadding=0.1): 
		minx = min(self.OD.indAttr())
		maxx = max(self.OD.indAttr())
		pl.xlim(minx-xprojection[0],maxx+xprojection[1])
		miny = min(self.OD.unscaledDepAttr())
		maxy = max(self.OD.unscaledDepAttr())
		yrange = maxy - miny
		pl.ylim(miny-(ypadding*yrange),maxy+(ypadding*yrange))
		#print "Figure limits set to X:",minx,maxx,"Y:",miny,maxy
	
	#Show the current plot(s).
	def show(self):
		pl.show()
	
	#Save the current plot to a given filename.
	def save(self,filename,printConfirmation=True):
		pl.savefig(os.path.join(prefix,filename))
		if printConfirmation:
			print 'Saved',filename
