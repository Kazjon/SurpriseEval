import sys, csv
from P import Parser
from sklearn.cluster import KMeans
import numpy as np
from matplotlib import pyplot as pl
import scipy
import scipy.cluster.vq
import scipy.spatial.distance
dst = scipy.spatial.distance.euclidean

class Novelty:

	def __init__(self, parser, k, startIndex=-1, parallel = True, batch=True):
		if startIndex==-1:
			startIndex = k
		self.Data = parser
		self.names = self.Data.getNames()
		self.k = k
		self.clusters = KMeans(k, n_jobs=1 - 2*(not parallel),n_init=10)
		self.props = self.Data.getProperties()
		self.artefacts = np.atleast_2d(self.Data.getList(self.props[0]))
		for attr in self.Data.getProperties()[1:]:
			self.artefacts = np.append(self.artefacts,np.atleast_2d(self.Data.getList(attr)),axis=0)
		self.artefacts = self.artefacts.T
		self.times = self.Data.getList()
		zipped = zip(self.times,self.artefacts,self.names)
		zipped = sorted(zipped,key=lambda x: x[0])
		unzipped = zip(*zipped)
		self.times = list(unzipped[0])
		self.artefacts = np.array(unzipped[1])
		self.names = list(unzipped[2])
		if batch:
			self.trainAll()
			self.currentIndex = len(self.names)-1
		else:
			self.currentIndex = startIndex
			self.noveltyList = np.zeros(len(self.artefacts))
			while self.currentIndex+1 < len(self.names) and self.times[self.currentIndex+1]==self.times[self.currentIndex]:
				self.currentIndex +=1
			
			while self.currentIndex < len(self.names):
				self.train()
				newArtefacts = [self.currentIndex+1]
				while newArtefacts[-1]+1 < len(self.names) and self.times[newArtefacts[-1]+1]==self.times[newArtefacts[0]]:
					newArtefacts.append(newArtefacts[-1]+1)
				novelties = []
				for i,a in enumerate(self.names[newArtefacts[0]:newArtefacts[-1]+1]):
					dist,cluster = self.novelty(a,normedDistance=False)
					time=self.times[self.names.index(a)]
					novelties.append((dist/self.sizes[cluster],cluster,time,a))
					self.noveltyList[self.currentIndex+i] = novelties[-1][0]
				novelties = sorted(novelties,key=lambda x: x[0])
				scales = {}
				translates = {}
				for k in self.Data.pastCalc.keys():
					if k in self.props:
						scales[k] = self.Data.pastCalc[k]['std']
						translates[k] = self.Data.pastCalc[k]['avg']
				for n in novelties[::-1]:
					cent = np.copy(self.centroids[n[1]])
					art = np.copy(self.artefacts[self.names.index(n[3])])
					c = self.clusters.predict(art)[0]
					for i,v in enumerate(self.props):
						cent[i] = np.around(cent[i] * scales[v] + translates[v],decimals=1)
						art[i] = np.around(art[i] * scales[v] + translates[v],decimals=1)
					print 'Closest cluster to',n[3],'(released',str(n[2])+') was #'+str(n[1]),'with distance',str(n[0])+'. Actual cluster was',str(c)+'.'
					if n[0] > 1:
						print 'Attrs:	  RAM	 ROM   CPU	 DDia  DWid  DLen	Wid   Len	 Dep	Vol	Mass   DPI'
						print 'Cluster:',cent
						print 'Design: ',art
						print 'Diff:   ',art-cent
				self.increment(len(newArtefacts))
				
	
	def train(self):
		# Train a k-means cluster on everything up to currentIndex and then return the novelties of all designs in currentIndices
		trainingArtefacts = self.artefacts[0:self.currentIndex+1]
		assignments = self.clusters.fit_predict(trainingArtefacts)
		self.centroids = self.clusters.cluster_centers_
		self.members = []
		self.sizes = np.zeros(len(self.centroids))
		for i,centroid in enumerate(self.centroids):
			self.members.append([d for d in range(len(assignments)) if assignments[d] == i])
			cluster = self.artefacts[self.members[i]]
			if cluster.shape[0] > 0:
				totaldist = 0
				for artefact in cluster:
					totaldist += self.clusterdist(artefact,i)
				self.sizes[i] = totaldist/cluster.shape[0]
		self.report()
		
	#Stage the next design(s) to add
	def increment(self,new):
		self.currentIndex+=new
		
	def trainAll(self):
		assignments = self.clusters.fit_predict(self.artefacts)
		self.centroids = self.clusters.cluster_centers_
		self.members = []
		self.sizes = np.zeros(len(self.centroids))
		for i,centroid in enumerate(self.centroids):
			self.members.append([d for d in range(len(assignments)) if assignments[d] == i])
			cluster = self.artefacts[self.members[i]]
			totaldist = 0
			for artefact in cluster:
				totaldist += self.clusterdist(artefact,i)
			self.sizes[i] = totaldist/cluster.shape[0]
		
	def report(self):
		print '---------------k='+str(self.k)+', n='+str(self.currentIndex)+', t='+str(round(self.times[self.currentIndex],2))+'---------------'
		#print 'Cluster centres:',self.clusters.cluster_centers_
		print 'Inertia:',self.clusters.inertia_
		clusterlengths = []
		for c in self.members:
			clusterlengths.append(len(c))
		for cluster in np.argsort(clusterlengths)[::-1]:
			print 'Cluster',cluster,'has',len(self.members[cluster]),'members and size',self.sizes[cluster]
	
	def novelty(self,name, normedDistance=True):
		artefact = self.artefacts[self.names.index(name)]
		if normedDistance:
			minDist = sys.maxint
			closest = -1
			for cluster in xrange(len(self.centroids)):
				normedDist = self.clusterdist(artefact,cluster) / self.sizes[cluster]
				if normedDist < minDist:
					minDist = normedDist
					closest = cluster
			return minDist,closest
		else:
			cluster = self.clusters.predict(artefact)[0]
			return self.clusterdist(artefact,cluster) / self.sizes[cluster],cluster
			
	def clusterdist(self,artefact,cluster):
		return np.sqrt(np.sum((artefact-self.centroids[cluster])**2))
		
def gap(data, refs=None, nrefs=20, ks=range(1,11), iter=10):
	"""
	Compute the Gap statistic for an nxm dataset in data.

	Either give a precomputed set of reference distributions in refs as an (n,m,k) scipy array,
	or state the number k of reference distributions in nrefs for automatic generation with a
	uniformed distribution within the bounding box of data.

	Give the list of k-values for which you want to compute the statistic in ks.
	"""
	shape = data.shape
	if refs==None:
		tops = data.max(axis=0)
		bots = data.min(axis=0)
		dists = scipy.matrix(scipy.diag(tops-bots))
		rands = scipy.random.random_sample(size=(shape[0],shape[1],nrefs))
		for i in range(nrefs):
			rands[:,:,i] = rands[:,:,i]*dists+bots
	else:
		rands = refs
	
	gaps = scipy.zeros((len(ks),))
	for (i,k) in enumerate(ks):
		(kmc,kml) = scipy.cluster.vq.kmeans2(data, k, iter=iter)
		disp = sum([dst(data[m,:],kmc[kml[m],:]) for m in range(shape[0])])
	
		refdisps = scipy.zeros((rands.shape[2],))
		print 'For k =',k,'calculating random distribution #',
		for j in range(rands.shape[2]):
			print j,
			(kmc,kml) = scipy.cluster.vq.kmeans2(rands[:,:,j], k, iter=iter)
			refdisps[j] = sum([dst(rands[m,:,j],kmc[kml[m],:]) for m in range(shape[0])])
		gaps[i] = scipy.log(scipy.mean(refdisps))-scipy.log(disp)
		print ""
	return gaps
	
if __name__ == "__main__":
	np.set_printoptions(suppress=True,linewidth=150)
	namecols = [0]
	timecols = [2]
	valcols = [3,4,5,6,7,8,9,10,11,12,13,14]
	parser = Parser("data/AllPhoneData_pruned.csv",namecols,timecols,valcols)
	
	k = 6
	
	if k == -1:
		maxk = 50
		'''inertias = []
		for k in xrange(2,maxk+1):
			novelty = Novelty(parser,k)
			novelty.report()
			inertias.append(novelty.clusters.inertia_)
		deltas = []
		for i in xrange(1,len(inertias)):
			deltas.append(inertias[i-1] - inertias[i])
		plot1 = pl.figure().add_subplot(2,1,1)
		plot1.plot(range(2,maxk+1),inertias, '-bo')
		pl.bar(np.array(range(3,maxk+1))-0.5,deltas)'''
		
		plot2 = pl.gcf().add_subplot(1,1,1)
		for i in xrange(10):
			print '-------- Trial',i,'--------'
			novelty =  Novelty(parser,2)
			gaps = gap(novelty.artefacts, nrefs=10, ks=range(2,maxk+1),iter=10)
			print gaps
			plot2.plot(range(2,maxk+1),gaps)
		pl.show()
	else:
		novelty = Novelty(parser,k,startIndex=k+1,batch=False)
		with open('novelties.csv','w') as f:
			csvFile = csv.writer(f, delimiter=',', quotechar='"')
			for tup in zip(novelty.names,novelty.noveltyList):
				csvFile.writerow(list(tup))
		print 'Novelty List written to novelties.csv'