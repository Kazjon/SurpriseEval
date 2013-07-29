from lxml import etree
import sys
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as pl
import os

class test:
	def __init__(self, node):
		self.C = float(node.get('C'))
		self.gamma = float(node.get('gamma'))
		self.error = float(node.get('error'))
	
	def __lt__(self, other):
		return self.error < other.error
	
	def __repr__(self):
		return str((self.C, self.gamma, self.error))
	
	def __str__(self):
		return 'C='+str(self.C)+' gamma='+str(self.gamma)+' error='+str(self.error)

class model:
	def __init__(self, node):
		self.X = node.get('ind')
		self.Y = node.get('dep')
		self.tests = []
		for test_xml in node.findall('test'):
			self.tests.append(test(test_xml))
		self.tests.sort()
	
	def __repr__(self):
		return str(self.X)+" "+str(self.Y)
	
	def __str__(self):
		return 'X:'+str(self.X)+' Y:'+str(self.Y)
	
def convertXML(filename):
	tree = etree.parse(filename)
	root = tree.getroot()
	models = []
	for model_xml in root.findall('model'):
		models.append(model(model_xml))
	return models

def printResults(models):
	for m in models:
		print m
		m_c = list(set([round(t.C, 3) for t in m.tests]))
		m_c.sort()
		m_gamma = list(set([round(t.gamma, 3) for t in m.tests]))
		m_gamma.sort()
		print '\t'+str(m_c)
		print '\t'+str(m_gamma)
		print '\t'+str(m.tests[0])

def plotResults(m, show=False, prefix=""):
	fig = pl.figure()
	mainfig = fig.add_subplot(1,1,1)
	X = [t.gamma for t in m.tests]
	Y = [t.C for t in m.tests]
	S = np.array([t.error * 100 for t in m.tests])
	S = S-min(S) + 1
	S = S.tolist()
	mainfig.scatter(X, Y, c='k', s=S)
	mainfig.set_xlabel('gamma')
	mainfig.set_ylabel('C')
	mainfig.set_xlim(min(X)*.1, max(X)*10)
	mainfig.set_title(str(m))
	mainfig.set_yscale('log')
	mainfig.set_xscale('log')
	if show:
		pl.show()
	else:
		pl.savefig(os.path.join(prefix,m.X+'-'+m.Y+'.png'))

def plotGridErrors(logfn, outprefix="griderrors"):
	models = convertXML(logfn)
	for m in models:
		plotResults(m,prefix=outprefix)

if __name__ == '__main__':
	if len(sys.argv<2):
		print "Usage: python IV.py <log.xml> <folder in which to put plots (defaults to \"gridplots/\")>"
	if len(sys.argv)<4:
		prefix = "griderrors"
	else:
		prefix = sys.argv[2]
	investigate(sys.argv[1],prefix)