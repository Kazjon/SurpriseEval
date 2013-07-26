# Author: Kazjon Grace <k.grace@uncc.edu>
# Author: Katherine Brady <katherine.a.brady@vanderbilt.edu>

import numpy as np

class DistanceUncertainty:
	
	def __init__(self, _OD):
		self.OD = _OD
		self.ind_list = _OD.ind_list
	
	def distToNearestPoint(self,value):
		idx = (np.abs(self.ind_list-value)).argmin()
		return abs(self.ind_list[idx]-value)
	
	def distanceUncertainty(self,values):
		values = np.atleast_1d(values)
		errors = np.zeros(len(values))
		for i,value in enumerate(values):
			errors[i] = 1 - np.exp(self.OD.weightFactor*(-(self.distToNearestPoint(value) ** 2)))
		return errors
