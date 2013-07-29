import numpy as np

class MisclassUncert:
	def __init__(self, _ED, _OD):
		self.ED = _ED
		self.OD = _OD
		self.ind_list = _OD.indAttr
		self.dep_list = _OD.scaledDepAttr()
		self.bins = _OD.bins
		
	def misclassUncertainty(self, ind_values, ignore_eps=False, verbose=False):
		if ignore_eps:
			return self.epsilonInsensitive(ind_values, verbose)
		ind_values = np.atleast_1d(ind_values)
		errors = np.zeros(len(ind_values))
		
		# Set up our predicted values and 
		predictionVectors =self.ED.getExpectationsAt(np.atleast_2d(self.ind_list).T, returnScaled=True)
		# Get their histogram of expected frequencies (each element is the width of the respective bin).
		hist = np.array(self.bins + [1]) - np.array([0] + self.bins)
		
		# Iterate through the points we want to calculate error for
		weight_sums = np.zeros(len(ind_values))
		for i,value in enumerate(ind_values):
			observed = np.zeros(len(self.bins)+1)
			
			#Calculate the weights at this point.
			weights = self.OD.weightFunction(value)
			weight_sums[i] = np.sum(weights)
			weighted_hist = hist*weight_sums[i]

			# Calculate the (weighted) observed frequency of designs within each histogram bin at this point.
			# Use this array to keep track of which points appear in the highest bin (are above all boundaries)
			inLastBin = np.ones(len(weights))
			# Enumerate through the observed values and find which bins they fall into
			for k,point in enumerate(self.dep_list):
				k_index = self.OD.listToContours[k]
				# Enumerate through the bins starting with the smallest one
				for j,b in enumerate(self.bins):
					# if a point occurs bellow the bin boundary then it is in that bin since we are moving up through the bins
					if point < predictionVectors[b][k_index]:
						inLastBin[k] = 0
						observed[j] += weights[k]
						break
			observed[-1] = sum(weights*inLastBin)

			#Store the percentage difference between the predicted and observed distributions as the error
			diff = np.abs(observed-weighted_hist)			
			if weight_sums[i] > 0:
				# We divide the difference by two because each error is seen twice - once in the bin it came from and once in the bin it went to.  eg: if we expect [1 1 1] and get [1 2 0] the calculated distance will be 2, but only one moved.
				errors[i] = np.sum(diff/weight_sums[i]*0.5)
			else:
				print "Zero-weighted element found, returning a perfect goodness-of-fit due to lack of evidence - the uncertainty component should compensate."
				errors[i] = 0
		return errors
		
	def epsilonInsensitive(self, ind_values, verbose=False):
		ind_values = np.atleast_1d(ind_values)
		errors = np.zeros(len(ind_values))
		
		# Set up our predicted values and 
		predictionVectors =self.ED.getExpectationsAt(np.atleast_2d(self.ind_list).T, returnScaled=True)
		# Get their histogram of expected frequencies (each element is the width of the respective bin).
		hist = np.array(self.bins + [1]) - np.array([0] + self.bins)
		
		epsilon = self.ED.getParams()['epsilon']
		# Iterate through the points we want to calculate error for
		weight_sums = np.zeros(len(ind_values))
		for i,value in enumerate(ind_values):
			observed = np.zeros(len(self.bins)+1)
			observedWithinEpsilon = np.zeros(len(self.bins))
			
			#Calculate the weights at this point.
			weights = self.OD.weightFunction(value)
			weight_sums[i] = np.sum(weights)
			weighted_hist = hist*weight_sums[i]

			# Calculate the (weighted) observed frequency of designs within each histogram bin at this point.
			# Use this array to keep track of which points appear in the highest bin (are above all boundaries)
			inLastBin = np.ones(len(weights))
			# Enumerate through the observed values and find which bins they fall into
			for k,point in enumerate(self.dep_list):
				k_index = self.OD.listToContours[k]
				# Enumerate through the bins starting with the smallest one
				for j,b in enumerate(self.bins):
					# if a point occurs bellow the bin boundary then it is in that bin since we are moving up through the bins
					if point < predictionVectors[b][k_index] - epsilon * 0.5:
						inLastBin[k_index] = 0
						observed[j] += weights[k_index]
						break
					elif point < predictionVectors[b][k_index] + epsilon * 0.5:
						inLastBin[k_index] = 0
						overlapIndex = j+1
						spreadCount = 1
						#While this point is still within epsilon of the next bin, spread its weights out over the next few bins.
						while overlapIndex < len(self.bins) and point > predictionVectors[self.bins[overlapIndex]][k_index] - epsilon * 0.5:
							overlapIndex += 1
							spreadCount += 1
						for v in xrange(j,overlapIndex):
							observedWithinEpsilon[v] += weights[k_index]/spreadCount
						break
			observed[-1] = sum(weights*inLastBin)
			diffs = np.subtract(observed,weighted_hist)
			if verbose:
				observedString = str(round(observed[0],1))
				for index in xrange(len(self.bins)):
					observedString = observedString+" ("+str(round(observedWithinEpsilon[index],1))+") "+str(round(observed[index+1],1))
				print "of",weight_sums[i],"found",np.sum(observed),"in contours and",np.sum(observedWithinEpsilon),"within epsilon bands."
				print (weight_sums[i]-np.sum(observed)-np.sum(observedWithinEpsilon))," weight unaccounted for."
				print "hist:",np.around(weighted_hist,1)
				print "---------------PRE FOLD-IN"
				print "observed:",observedString
				print "diffs:",np.around(diffs,1)
				print np.sum(observed)-np.sum(weighted_hist),"difference between distribution sums"
			#Start with the first bin that has a positive error (ie: too many points in it even after the within-epsilon designs were removed)
			startbin = np.argmax(diffs)
			#print "starting at bin",startbin,"with",diffs[startbin]
			#For each bin lower than startbin in descending order
			for j in reversed(xrange(1,startbin+1)):
				if observedWithinEpsilon[j-1] > 0 and diffs[j] < 0: #In case this bin is still lower than it could be and we can add some of this epsilon to it
					max_addable = min(abs(diffs[j]),observedWithinEpsilon[j-1]) # How much error can we shift from the epsilon band into the observed bin?
					observed[j] += max_addable
					observedWithinEpsilon[j-1] -= max_addable
					diffs[j] = observed[j]-weighted_hist[j]
				if observedWithinEpsilon[j-1] > 0 and diffs[j-1] < 0: #Otherwise the next bin is also over its expected distribution, so the epsilon band is all error
					max_addable = min(abs(diffs[j-1]),observedWithinEpsilon[j-1]) # How much error can we shift from the epsilon band into the observed bin?
					observed[j-1] += max_addable
					observedWithinEpsilon[j-1] -= max_addable
					diffs[j-1] = observed[j-1]-weighted_hist[j-1]
				observed[j-1] += observedWithinEpsilon[j-1] # Add back in any remaining error
				observedWithinEpsilon[j-1] = 0
			#For each bin higher than startbin in ascending order
			for j in xrange(startbin,len(observed)-1):
				if observedWithinEpsilon[j] > 0 and diffs[j] < 0: #In case this bin is still lower than it could be and we can add some of this epsilon to it
					max_addable = min(abs(diffs[j]),observedWithinEpsilon[j]) # How much error can we shift from the epsilon band into the observed bin?
					observed[j] += max_addable
					observedWithinEpsilon[j] -= max_addable
					diffs[j] = observed[j]-weighted_hist[j]
				if observedWithinEpsilon[j] > 0 and diffs[j+1] < 0: #Otherwise the next bin is also over its expected distribution, so the the epsilon band is all error
					max_addable = min(abs(diffs[j+1]),observedWithinEpsilon[j]) # How much error can we shift from the epsilon band into the observed bin?
					observed[j+1] += max_addable
					observedWithinEpsilon[j] -= max_addable
					diffs[j+1] = observed[j+1]-weighted_hist[j+1]
				observed[j+1] += observedWithinEpsilon[j] # Add back in any remaining error
				observedWithinEpsilon[j] = 0
			#Now we've folded back in all the epsilon bands in an optimal way, store the percentage difference between the predicted and observed distributions as the error
			diffs = np.subtract(observed,weighted_hist)
			diff = np.sum(np.abs(diffs))
			if weight_sums[i] > 0:
				# We divide the difference by two because each error is seen twice - once in the bin it came from and once in the bin it went to.  eg: if we expect [1 1 1] and get [1 2 0] the calculated distance will be 2, but only one moved.
				errors[i] = np.sum(diff/weight_sums[i]*0.5)
			else:
				print "Zero-weighted element found, returning a perfect goodness-of-fit due to lack of evidence - the uncertainty component should compensate."
				errors[i] = 0
			if verbose:
				print "---------------POST FOLD-IN"
				print "observed:",np.around(observed,1)
				print "diffs:",np.around(diffs,1)
				print np.sum(observed)-np.sum(weighted_hist),"difference between distribution sums"
				regular = self.misclassUncertainty(value, ignore_eps=False)
				print 'Value:',value,'E-Insensitive:',errors[i],'Regular:',regular, 'Difference:',regular-errors[i]
				print "--------------------------"
				print ""
				print ""
		return errors
