# -*- coding: utf-8 -*-
# import the necessary packages
import numpy as np
import csv
import re
 
class Searcher:
	def __init__(self, colorIndexPath, structureIndexPath):
		# store our index path
		self.indexPath, self.structureIndexPath = colorIndexPath, structureIndexPath

	def chi2_distance(self, histA, histB, eps = 1e-10):
		# compute the chi-squared distance
		d = 0.5 * np.sum([((a - b) ** 2) / (a + b + eps)
		    for (a, b) in zip(histA, histB)])

		# return the chi-squared distance
		return d
	
	# 这里的返回值是一个字典，内容是图像名和特征的对应;
	def searchByColor(self, queryFeatures):
		# initialize our dictionary of results
		results = {}

		# open the index file for reading
		with open(self.indexPath) as f:
			# initialize the CSV reader
			reader = csv.reader(f)
 
			# loop over the rows in the index
			for row in reader:
				# parse out the image ID and features, then compute the
				# chi-squared distance between the features in our index
				# and our query features
				features = [float(x) for x in row[1:]]
				d = self.chi2_distance(features, queryFeatures)
 
				# now that we have the distance between the two feature
				# vectors, we can udpate the results dictionary -- the
				# key is the current image ID in the index and the
				# value is the distance we just computed, representing
				# how 'similar' the image in the index is to our query
				results[row[0]] = d
 
			# close the reader
			f.close()
 		return results
		# 这里还不需要排序
		# sort our results, so that the smaller distances (i.e. the
		# more relevant images are at the front of the list)
		#results = sorted([(v, k) for (k, v) in results.items()])
 
		# return our (limited) results
		#return results[:limit]

	def structureDistance(self, structures, queryStructures, eps = 1e-5):
		distance = 0
		normalizeRatio = 5e3
		for index in xrange(len(queryStructures)):
			for subIndex in xrange(len(queryStructures[index])):
				a = structures[index][subIndex]
				b = queryStructures[index][subIndex]
				distance += (a - b) ** 2 / (a + b + eps)
		return distance / normalizeRatio
    
    # 把矩阵转化为向量;
	def transformRawQuery(self, rawQueryStructures):
		queryStructures = []
		for substructure in rawQueryStructures:
			structure = []
			for line in substructure:
				for tripleColor in line:
					structure.append(float(tripleColor))
			queryStructures.append(structure)
		return queryStructures
	    
	def searchByStructure(self, rawQueryStructures):
		searchResults = {}
		queryStructures = self.transformRawQuery(rawQueryStructures)
		with open(self.structureIndexPath) as indexFile:
			reader = csv.reader(indexFile)
			for line in reader:
				structures = []
				for structure in line[1:]:
					structure = structure.replace("[", "").replace("]", "")
					structure = re.split("\s+", structure)
					if structure[0] == "":
						structure = structure[1:]
					structure = [float(eachValue) for eachValue in structure]
					structures.append(structure)
				distance = self.structureDistance(structures, queryStructures)
				searchResults[line[0]] = distance
			indexFile.close()
		# print "structure", sorted(searchResults.iteritems(), key = lambda item: item[1], reverse = False)
		return searchResults


	
	def search(self, queryFeatures, rawQueryStructures, limit = 3):
		featureResults = self.searchByColor(queryFeatures)
		structureResults = self.searchByStructure(rawQueryStructures)
		results = {}
		for key, value in featureResults.iteritems():
			results[key] = value + structureResults[key]
		results = sorted(results.iteritems(), key = lambda item: item[1], reverse = False)
		return results[ : limit]

