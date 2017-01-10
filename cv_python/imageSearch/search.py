import colordescriptor 
import structuredescriptor
import searcher
import argparse
import cv2

searchArgParser = argparse.ArgumentParser()
searchArgParser.add_argument("-c", "--colorindex", required = True, help = "Path to where the computed color index will be stored")
searchArgParser.add_argument("-s", "--structureindex", required = True, help = "Path to where the computed structure index will be stored")
searchArgParser.add_argument("-q", "--query", required = True, help = "Path to the query image")
searchArgParser.add_argument("-r", "--resultpath", required = True, help = "Path to the result path")
searchArguments = vars(searchArgParser.parse_args())

idealBins = (8, 12, 3)
idealDimension = (16, 16)

colorDescriptor = colordescriptor.ColorDescriptor(idealBins)
structureDescriptor = structuredescriptor.StructureDescriptor(idealDimension)
queryImage = cv2.imread(searchArguments["query"])
colorIndexPath = searchArguments["colorindex"]
structureIndexPath = searchArguments["structureindex"]
resultPath = searchArguments["resultpath"]

queryFeatures = colorDescriptor.describe(queryImage)
queryStructures = structureDescriptor.describe(queryImage)

imageSearcher = searcher.Searcher(colorIndexPath, structureIndexPath)
searchResults = imageSearcher.search(queryFeatures, queryStructures,5)

for imageName, score in searchResults:
    queryResult = cv2.imread(resultPath + "/" + imageName)
    cv2.imshow("Result Score: " + str(int(score)) + " (lower is better)", queryResult)
    cv2.waitKey(0)

cv2.imshow("Query", queryImage)
cv2.waitKey(0)


