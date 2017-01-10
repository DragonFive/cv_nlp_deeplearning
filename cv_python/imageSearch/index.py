import colordescriptor
import structuredescriptor
import glob
import argparse
import cv2

searchArgParser = argparse.ArgumentParser()
searchArgParser.add_argument("-d", "--dataset", required = True, help = "Path to the directory that contains the images to be indexed")
searchArgParser.add_argument("-c", "--colorindex", required = True, help = "Path to where the computed color index will be stored")
searchArgParser.add_argument("-s", "--structureindex", required = True, help = "Path to where the computed structure index will be stored")
arguments = vars(searchArgParser.parse_args())

idealBins = (8, 12, 3)
colorDesriptor = colordescriptor.ColorDescriptor(idealBins)

output = open(arguments["colorindex"], "w")

for imagePath in glob.glob(arguments["dataset"] + "/*.png"):
    imageName = imagePath[imagePath.rfind("/") + 1 : ]
    image = cv2.imread(imagePath)
    features = colorDesriptor.describe(image)
    # write features to file
    features = [str(feature).replace("\n", "") for feature in features]
    output.write("%s,%s\n" % (imageName, ",".join(features)))
# close index file
output.close()

idealDimension = (16, 16)
structureDescriptor = structuredescriptor.StructureDescriptor(idealDimension)

output = open(arguments["structureindex"], "w")

for imagePath in glob.glob("dataset" + "/*.png"):
    imageName = imagePath[imagePath.rfind("/") + 1 : ]
    image = cv2.imread(imagePath)
    structures = structureDescriptor.describe(image)
    # write structures to file
    structures = [str(structure).replace("\n", "") for structure in structures]
    output.write("%s,%s\n" % (imageName, ",".join(structures)))
# close index file
output.close()
