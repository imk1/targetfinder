import sys
import chromatics
from glob import glob
import pandas as pd

def createHiCGenerator(HiCDataFileNameList, HiCFileNamePrefix):
	# Create a HiC generator
	HiCGenerators = []
	for i in range(len(HiCDataFileNameList)):
		# Iterate through the HiC files
		HiCDataFileName = HiCDataFileNameList[i]
		hic_df = chromatics.read_bed(glob(HiCDataFileNameList[i])[0], 
					names=chromatics.hic_bed_columns, 
					usecols=chromatics.hic_bed_columns)
		hic_df['name'] = HiCDataFileName + '(Hi-C_Sum)'
		HiCFileName = HiCFileNamePrefix + str(i) + ".bed.gz"
		chromatics.write_bed(hic_df, HiCFileName, compression='gzip')
		HiCGenerators.append((chromatics.generate_average_signal_features, HiCFileName))
	return HiCGenerators
	
def createbigwigGenerator(bigwigDataFileNameList, regionWidth=200):
	# Create a list of bigwig generators
	bigwigGenerators = []
	for bigwigDataFileName in bigwigDataFileNameList:
		# Iterate through the bigwig files
		bigwigGenerators.append((chromatics.get_bigwig_features, bigwigDataFileName))
		bigwigGenerators.append((chromatics.get_average_bigwig_features, (bigwigDataFileName, regionWidth)))
	return bigwigGenerators

def createPeakGenerator(peakDataFileNameList, peaksFileName):
	# Create a peak generator
	assays = []
	for peakDataFileName in peakDataFileNameList:
		# Iterate through the peak data files
		columns = chromatics.broadpeak_bed_columns if peakDataFileName.endswith('broadPeak.gz') else chromatics.narrowpeak_bed_columns
		assay_df = chromatics.read_bed(peakDataFileName.format(peakDataFileName), names = columns, usecols = chromatics.generic_bed_columns + ['signal_value'])
		assay_df['name'] = peakDataFileName
		assays.append(assay_df)
	peaks_df = pd.concat(assays, ignore_index = True)
	chromatics.write_bed(peaks_df, peaksFileName, compression = 'gzip')
	return (chromatics.generate_average_signal_features, peaksFileName)
	
def createClosestPositionGeneratorList(peakDataFileNameList, maxDistance, logPositions, positionCol, offset=True):
	# Create a list of closest position generators
	closestPositionGenerators = []
	for peakDataFileName in peakDataFileNameList:
		# Iterate through the peak data files
		if (not peakDataFileName.endswith('broadPeak.gz')):
			# The peak file is a narrowPeak file, so create a position generator for it
			closestPositionGenerators.append((chromatics.get_closest_position_features, (peakDataFileName, maxDistance, logPositions, positionCol, offset)))
	return closestPositionGenerators
	
def createIntersectionGeneratorList(bedDataFileNameList):
	# Create a list of intersection generators
	intersectionPositionGenerators = []
	for bedDataFileName in bedDataFileNameList:
		# Iterate through the bed data files
		intersectionPositionGenerators.append((chromatics.generate_intersection_features, bedDataFileName))
	return intersectionPositionGenerators
