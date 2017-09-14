import os
import sys
import argparse
import gzip
import pandas as pd
from scipy.stats import pearsonr, spearmanr, fisher_exact
from sklearn.metrics import *
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from createGenerators import *
from auPRG_with_rpy2 import auPRG_R # REQUIRES r/3.2.0
from MLEvaluationOperations import *
from interpret import recursive_elimination
from interact import forest_feature_paths
import warnings
warnings.filterwarnings("ignore")

def parseArgument():
	# Parse the input
	parser=argparse.ArgumentParser(description=\
			"Train a QTL overlap classifier")
	parser.add_argument("--positiveSetFileName", required=False,\
			help='Name of file with the positive set')
	parser.add_argument("--negativeSetFileName", required=False,\
			help='Name of file with the negative set')
	parser.add_argument("--positionCol", type=int, required=False, default=1, \
			help='Column of input files with the SNP position (0-indexed)')
	parser.add_argument("--regionHalfWidth", type=int, required=False, default=100, \
			help='Half width of regions that will be considered for signal features (regions will be SNP +/- regionHalfWidth)')
	parser.add_argument("--testChroms", nargs='*', required=False,\
			help='Chromosomes that will be in the test set')
	parser.add_argument("--csvTrainFileName", required=False,\
			help='Name of csv where the training QTL regions, positions, and labels will be recorded')
	parser.add_argument("--csvTestFileName", required=False,\
			help='Name of csv where the testing QTL regions, positions, and labels will be recorded')
	parser.add_argument("--dataFileNameListFileName", required=False,\
			help='Name of file with list of the files with all of the data')
	parser.add_argument("--maxDistance", type=int, required=False, default=5000, \
			help='Maximum distance allowed for computing distances between SNPs and peak summits')
	parser.add_argument("--HiCFileNamePrefix", required=False,\
			default="hi-c", \
			help='Name of file where featurized HiC data will be written, should not end with ".bed.gz"')
	parser.add_argument("--peaksFileName", required=False,\
			default="peaks.bed.gz", \
			help='Name of file where featurized peak data will be written')
	parser.add_argument("--chunkSize", type=int, required=False, default=2**20, \
			help='Size of chunks for featurization')
	parser.add_argument("--csvDataFileName", required=False,\
			default="training.csv", \
			help='Name of file where training data will be written')
	parser.add_argument("--featureCorrMatFileNamePrefix", required=False,\
			default="trainingFeatureCorrMat", \
			help='Prefix of name of file where feature correlation matrix and p-values will be written')
	parser.add_argument("--classifier", required=False,\
			default="DecisionTree", \
			help='Type of classifier')
	parser.add_argument("--learningRate", type=float, required=False, default=0.1, \
			help='The learning rate for the classifier')
	parser.add_argument("--maxDepth", type=int, required=False, default=10, \
			help='Maximum depth for tree classifiers')
	parser.add_argument("--degree", type=int, required=False, default=2, \
			help='The degree for the polynomial SVM')
	parser.add_argument("--numCrossValFolds", type=int, required=False, default=10, \
			help='Number of folds for cross-validation')
	parser.add_argument("--treeFileNamePrefix", required=False, \
			help='The prefix of the file names where the tree will be recorded, should not end with .')
	parser.add_argument("--RuleFitFileNamePrefix", required=False, \
			help='The name of the file where the data frames for running RuleFit will be stored, should not end with _')
	parser.add_argument("--regression", action='store_true', required=False,\
			help='Train a regression model instead of a classification model')
	parser.add_argument("--createFeatureCorrMat", action='store_true', required=False,\
			help='Create a feature correlation matrix with the training data')
	parser.add_argument("--classifyOnly", action='store_true', required=False,\
			help='Train the classifier from an existing hdf5 file')
	parser.add_argument("--logPeakPositions", action='store_true', required=False,\
			help='log the peak positions')
	parser.add_argument("--excludeHiC", action='store_true', required=False,\
			help='Exclude the HiC data')
	parser.add_argument("--excludePeakSignals", action='store_true', required=False,\
			help='Exclude the peak signal data')
	parser.add_argument("--excludePeakPositions", action='store_true', required=False,\
			help='Exclude the peak position data')
	parser.add_argument("--excludeIntersections", action='store_true', required=False,\
			help='Exclude the region intersection data')
	parser.add_argument("--excludegffs", action='store_true', required=False,\
			help='Exclude the peak signal data')
	parser.add_argument("--excludebigwigs", action='store_true', required=False,\
			help='Exclude the bigwig data')
	parser.add_argument("--doRecursiveElimination", action='store_true', required=False,\
			help='Do recursive elimination on features')
	parser.add_argument("--getInteractions", action='store_true', required=False,\
			help='Get interactions between features')
	options = parser.parse_args()
	return options
	
def putSNPsIntoTrainTestSets(SNPFileName, positionCol, testChroms, csvTrainFileUnsorted, csvTestFileUnsorted, givenLabel):
	# Divide SNPs into training and test sets
	SNPFile = ""
	if SNPFileName.endswith("gz"):
		# Use gzip to open the file
		SNPFile = gzip.open(SNPFileName)
	else:
		SNPFile = open(SNPFileName)
	for line in SNPFile:
		# Iterate through the positive set and record each example's region and position in the appropriate file
		lineElements = line.split("\t")
		position = int(lineElements[positionCol])
		if position - options.regionHalfWidth < 0:
			# Ignore the current position because the region around it is before the beginning of the chromosome
			continue
		label = givenLabel
		if givenLabel == None:
			# Get the label/signal from the file
			QTLInfoElements = lineElements[6].split(";")
			label = abs(float(QTLInfoElements[2]))
		if lineElements[0] in testChroms:
			# The current example is on a test chromosome, so record it in the test output file
			csvTestFileUnsorted.write(",".join([lineElements[0], str(position - options.regionHalfWidth), str(position + options.regionHalfWidth), lineElements[0]+":"+str(position), str(position), str(label)]) + "\n")
		else:
			# The current example is not on a test chromosome, so record it in the training output file
			csvTrainFileUnsorted.write(",".join([lineElements[0], str(position - options.regionHalfWidth), str(position + options.regionHalfWidth), lineElements[0]+":"+str(position), str(position), str(label)]) + "\n")
	SNPFile.close()

def generateQTLRegions(options):
	# Create a csv file with QTL regions, positions, and labels
	print("Creating training and test sets!")
	csvTrainFileUnsorted = open(options.csvTrainFileName + "_unsorted", 'w+')
	csvTrainFileUnsorted.write("QTLRegion_chrom,QTLRegion_start,QTLRegion_end,QTLRegion_name,QTLRegion_position,label" + "\n")
	csvTestFileUnsorted = open(options.csvTestFileName + "_unsorted", 'w+')
	csvTestFileUnsorted.write("QTLRegion_chrom,QTLRegion_start,QTLRegion_end,QTLRegion_name,QTLRegion_position,label" + "\n")
	if options.regression:
		# Create a training set for regression
		putSNPsIntoTrainTestSets(options.positiveSetFileName, options.positionCol, options.testChroms, csvTrainFileUnsorted, csvTestFileUnsorted, \
			None)
	else:
		# Create a training set for classification
		putSNPsIntoTrainTestSets(options.positiveSetFileName, options.positionCol, options.testChroms, csvTrainFileUnsorted, csvTestFileUnsorted, 1)
		putSNPsIntoTrainTestSets(options.negativeSetFileName, options.positionCol, options.testChroms, csvTrainFileUnsorted, csvTestFileUnsorted, 0)
	csvTrainFileUnsorted.close()
	csvTestFileUnsorted.close()
	os.system(" ".join(["(head -n 1", options.csvTrainFileName + "_unsorted", "&& tail -n +2", options.csvTrainFileName + "_unsorted", \
		"| sort -u -k1,1 -k5,5n -t ,) >", options.csvTrainFileName]))
	os.system(" ".join(["(head -n 1", options.csvTestFileName + "_unsorted", "&& tail -n +2", options.csvTestFileName + "_unsorted", \
		"| sort -u -k1,1 -k5,5n -t ,) >", options.csvTestFileName]))
	os.remove(options.csvTrainFileName + "_unsorted")
	os.remove(options.csvTestFileName + "_unsorted")
	
def createFeatures(options):
	# Create a feature matrix for all of the examples
	print("Creating features!")
	dataFileNameListFile = open(options.dataFileNameListFileName)
	dataFileNameList = [line.strip() for line in dataFileNameListFile]
	dataFileNameListFile.close()
	generators = []
	HiCDataFileNameList = [dataFileName for dataFileName in dataFileNameList if "sum" in dataFileName]
	if (not options.excludeHiC) and (len(HiCDataFileNameList) > 0):
		# There is a HiC file, so create a generator for it
		generators.extend(createHiCGenerator(HiCDataFileNameList, options.HiCFileNamePrefix))
	peakDataFileNameList = [dataFileName for dataFileName in dataFileNameList if (dataFileName.endswith("narrowPeak.gz") or dataFileName.endswith("broadPeak.gz")) or (("ChIPseq" in dataFileName) or dataFileName.endswith("regionPeak.gz"))]
	if len(peakDataFileNameList) > 0:
		# There are peak files, so create a generator for them
		if (not options.excludePeakSignals):
			# Add a peak signal generator
			generators.append(createPeakGenerator(peakDataFileNameList, options.peaksFileName))
		if (not options.excludePeakPositions):
			# Add a distance from peak position generator
			generators.extend(createClosestPositionGeneratorList(peakDataFileNameList, options.maxDistance, options.logPeakPositions, 9, offset=True))
	bedDataFileNameList = [dataFileName for dataFileName in dataFileNameList if dataFileName.endswith("bed")]
	if (not options.excludeIntersections) and (len(bedDataFileNameList) > 0):
		# There are bed files, so create a generator for them
		generators.extend(createIntersectionGeneratorList(bedDataFileNameList))
	gffDataFileNameList = [dataFileName for dataFileName in dataFileNameList if dataFileName.endswith("gff.gz")]
	if (not options.excludegffs) and (len(gffDataFileNameList) > 0):
		# There are gff files, so create a generator for them
		generators.extend(createClosestPositionGeneratorList(gffDataFileNameList, options.maxDistance, options.logPeakPositions, 3, offset=False))
	bigwigDataFileNameList = [dataFileName for dataFileName in dataFileNameList if dataFileName.endswith("bw")]
	if (not options.excludebigwigs) and (len(bigwigDataFileNameList) > 0):
		# There are bigwig files, so create a generator for them
		generators.extend(createbigwigGenerator(bigwigDataFileNameList, 2*options.regionHalfWidth))
	SNPData = pd.read_csv(options.csvTrainFileName)
	trainingData = chromatics.generate_training(SNPData, ["QTLRegion"], generators, chunk_size = options.chunkSize, n_jobs = 1)
	trainingData = trainingData.set_index(['QTLRegion_name'])
	trainingData.to_csv(options.csvDataFileName, sep = "\t", mode = 'w')
	
def loadFeatures(options):
	# Load the feature data from the hdf5 file
    nonpredictors = [
        'QTLRegion_chrom', 'QTLRegion_start', 'QTLRegion_end',
        'QTLRegion_position', 'label']
    trainingData = pd.read_csv(options.csvDataFileName, sep = "\t", index_col=0)
    trainingDataFilt = trainingData.drop(nonpredictors, axis=1)
    labels = trainingData['label']
    print("The first label is: " + str(labels[0]))
    return trainingDataFilt, labels
    
def createFeatureCorrMat(options, trainingDataFilt):
	# Create a correlation matrix of all of the features
	featureCorrMat = np.zeros((len(trainingDataFilt.columns), len(trainingDataFilt.columns)))
	featureCorrMatpVal = np.zeros((len(trainingDataFilt.columns), len(trainingDataFilt.columns)))
	featureSpearCorrMat = np.zeros((len(trainingDataFilt.columns), len(trainingDataFilt.columns)))
	featureSpearCorrMatpVal = np.zeros((len(trainingDataFilt.columns), len(trainingDataFilt.columns)))
	rowIndex = 0
	for column in trainingDataFilt:
		# Iterate through the features and find each feature's correlation with each other features
		columnIndex = 0
		for otherColumn in trainingDataFilt:
			# Iterate through the other features and find their correlations with the current feature
			(featureCorrMat[rowIndex, columnIndex], featureCorrMatpVal[rowIndex, columnIndex]) = pearsonr(trainingDataFilt[column], trainingDataFilt[otherColumn])
			(featureSpearCorrMat[rowIndex, columnIndex], featureSpearCorrMatpVal[rowIndex, columnIndex]) = spearmanr(trainingDataFilt[column], trainingDataFilt[otherColumn])
			columnIndex = columnIndex + 1
		rowIndex = rowIndex + 1
	np.savetxt(options.featureCorrMatFileNamePrefix + "-featurePearsonCorr.txt", featureCorrMat, fmt='%.4f', delimiter='\t')
	np.savetxt(options.featureCorrMatFileNamePrefix + "-featurePearsonCorrpVal.txt", featureCorrMatpVal, fmt='%.4f', delimiter='\t')
	np.savetxt(options.featureCorrMatFileNamePrefix + "-featureSpearmanCorr.txt", featureSpearCorrMat, fmt='%.4f', delimiter='\t')
	np.savetxt(options.featureCorrMatFileNamePrefix + "-featureSpearmanCorrpVal.txt", featureSpearCorrMatpVal, fmt='%.4f', delimiter='\t')
    
def prepareRuleFit(options, trainingDataFilt, labels):
	# Prepare files for RuleFit3
	labelsArePos = np.where(labels == 1)[0]
	labelsAreNeg = np.where(labels == 0)[0]
	trainingDataFiltPos = trainingDataFilt.iloc[np.array(range(len(labels)))[labelsArePos]]
	trainingDataFiltNeg = trainingDataFilt.iloc[np.array(range(len(labels)))[labelsAreNeg]]
	csvPosFileName = options.RuleFitFileNamePrefix + "_pos.csv"
	csvNegFileName = options.RuleFitFileNamePrefix + "_neg.csv"
	trainingDataFiltPos.to_csv(csvPosFileName, mode = 'w')
	trainingDataFiltNeg.to_csv(csvNegFileName, mode = 'w')
	
def trainClassifierProba(options, trainingDataFilt, labels, classifier):
	print ("Doing cross-validation")
	cv = StratifiedKFold(y=labels, n_folds=options.numCrossValFolds, shuffle=True, random_state=0)
	results = np.zeros((9, 10))
	cvNum = 0
	for trainIndices, validIndices in cv:
		# Iterate through the cross-validation folds, fit a model for each fold, and evaluate that model
		classifier.fit(trainingDataFilt.iloc[trainIndices], labels[trainIndices])
		predictedClasses = classifier.predict(trainingDataFilt.iloc[validIndices])
		predictedProba = classifier.predict_proba(trainingDataFilt.iloc[validIndices])[:,1]
		results[0, cvNum] = accuracy_score(labels[validIndices], predictedClasses)
		results[1, cvNum] = recall_score(labels[validIndices], predictedClasses)
		results[2, cvNum] = negative_accuracy(labels[validIndices], predictedClasses)
		results[3, cvNum] = roc_auc_score(labels[validIndices], predictedProba)
		print ("AUC for fold " + str(cvNum) + ":" + str(results[3, cvNum]))
		results[4, cvNum] = precision(labels[validIndices], predictedProba, decisionBoundary=0.5)
		results[5, cvNum] = auprc(labels[validIndices], predictedProba)
		results[6, cvNum] = auPRG_R(labels[validIndices], predictedProba)
		results[7, cvNum] = f1_score(labels[validIndices], predictedClasses)
		results[8, cvNum] = recall_at_fdr(labels[validIndices], predictedProba, fdr_cutoff=0.2)
		cvNum = cvNum + 1
	meanResults = np.mean(results, axis=1)
	stdResults = np.std(results, axis=1)
	scoreFunctionNum = 0
	for scoreFunction in ['accuracy', 'recall', 'specificity', 'roc_auc', 'precision', 'auprc', 'auprg', 'f1', 'recall at 0.2 FDR']:
		# Iterate through the cross-validation metrics and print the average and standard deviation for each across the cross-validation folds
		print(scoreFunction)
		print('{:2f} {:2f}'.format(meanResults[scoreFunctionNum], stdResults[scoreFunctionNum]))
		scoreFunctionNum = scoreFunctionNum + 1
		
def trainClassifierNonProba(trainingDataFilt, labels, classifier):
	print ("Doing cross-validation")
	cv = StratifiedKFold(y=labels, n_folds=10, shuffle=True, random_state=0)
	results = np.zeros((4, 10))
	cvNum = 0
	for trainIndices, validIndices in cv:
		# Iterate through the cross-validation folds, fit a model for each fold, and evaluate that model
		classifier.fit(trainingDataFilt.iloc[trainIndices], labels[trainIndices])
		predictedClasses = classifier.predict(trainingDataFilt.iloc[validIndices])
		results[0, cvNum] = accuracy_score(labels[validIndices], predictedClasses)
		results[1, cvNum] = recall_score(labels[validIndices], predictedClasses)
		results[2, cvNum] = negative_accuracy(labels[validIndices], predictedClasses)
		results[3, cvNum] = f1_score(labels[validIndices], predictedClasses)
		cvNum = cvNum + 1
	meanResults = np.mean(results, axis=1)
	stdResults = np.std(results, axis=1)
	scoreFunctionNum = 0
	for scoreFunction in ['accuracy', 'recall', 'specificity', 'f1']:
		# Iterate through the cross-validation metrics and print the average and standard deviation for each across the cross-validation folds
		print(scoreFunction)
		print('{:2f} {:2f}'.format(meanResults[scoreFunctionNum], stdResults[scoreFunctionNum]))
		scoreFunctionNum = scoreFunctionNum + 1
	
def getInteractions(trainingDataFilt, classifier):
	NUM_PATHS = 0
	features = trainingDataFilt.columns
	count, shared_count, NUM_PATHS = forest_feature_paths(classifier, features)
	for f1, feature_1 in enumerate(features):
		for f2, feature_2 in enumerate(features):
			if f1 >= f2:
				continue
			f1_f2 = shared_count[f1, f2]
			f1_not_f2 = count[f1] - f1_f2
			f2_not_f1 = count[f2] - f1_f2
			not_f1_not_f2 = NUM_PATHS - (f1_f2 + f1_not_f2 + f2_not_f1)
			contingency = [ [ f1_f2,     f2_not_f1 ],
							[ f1_not_f2, not_f1_not_f2 ] ]
			odds_ratio, p_val = fisher_exact(contingency)

			fields = [ feature_1, feature_2, p_val,
						f1_f2, f1_not_f2, f2_not_f1, not_f1_not_f2 ]
			print('\t'.join([ str(f) for f in fields ]))
	
def classifyQTLs(options, trainingDataFilt, labels):
	# Train the QTL overlap classifier
	print("Training the classifier!")
	classifier = DecisionTreeClassifier(max_depth=options.maxDepth, splitter="random", 
					max_features='log2',
					random_state=0)
	if (options.classifier == "AdaBoost"):
		# Use an AdaBoost classifier
		classifier = AdaBoostClassifier(n_estimators=4000,
						learning_rate=options.learningRate,
						random_state=0)
	elif (options.classifier == "GradientBoosting"):
		# Use a gradient boosting classifier
		classifier = GradientBoostingClassifier(n_estimators=4000,
						learning_rate=options.learningRate, max_depth=options.maxDepth, # maxDepth is usually 5
						max_features='log2',
						random_state=0)
	elif (options.classifier == "LogisticRegression"):
		# Use a logistic regression classifier
		classifier = LogisticRegression(random_state=0)
	elif (options.classifier == "PassiveAgressive"):
		# Use a passive agressive classifier
		classifier = PassiveAggressiveClassifier(fit_intercept=True, n_iter=50, loss='hinge', class_weight='balanced', random_state=0)
	elif (options.classifier == "SVM"):
		# Use a support vector machine classifier
		classifier = LinearSVC(loss='squared_hinge', dual=False, class_weight='balanced', random_state=0)
	elif (options.classifier == "PolySVM"):
		# Use a support vector machine classifier with a polynomial kernel
		classifier = SVC(kernel='poly', degree=options.degree, shrinking=False, class_weight='balanced', max_iter=100000, random_state=0)
	elif (options.classifier == "RuleFit"):
		# Use RuleFit for the classifier
		prepareRuleFit(options, trainingDataFilt, labels)
	if options.classifier != "RuleFit":
		# Evaluate the performance of the classifier
		if (options.classifier != "SVM") and ((options.classifier != "PolySVM") and (options.classifier != "PassiveAgressive")):
			# Train a classifier that can get probabilities
			trainClassifierProba(options, trainingDataFilt, labels, classifier)
		else:
			# Train a classifier that cannot get probabilities
			trainClassifierNonProba(trainingDataFilt, labels, classifier)
		classifier.fit(trainingDataFilt, labels)
		featureImportances = []
		if ((options.classifier != "LogisticRegression") and (options.classifier != "SVM")) and ((options.classifier != "PolySVM") and (options.classifier != "PassiveAgressive")):
			# Get the feature importances, which are not the coefficients
			featureImportances = pd.Series(classifier.feature_importances_,
									index=trainingDataFilt.columns).sort_values(ascending=False)
		elif options.classifier != "PolySVM":
			# The feature importances are the coefficients
			featureImportances = pd.Series(np.abs(classifier.coef_)[:,0],
									index=trainingDataFilt.columns).sort_values(ascending=False)
		if options.classifier != "PolySVM":
			# Print the most important features and their importances
			print(featureImportances.head(25))
		if options.classifier == "DecisionTree":
			# Record a picture of the tree
			export_graphviz(classifier, out_file=options.treeFileNamePrefix + ".dot", class_names=True)
			cmdline = "dot -Tpng " + options.treeFileNamePrefix + ".dot -o " + options.treeFileNamePrefix + ".png"
			os.system(cmdline)
		if (options.doRecursiveElimination) and (((options.classifier != "LogisticRegression") and (options.classifier != "SVM")) and ((options.classifier != "PolySVM") and (options.classifier != "PassiveAgressive"))):
			# Use recursive elimination to get another value for feature importances
			recursive_elimination(classifier, trainingDataFilt, labels)
		if options.getInteractions:
			# Get the interactions between features
			getInteractions(trainingDataFilt, classifier)
		
def regressQTLs(options, trainingDataFilt, labels):
	# Train the QTL overlap classifier
	print("Training the regressor!")
	regressor = DecisionTreeRegressor(max_depth=40, splitter="random", 
					max_features='log2',
					random_state=0)
	if (options.classifier == "AdaBoost"):
		# Use an AdaBoost classifier
		regressor = AdaBoostRegressor(n_estimators=4000,
						learning_rate=options.learningRate,
						random_state=0)
	elif (options.classifier == "GradientBoosting"):
		# Use a gradient boosting classifier
		regressor = GradientBoostingRegressor(n_estimators=4000,
						learning_rate=options.learningRate, max_depth=5,
						max_features='log2',
						random_state=0)
	for scoreFunction in ['mean_squared_error', 'r2']:
		# Iterate through the cross-validation metrics and compute the average for each across the cross-validation folds
		print(scoreFunction)
		cv = StratifiedKFold(y=labels, n_folds=10, shuffle=True, random_state=0)
		scores = cross_val_score(regressor, trainingDataFilt, labels, scoring=scoreFunction,
					cv=cv, n_jobs=-1)
		print('{:2f} {:2f}'.format(scores.mean(), scores.std()))
	regressor.fit(trainingDataFilt, labels)
	featureImportances = pd.Series(regressor.feature_importances_,
							index=trainingDataFilt.columns).sort_values(ascending=False)
	print(featureImportances.head(25))
	if options.classifier == "DecisionTree":
		# Record a picture of the tree
		export_graphviz(regressor, out_file=options.treeFileNamePrefix + ".dot", class_names=True)
		cmdline = "dot -Tpng " + options.treeFileNamePrefix + ".dot -o " + options.treeFileNamePrefix + ".png"
		os.system(cmdline)

if __name__ == '__main__':
	options = parseArgument()
	if not options.classifyOnly:
		# Create the labels and the features
		generateQTLRegions(options)
		createFeatures(options)
	trainingDataFilt, labels = loadFeatures(options)
	if options.createFeatureCorrMat:
		# Create a correlation matrix between all pairs of features
		createFeatureCorrMat(options, trainingDataFilt)
	if options.regression:
		# Train a regression model
		regressQTLs(options, trainingDataFilt, labels)
	else:
		# Train a classification model
		classifyQTLs(options, trainingDataFilt, labels)        
