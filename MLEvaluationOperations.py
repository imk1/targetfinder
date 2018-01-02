import numpy as np
from sklearn.metrics import precision_recall_curve, auc
import rpy2.robjects as robjects
from rpy2.robjects.packages import STAP

# Compute the specificity
def negative_accuracy(labels, predictedLabels):
	negatives = np.array(labels == 0)
	numTrueNegatives = (predictedLabels[negatives] == 0).sum()
	numNegatives = negatives.sum()
	return float(numTrueNegatives)/numNegatives
	
# Get the precision
def precision(y_true, y_score, decisionBoundary=0.5):
	precision, recall, thresholds = precision_recall_curve(y_true, y_score)
	decision_index = next(i for i, x in enumerate(thresholds) if x < decisionBoundary)
	return precision[decision_index]
	
# Get the AUPRC
def auprc(y_true, y_score, path=""):
	with open(path + 'PRROC.R', 'r') as f:#load in the R code. 
		r_fxn_string = f.read()
	r_auc_func = STAP(r_fxn_string, "auc_func")
	r_auprc_results = r_auc_func.pr_curve(scores_class0 = robjects.vectors.FloatVector(y_score), weights_class0 = robjects.vectors.FloatVector(y_true))
	AUPRC = float(float(r_auprc_results.rx('auc.davis.goadrich')[0][0]))
	return AUPRC

# Get the recall at a specific FDR
def recall_at_fdr(y_true, y_score, fdr_cutoff=0.2):
	precision, recall, thresholds = precision_recall_curve(y_true, y_score)
	fdr = 1- precision
	cutoff_index = next(i for i, x in enumerate(fdr) if x < fdr_cutoff)
	return recall[cutoff_index]
