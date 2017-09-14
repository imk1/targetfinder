import sys
import pandas as pd
import itertools
import numpy as np
import scipy.stats as ss
from sklearn.tree import  _tree
import treeinterpreter as ti
import classify as cl
from sklearn.ensemble import GradientBoostingClassifier

def tree_feature_paths(tree, features, num_paths):
    paths = ti._get_tree_paths(tree, 0)
    num_paths += len(paths)
    
    features_shared_count = np.zeros((len(features), len(features)))
    features_count = np.zeros(len(features))
    for path in paths:
        path_feature_set = set(path)
        # Count the number of times a feature appears in a path.
        for node in path_feature_set:
            feature = tree.feature[node]
            if feature == _tree.TREE_UNDEFINED:
                continue
            features_count[feature] += 1
        # Count the number of times a pair of features appears in a path.
        for node_1, node_2 in itertools.product(path_feature_set,
                                                repeat=2):
            feature_1 = tree.feature[node_1]
            feature_2 = tree.feature[node_2]
            if (feature_1 == _tree.TREE_UNDEFINED or
                feature_2 == _tree.TREE_UNDEFINED):
                continue
            features_shared_count[feature_1, feature_2] += 1
    return features_count, features_shared_count, num_paths

def forest_feature_paths(forest, features):
	num_paths = 0
	features_count = np.zeros(len(features))
	features_shared_count = np.zeros((len(features), len(features)))
	for estimator in forest.estimators_:
		tree = estimator[0].tree_  # For Sklearn Gradient boosting
		#tree = estimator[1].tree_ # For Sklearn Random Forest
		tree_count, tree_shared_count, num_paths = tree_feature_paths(tree, features, num_paths)
		features_count = np.add(features_count, tree_count)
		features_shared_count = np.add(features_shared_count,
                                       tree_shared_count)
	return features_count, features_shared_count, num_paths

if __name__ == '__main__':
    X, y = cl.load_data(sys.argv[1])
    clf = cl.get_clf()
    clf.fit(X, y)
    features = X.columns
    count, shared_count, num_paths = forest_feature_paths(clf, features)

    for f1, feature_1 in enumerate(features):
        for f2, feature_2 in enumerate(features):
            if f1 >= f2:
                continue
            f1_f2 = shared_count[f1, f2]
            f1_not_f2 = count[f1] - f1_f2
            f2_not_f1 = count[f2] - f1_f2
            not_f1_not_f2 = num_paths - (f1_f2 + f1_not_f2 + f2_not_f1)
            contingency = [ [ f1_f2,     f2_not_f1 ],
                            [ f1_not_f2, not_f1_not_f2 ] ]
            odds_ratio, p_val = ss.fisher_exact(contingency)

            fields = [ feature_1, feature_2, p_val,
                       f1_f2, f1_not_f2, f2_not_f1, not_f1_not_f2 ]
            print('\t'.join([ str(f) for f in fields ]))
