import sys
import pandas as pd
import numpy as np

from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from voting import VotingEnsemble

def load_data(training_file_name):
    nonpredictors = [
        'enhancer_chrom', 'enhancer_start', 'enhancer_end',
        'promoter_chrom', 'promoter_start', 'promoter_end',
        'window_chrom', 'window_start', 'window_end',
        'window_name',
        'active_promoters_in_window', 'interactions_in_window', 'bin',
        #'enhancer_distance_to_promoter',
        'label']

    training_df = pd.read_hdf(training_file_name, 'training')
    predictors_df = training_df.drop(nonpredictors, axis=1)
    labels = training_df['label']
    return predictors_df, labels

def get_clf():
#    return DecisionTreeClassifier()
#    return LogisticRegression()
    return GradientBoostingClassifier(n_estimators=4000,
                                      learning_rate=0.1, max_depth=5,
                                      max_features='log2',
                                      random_state=0)
    
if __name__ == '__main__':
    predictors_df, labels = load_data(sys.argv[1])
    clf = get_clf()
    
    cv = StratifiedKFold(y=labels, n_folds=10, shuffle=True, random_state=0)
    
    for score_fn in [ 'f1' ]:#, 'precision', 'recall', 'f1' ]:
        print(score_fn)
        scores = cross_val_score(clf, predictors_df, labels, scoring=score_fn,
                                 cv=cv, n_jobs=-1)
        print('{:2f} {:2f}'.format(scores.mean(), scores.std()))

    clf.fit(predictors_df, labels)
    importances = pd.Series(clf.feature_importances_,
                            index=predictors_df.columns).sort_values(ascending=False)
    print(importances.head(25))
