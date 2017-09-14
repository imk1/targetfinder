import sys
import numpy as np
import pandas as pd
from sklearn.cross_validation import StratifiedKFold, cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
import classify as cl

def recursive_elimination(clf, X, y):
    cv = StratifiedKFold(y, n_folds=10, shuffle=True,
                         random_state=0)
    while True:
        f1 = cross_val_score(clf, X, y, scoring='f1', cv=cv,
                             n_jobs=5)
        clf.fit(X, y)
        importances = pd.Series(clf.feature_importances_,
                            index=X.columns).sort_values(ascending=False)
        lowest = importances.tail(1)
        X = X.drop(lowest.index, axis=1)
        fields = [ lowest.index[0], f1.mean(), f1.std() ]
        print('\t'.join([ str(f) for f in fields ]))
        if len(X.columns) == 0:
            return

def permute(clf, X, y):
    pass

def oob(clf, X, y):
    clf.fit(X, y)
    importances = pd.Series(clf.feature_importances_,
                            index=X.columns).sort_values(ascending=False)
    pd.set_option('display.max_rows', len(importances))
    print(importances)

if __name__ == '__main__':
    training_file_name = sys.argv[1]
    metric = sys.argv[2]

    predictors_df, labels = cl.load_data(training_file_name)
    clf = cl.get_clf()

    if 'recur' in metric:
        recursive_elimination(clf, predictors_df, labels)
    if 'permut' in metric:
        permute(clf, predictors_df, labels)
    if 'oob' in metric:
        oob(clf, predictors_df, labels)
