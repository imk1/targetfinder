import numpy as np
from sklearn.cross_validation import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier

class VotingEnsemble:
    def __init__(self, n_estimators=10, max_depth=5, cutoff=5, n_models=5):
        self.models = []
        self.feature_importances_ = []
        self.n_estimators = n_estimators
        self.n_models = n_models
        self.max_depth = max_depth
        self.cutoff = cutoff

    def fit(self, X, y):
        skf = StratifiedKFold(y, n_folds=self.n_models)
        for _, split in skf:
            split = np.array(split)
            model = RandomForestClassifier(n_estimators=self.n_estimators,
                                           max_depth=self.max_depth)
#            model = GradientBoostingClassifier(n_estimators=self.n_estimators,
#                                               max_depth=self.max_depth)
            model.fit(X[split, :], y[split])
            self.models.append(model)
            if len(self.feature_importances_) == 0:
                self.feature_importances_ = model.feature_importances_
            else:
                self.feature_importances_ = np.add(self.feature_importances_,
                                                   model.feature_importances_)
        self.feature_importances_ /= self.n_models

    def predict(self, X):
        y_sum = np.zeros(X.shape[0], dtype='int')
        for model in self.models:
            y_pred = [ int(y_i) for y_i in model.predict(X) ]
            y_sum = np.add(y_sum, y_pred)
        return np.array([ 1 if y_i >= self.cutoff else 0 for y_i in y_sum ])

    def get_params(self, deep=True):
        return { 'n_estimators':self.n_estimators,
                 'max_depth':self.max_depth,
                 'cutoff':self.cutoff }
