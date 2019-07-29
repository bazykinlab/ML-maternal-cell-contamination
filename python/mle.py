from copy import deepcopy
import numpy as np
from scipy.special import factorial

from sklearn.base import BaseEstimator, ClassifierMixin

class MLEClassifier(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self, idx=[19, 20]):
        """
        Called when initializing the classifier
        """
        self.idx = idx



    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.
        """

        return self

    # def predict(self, X, y=None):
    #     probs = self.predict_proba(X)
    #     preds = np.argmax(probs, axis=1)

    #     return preds


    def predict(self, X, y=None):

    # def predict_proba(self, X, y=None):
        ab_gts = np.argmax(X[:, -10:-7], axis=1)
        mo_gts = np.argmax(X[:, -7:-4], axis=1)
        # predictions = deepcopy(ab_gts)
        idx_candidates = np.logical_and(ab_gts==1,  mo_gts==1)
        contaminations = X[:, -1].astype(np.float64)
        
        ad0 = X[:, self.idx[0]].astype(np.int64)
        ad1 = X[:, self.idx[1]].astype(np.int64)

        with np.errstate(divide='ignore', invalid='ignore'):
            p0 = (contaminations/2)**ad1 * (1 - contaminations/2)**ad0
            p1 = (1/2)**ad1*(1/2)**ad0
            p2 = (contaminations/2)**ad0 * (1 - contaminations/2)**ad1

        probs = np.array([p0, p1, p2]).T
        probs[~idx_candidates, :] = 0
        probs[~idx_candidates, ab_gts[~idx_candidates]] = 1
        # probs[np.arange(preds.shape[0]), preds] = 1


        preds = np.argmax(probs, axis=1)

        return preds

    def predict_proba(self, X, y=None):
        preds = self.predict(X)
        probs = np.zeros((preds.shape[0], 3))
        probs[np.arange(preds.shape[0]), preds] = 1

        return probs


    def score(self, X, y=None):
        return(np.sum(self.predict(X) == y)) 