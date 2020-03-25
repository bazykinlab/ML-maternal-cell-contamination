import numpy as np
from tqdm import tqdm
from scipy.special import binom

from sklearn.base import BaseEstimator, ClassifierMixin


class BayesianClassifier(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self, idx=[19, 20], eps=0):
        """
        Called when initializing the classifier
        """
        self.idx = idx
        self.eps = eps


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
        gts = [0, 1, 2]

        def p_men(m_gt, f_gt):
            if m_gt == f_gt == 0:
                p0 = 1
            elif (m_gt == 1 and f_gt == 0) or (m_gt == 0 and f_gt == 1):
                p0 = 0.5
            elif m_gt == f_gt == 1:
                p0 = 0.25
            else:
                p0 = 0

            if (m_gt == 0 and f_gt == 2) or (m_gt == 2 and f_gt == 0):
                p1 = 1
            elif (m_gt == 0 and f_gt == 0) or (m_gt ==2 and f_gt == 2):
                p1 = 0
            else:
                p1 = 0.5

            if m_gt == f_gt == 2:
                p2 = 1
            elif (m_gt == 1 and f_gt == 2) or (m_gt == 2 and f_gt == 1):
                p2 = 0.5
            elif m_gt == f_gt == 1:
                p2 = 0.25
            else:
                p2 = 0

            return np.array([p0, p1, p2])

        def p_ad(m_gt, c_gt, ad0, ad1, contamination):
            c = binom(ad0+ad1, ad0)

            if m_gt == 0:
                if c_gt == 0:
                    return c*(1-self.eps)**ad0*self.eps**ad1

                elif c_gt == 1:
                    return c*((1+contamination)/2 - contamination*self.eps)**ad0*((1-contamination)/2+contamination*self.eps)**ad1

                elif c_gt == 2:
                    return c*(contamination+(1-2*contamination)*self.eps)**ad0*(1-contamination-(1-2*contamination)*self.eps)**ad1

            elif m_gt == 1:
                if c_gt == 0:
                    return c*(1-contamination/2-(1-contamination)*self.eps)**ad0*(contamination/2+(1-contamination)*self.eps)**ad1
                if c_gt == 1:
                    return c*(1/2)**(ad0+ad1)
                if c_gt == 2:
                    return c*(contamination/2+(1-contamination)*self.eps)**ad0*(1-contamination/2-(1-contamination)*self.eps)**ad1

            elif m_gt == 2:
                if c_gt == 0:
                    return c*(1-contamination-(1-2*contamination)*self.eps)**ad0*(contamination+(1-2*contamination)*self.eps)**ad1
                if c_gt == 1:
                    return c*((1-contamination)/2+contamination*self.eps)**ad0*((1+contamination)/2-contamination*self.eps)**ad1
                if c_gt == 2:
                    return c*self.eps**ad0*(1-self.eps)**ad1

        def p_joint(c_gt, m_gt, f_gt, ad0, ad1, m_priors, f_priors, contamination):
            return p_ad(m_gt, c_gt, ad0, ad1, contamination)*p_men(m_gt, f_gt)[c_gt]*m_priors[m_gt]*f_priors[f_gt]

        def p(ad0, ad1, contamination, m_priors, f_priors):
            z = sum([p_joint(c_gt, m_gt, f_gt, ad0, ad1, m_priors, f_priors, contamination) for c_gt in gts for m_gt in gts for f_gt in gts])
            ps = [sum([p_joint(c_gt, m_gt, f_gt, ad0, ad1,  m_priors, f_priors, contamination)/z for m_gt in gts for f_gt in gts]) for c_gt in gts]
            
            return ps


        contaminations = X[:, -1].astype(np.float64)
        ad0 = X[:, self.idx[0]].astype(np.int64)
        ad1 = X[:, self.idx[1]].astype(np.int64)
        m_pls = np.power(10, X[:,13:16])-1
        f_pls = np.power(10, X[:,16:19])-1

        m_priors = 10**(-m_pls/10)
        m_priors /= np.sum(m_priors, axis=1, keepdims=True) # Normalize

        f_priors = 10**(-f_pls/10)
        f_priors /= np.sum(f_priors, axis=1, keepdims=True)

        ps = np.empty((X.shape[0], 3))

        for i in tqdm(range(X.shape[0])):
            ps[i] = p(ad0[i], ad1[i], contaminations[i], m_priors[i], f_priors[i])

        preds = np.argmax(ps, axis=1)

        return preds

    def predict_proba(self, X, y=None):
        preds = self.predict(X)
        probs = np.zeros((preds.shape[0], 3))
        probs[np.arange(preds.shape[0]), preds] = 1

        return probs


    def score(self, X, y=None):
        return(np.sum(self.predict(X) == y)) 