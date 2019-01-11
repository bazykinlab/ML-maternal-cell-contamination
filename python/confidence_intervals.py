from copy import deepcopy
import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin

class ConfidenceIntervalClassifier(BaseEstimator, ClassifierMixin):  
    """An example of classifier"""

    def __init__(self, z=3):
        """
        Called when initializing the classifier
        """
        self.z = z


    def fit(self, X, y=None):
        """
        This should fit classifier. All the "work" should be done here.
        """

#         assert (type(self.z) == float), "intValue parameter must be integer"

        return self

    def predict(self, X, y=None):
        gts = np.argmax(X[:, -4:-1], axis=1)
        predictions = deepcopy(gts)
        idx_hetero = (gts == 1)
        contaminations = X[:, -1].astype(np.float64)
        dps = X[:, 6].astype(np.float64)
        
        ab_ad0 = X[:, 20].astype(np.float64)
        ab_ad1 = X[:, 21].astype(np.float64)
        mo_ad0 = X[:, 16].astype(np.float64)
        mo_ad1 = X[:, 17].astype(np.float64)
        
        lower_bound = contaminations - self.z*np.sqrt(contaminations*(1 - contaminations)/dps)
        upper_bound = contaminations + self.z*np.sqrt(contaminations*(1 - contaminations)/dps)

        idx_0 = idx_hetero & (ab_ad0/ab_ad1 > mo_ad0/mo_ad1)
        mo_share = (2*ab_ad1/dps)
        idx_0_confirmed = (mo_share > lower_bound) & (mo_share < upper_bound) & idx_0
        predictions[idx_0_confirmed] = 0

        idx_1 = idx_hetero & (ab_ad0/ab_ad1 < mo_ad0/mo_ad1)
        mo_share = (2*ab_ad0/dps)
        idx_1_confirmed = (mo_share > lower_bound) & (mo_share < upper_bound) & idx_1
        predictions[idx_1_confirmed] = 2

        return predictions

    def predict_proba(self, X, y=None):
        preds = self.predict(X)
        probs = np.zeros((preds.shape[0], 3))
        probs[np.arange(preds.shape[0]), preds] = 1

        return probs


    def score(self, X, y=None):
        return(np.sum(self.predict(X) == y)) 

def confidence_intervals(df_test, ab_name='abortus', mo_name="mother", z=3):
    predictions = deepcopy(df_test[ab_name + '^GT'])
    idx_hetero = (df_test[ab_name + '^GT'] == 1)
    contaminations = df_test['contamination'].values

    lower_bound = contaminations - z*np.sqrt(contaminations*(1 - contaminations)/df_test[ab_name + '^DP'].values)
    upper_bound = contaminations + z*np.sqrt(contaminations*(1 - contaminations)/df_test[ab_name + '^DP'].values)

    idx_0 = idx_hetero & (df_test[ab_name + '^AD0']/df_test[ab_name + '^AD1'] > df_test[mo_name + '^AD0']/df_test[mo_name + '^AD1'])
    mo_share = (2*df_test[ab_name + '^AD1']/df_test[ab_name + '^DP'])
    idx_0_confirmed = (mo_share > lower_bound) & (mo_share < upper_bound) & idx_0
    predictions.loc[idx_0_confirmed] = 0

    idx_1 = idx_hetero & (df_test[ab_name + '^AD0']/df_test[ab_name + '^AD1'] < df_test[mo_name + '^AD0']/df_test[mo_name + '^AD1'])
    mo_share = (2*df_test[ab_name + '^AD0']/df_test[ab_name + '^DP'])
    idx_1_confirmed = (mo_share > lower_bound) & (mo_share < upper_bound) & idx_1
    predictions.loc[idx_1_confirmed] = 2

    return predictions.values