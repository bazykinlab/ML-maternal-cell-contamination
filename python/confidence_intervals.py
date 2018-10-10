from copy import deepcopy
import numpy as np

def confidence_intervals(df_test, sample_name='abortus'):
    z = 3
    predictions = deepcopy(df_test[sample_name + '^GT'])
    idx_hetero = (df_test[sample_name + '^GT'] == 1)
    contaminations = df_test['contamination'].values

    lower_bound = contaminations - z*np.sqrt(contaminations*(1 - contaminations)/df_test[sample_name + '^DP'].values)
    upper_bound = contaminations + z*np.sqrt(contaminations*(1 - contaminations)/df_test[sample_name + '^DP'].values)

    idx_0 = idx_hetero & (df_test[sample_name + '^AD0']/df_test[sample_name + '^AD1'] > df_test['mother^AD0']/df_test['mother^AD1'])
    mo_share = (2*df_test[sample_name + '^AD1']/df_test[sample_name + '^DP'])
    idx_0_confirmed = (mo_share > lower_bound) & (mo_share < upper_bound) & idx_0
    predictions.loc[idx_0_confirmed] = 0

    idx_1 = idx_hetero & (df_test[sample_name + '^AD0']/df_test[sample_name + '^AD1'] < df_test['mother^AD0']/df_test['mother^AD1'])
    mo_share = (2*df_test[sample_name + '^AD0']/df_test[sample_name + '^DP'])
    idx_1_confirmed = (mo_share > lower_bound) & (mo_share < upper_bound) & idx_1
    predictions.loc[idx_1_confirmed] = 2

    return predictions.values