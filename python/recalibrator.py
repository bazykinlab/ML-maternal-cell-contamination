from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class Recalibrator:
    def __init__(self):
        self.model_lr = LogisticRegression(random_state=0)
        self.model_xgb = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=1000, n_jobs=-1, subsample=0.8, colsample_bytree=1)        
        
    def train(self, X_train, y_train):
        self.model_lr.fit(X_train, y_train)
        self.model_xgb.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=20)