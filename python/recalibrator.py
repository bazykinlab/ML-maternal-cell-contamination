from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split

class Recalibrator:
    def __init__(self):
        self.model_lr = LogisticRegression(random_state=0)
        self.model_xgb = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=1000, n_jobs=-1, subsample=0.8, colsample_bytree=1)        
        
    def train(self, X_train, y_train):
        print("Training logistic regression")
        self.model_lr.fit(X_train, y_train)
        
        X_train_red, X_val, y_train_red, y_val = train_test_split(X_train, y_train, shuffle=True, random_state=0, train_size=0.8) 
        print("Training XGB")
        self.model_xgb.fit(X_train_red, y_train_red, eval_set=[(X_val, y_val)], early_stopping_rounds=20)