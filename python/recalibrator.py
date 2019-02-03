from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle

from confidence_intervals import ConfidenceIntervalClassifier

class Recalibrator:
    """ Class to train, load and apply recalibrator models

    Attributes:
        model_lr: Logistic Regression model (scikit-learn)
        model_xgb: XGBoost model
    """
    def __init__(self):
        """ Initialize classifiers
        """
        self.model_lr = LogisticRegression(random_state=0)
        self.model_xgb = XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=1000, n_jobs=-1, subsample=0.8, colsample_bytree=1) 
        self.model_ci = ConfidenceIntervalClassifier()
        self.model_meta = VotingClassifier([('ci', ConfidenceIntervalClassifier()), ('lr', LogisticRegression(random_state=0)), ('xgb', XGBClassifier(max_depth=6, learning_rate=0.1, n_estimators=1000, n_jobs=-1, subsample=0.8, colsample_bytree=1))],
                                           voting='soft')
        self.scaler = StandardScaler()

    def train(self, X_train, y_train):
        """ Strandardize training data and train models.

        Args:
            X_train (numpy.ndarray): independent variables of training data
            Y_train (numpy.ndarray): dependent variables of training data
        """
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        print("Training logistic regression")
        self.model_lr.fit(X_train_scaled, y_train)
        
        X_train_red, X_val, y_train_red, y_val = train_test_split(X_train_scaled, y_train, shuffle=True, random_state=0, train_size=0.8, test_size=0.2) 
        print("Training XGB")
        self.model_xgb.fit(X_train_red, y_train_red, eval_set=[(X_val, y_val)], early_stopping_rounds=20)

        print("Training meta-classifier")
        self.model_meta.fit(X_train, y_train)

    def save(self, filename):
        """ Serialize and save recalibrator object to a file

        Args:
            filename (str): name of file to save to
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, filename):
        """ Load recalibrator from a serialized file

        Args:
            filename (str): name of file to load from
        """
        with open(filename, 'rb') as f:
            loaded = pickle.load(f)
            self.__dict__.update(loaded.__dict__)

    def predict(self, X, model):
        """ Standardize input, and apply a model

        Args:
            X (numpy.ndarray): Input array

        Returns:
            Predicted genotypes (1d numpy array)
        """

        X_transformed = self.scaler.transform(X)
        return model.predict(X_transformed)

    def predict_lr(self, X):
        return self.predict(X, self.model_lr)

    def predict_xgb(self, X):
        return self.predict(X, self.model_xgb)

        