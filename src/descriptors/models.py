# import data science
import numpy as np
import pandas as pd

# import random forest model
from sklearn.ensemble import RandomForestRegressor as RFR

# import linear models
from sklearn.linear_model import LinearRegression as LR
from sklearn.linear_model import Ridge 
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet

# import sklearn utils
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


class FeatureImportance(object):

    def __init__(self, X, y, objective=None, features_to_keep=5):

        self.X = X.to_numpy()
        self.features = list(X.columns)
        self.y = y.to_numpy()
        self.obj = objective
        self.features_to_keep = features_to_keep

    def LRModel(self):

        """
        train_X : (n_samples, n_features)
        train_Y : (n_samples, n_targets)
        """
        
        model = LR().fit(self.X_train, self.y_train)

    def RFModel(self, n_estimators=100, test=False):

        """                                                                                                                                   
        train_X : (n_samples, n_features)                                                                                                     
        train_Y : (n_samples, n_targets)
        """

        model = RFR(n_estimators=n_estimators)
        if isinstance(test, float):
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=test)    
        else:
            X_train = self.X
            y_train = self.y
            
        model.fit(X_train, y_train)
        train_prediction = model.predict(X_train)
        train_score = r2_score(y_train, train_prediction)

        if isinstance(test, float):
            test_prediction = model.predict(X_test)
            test_score = r2_score(y_test, test_prediction)
        
        imp_feats_indices = model.feature_importances_.argsort()
        sorted_feature_importances = model.feature_importances_[imp_feats_indices]
        imp_feats = [self.features[i] for i in imp_feats_indices[-self.features_to_keep:]]
        
        return imp_feats, train_score, model.feature_importances_