"""
Se parte de un conjunto de training y otro de testing.
El conjunto de training es subdividido X_train y X_cv
Se entrenan los clasificadores sobre cada una de las partes y posteriormente 
se entrenan sobre todo el conjunto de testing
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn import (metrics, cross_validation, linear_model, preprocessing)

from utils import toString

SEED = 42  # always use a seed for randomized procedures

class Boosting(object):
    
    def __init__(self, models):        
        self.models = models
        
    def fit_predict(self, trainX, trainY, testX, testY):
        """

        """

        X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(trainX, trainY, test_size=0.5, random_state=SEED)

        predict = []
        # === Combine Models === #
        # Do a linear combination using a cross_validated data split
        for model in self.models:
            model.fit(X_cv, y_cv) 
            preds_model = model.predict_proba(X_train)[:, 1]            
            predict.append(preds_model)

            model_auc = compute_auc(y_train, preds_model)
            print "> AUC: %.4f [%s]" % (model_auc, toString(model))


        preds = np.hstack(tuple(predict)).reshape(len(predict),len(predict[-1])).transpose()
        preds[preds>0.9999999]=0.9999999
        preds[preds<0.0000001]=0.0000001
        preds = -np.log((1-preds)/preds)
        modelEN1 = linear_model.LogisticRegression()
        modelEN1.fit(preds, y_train)
        print "modelEN1.coef %s" % (modelEN1.coef_)

        predict = []
        for model in self.models:
            model.fit(X_train, y_train) 
            preds_model = model.predict_proba(X_cv)[:, 1]
            predict.append(preds_model)  

            model_auc = compute_auc(y_cv, preds_model)
            print "> AUC: %.4f [%s]" % (model_auc, toString(model))

            
        preds = np.hstack(tuple(predict)).reshape(len(predict),len(predict[-1])).transpose()
        preds[preds>0.9999999]=0.9999999
        preds[preds<0.0000001]=0.0000001
        preds = -np.log((1-preds)/preds)
        modelEN2 = linear_model.LogisticRegression()
        modelEN2.fit(preds, y_cv)
        print "modelEN2.coef %s" % (modelEN2.coef_)

        model_coefs = []
        for index in range(len(modelEN1.coef_[0])):
            model_coefs.append(modelEN1.coef_[0][index] + modelEN2.coef_[0][index])
            
    
    

        # === Predictions === #
        # When making predictions, retrain the model on the whole training set
        predict = []
        index = 0
        final_preds = np.zeros((testX.shape[0], ))
        
        
        for model in self.models:
            model.fit(trainX, trainY)
            preds_model = model.predict_proba(testX)[:, 1]
            preds_model[preds_model>0.9999999]=0.9999999
            preds_model[preds_model<0.0000001]=0.0000001
            preds_model = -np.log((1-preds_model)/preds_model)
            predict.append(preds_model)

            temp = model_coefs[index] * preds_model
            final_preds = final_preds + model_coefs[index] * preds_model

            index = index + 1

        mean_auc = compute_auc(testY, final_preds)

        print "> AUC: %.4f " % (mean_auc)

def compute_auc(y, y_pred):
        fpr, tpr, _ = metrics.roc_curve(y, y_pred)
        return metrics.auc(fpr, tpr)