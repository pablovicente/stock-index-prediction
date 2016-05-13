"""
Se parte de un conjunto de training y otro de testing.
El conjunto de training es subdividio X_train y X_cv
Se entrenan los clasificadores sobre cada una de las partes y posteriormente 
se entrenan sobre todo el conjunto de testing
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier)
from sklearn import (metrics, cross_validation, linear_model, preprocessing)

SEED = 42  # always use a seed for randomized procedures

class Stacking(object):
    
    def __init__(self, models, generalizer=None, model_selection=True,
                 stack=False, fwls=False):        
        self.models = models
        self.model_selection = model_selection
        self.stack = stack
        self.fwls = fwls
        self.generalizer = linear_model.RidgeCV(alphas=np.linspace(0, 200), cv=100)

    def fit(self):
        modelRF.fit(X_train, y_train) 
        modelXT.fit(X_train, y_train) 
        modelGB.fit(X_train, y_train) 

    def predict(self): 
        predsRF = modelRF.predict_proba(X_cv)[:, 1]
        predsXT = modelXT.predict_proba(X_cv)[:, 1]
        predsGB = modelGB.predict_proba(X_cv)[:, 1]



# === Combine Models === #
# Do a linear combination using a cross_validated data split
X_train, X_cv, y_train, y_cv = cross_validation.train_test_split(X, y, test_size=0.5, random_state=SEED)

modelRF.fit(X_cv, y_cv) 
modelXT.fit(X_cv, y_cv) 
modelGB.fit(X_cv, y_cv) 
predsRF = modelRF.predict_proba(X_train)[:, 1]
predsXT = modelXT.predict_proba(X_train)[:, 1]
predsGB = modelGB.predict_proba(X_train)[:, 1]
preds = np.hstack((predsRF, predsXT, predsGB)).reshape(3,len(predsGB)).transpose()
preds[preds>0.9999999]=0.9999999
preds[preds<0.0000001]=0.0000001
preds = -np.log((1-preds)/preds)
modelEN1 = linear_model.LogisticRegression()
modelEN1.fit(preds, y_train)
print modelEN1.coef_

modelRF.fit(X_train, y_train) 
modelXT.fit(X_train, y_train) 
modelGB.fit(X_train, y_train) 
predsRF = modelRF.predict_proba(X_cv)[:, 1]
predsXT = modelXT.predict_proba(X_cv)[:, 1]
predsGB = modelGB.predict_proba(X_cv)[:, 1]
preds = np.hstack((predsRF, predsXT, predsGB)).reshape(3,len(predsGB)).transpose()
preds[preds>0.9999999]=0.9999999
preds[preds<0.0000001]=0.0000001
preds = -np.log((1-preds)/preds)
modelEN2 = linear_model.LogisticRegression()
modelEN2.fit(preds, y_cv)
print modelEN2.coef_

coefRF = modelEN1.coef_[0][0] + modelEN2.coef_[0][0]
coefXT = modelEN1.coef_[0][1] + modelEN2.coef_[0][1]
coefGB = modelEN1.coef_[0][2] + modelEN2.coef_[0][2]

# === Predictions === #
# When making predictions, retrain the model on the whole training set
modelRF.fit(X, y)
modelXT.fit(X, y)
modelGB.fit(X, y)

### Combine here
predsRF = modelRF.predict_proba(X_test)[:, 1]
predsXT = modelXT.predict_proba(X_test)[:, 1]
predsGB = modelGB.predict_proba(X_test)[:, 1]
predsRF[predsRF>0.9999999]=0.9999999
predsXT[predsXT>0.9999999]=0.9999999
predsGB[predsGB>0.9999999]=0.9999999
predsRF[predsRF<0.0000001]=0.0000001
predsXT[predsXT<0.0000001]=0.0000001
predsGB[predsGB<0.0000001]=0.0000001
predsRF = -np.log((1-predsRF)/predsRF)
predsXT = -np.log((1-predsXT)/predsXT)
predsGB = -np.log((1-predsGB)/predsGB)
preds = coefRF * predsRF + coefXT * predsXT + coefGB * predsGB

