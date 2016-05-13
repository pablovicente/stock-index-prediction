import scipy as sp
import numpy as np
import multiprocessing
import itertools

from functools import partial
from operator import itemgetter

from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV
from sklearn import cross_validation, linear_model

from utils import toString

class Stacking(object):
    
    def __init__(self, models, generalizer=None, model_selection=True,
                 stack=False, fwls=False):        
        self.models = models
        self.model_selection = model_selection
        self.stack = stack
        self.fwls = fwls
        self.generalizer = linear_model.RidgeCV(alphas=np.linspace(0, 200), cv=100)    

    def fit_predict(self, y, train=None, predict=None, y_test=None, show_steps=True):
        
        stage0_train = []
        stage0_predict = []    

        y_train = y
        X_train = train
        X_predict = predict

        for model in self.models:

            model_preds = self._get_model_preds(model, X_train, X_predict, y_train)            
            stage0_predict.append(model_preds)

            # if stacking, compute cross-validated predictions on the train set
            if self.stack:
                model_cv_preds = self._get_model_cv_preds(model, X_train, y_train)
                stage0_train.append(model_cv_preds)

            # verbose mode: compute metrics after every model computation
            if show_steps:                
                    mean_preds, stack_preds, fwls_preds = self._combine_preds(
                        np.array(stage0_train).T, np.array(stage0_predict).T,
                        y_train, train, predict,
                        stack=self.stack, fwls=self.fwls)

                    model_auc = compute_auc(y_test, stage0_predict[-1])
                    mean_auc = compute_auc(y_test, mean_preds)
                    stack_auc = compute_auc(y_test, stack_preds) \
                        if self.stack else 0
                    fwls_auc = compute_auc(y_test, fwls_preds) \
                        if self.fwls else 0

                    print("> AUC: %.4f (%.4f, %.4f, %.4f) [%s]", model_auc,
                            mean_auc, stack_auc, fwls_auc,
                            toString(model))

        if self.model_selection and predict is not None:
            best_subset = self._find_best_subset(y_test, stage0_predict)
            stage0_train = [pred for i, pred in enumerate(stage0_train)
                            if i in best_subset]
            stage0_predict = [pred for i, pred in enumerate(stage0_predict)
                              if i in best_subset]

        mean_preds, stack_preds, fwls_preds = self._combine_preds(
            np.array(stage0_train).T, np.array(stage0_predict).T,
            y_train, stack=self.stack, fwls=self.fwls)

        if self.stack:
            selected_preds = stack_preds if not self.fwls else fwls_preds
        else:
            selected_preds = mean_preds

        return selected_preds

    def _get_model_preds(self, model, X_train, X_predict, y_train):
        """        
        
        """

        model.fit(X_train, y_train)
        model_preds = model.predict_proba(X_predict)[:, 1]
                       
        return model_preds

    def _get_model_cv_preds(self, model, X_train, y_train):
        """
        Return cross-validation predictions on the training set.       
        
        This is used if stacking is enabled (ie. a second model is used to
        combine the stage 0 predictions).
        """
        
        kfold = cross_validation.StratifiedKFold(y_train, 4)
        stack_preds = []
        indexes_cv = []
        for stage0, stack in kfold:
            model.fit(X_train[stage0], y_train[stage0])
            stack_preds.extend(list(model.predict_proba(X_train[stack])[:, 1]))
            indexes_cv.extend(list(stack))
        stack_preds = np.array(stack_preds)[sp.argsort(indexes_cv)]

        return stack_preds        

    def _combine_preds(self, X_train, X_cv, y, train=None, predict=None,
                       stack=False, fwls=False):
        """
        Combine preds, returning in order:
            - mean_preds: the simple average of all model predictions
            - stack_preds: the predictions of the stage 1 generalizer
            - fwls_preds: same as stack_preds, but optionally using more
                complex blending schemes (meta-features, different
                generalizers, etc.)
        """
        mean_preds = np.mean(X_cv, axis=1)
        stack_preds = None
        fwls_preds = None

        if stack:
            self.generalizer.fit(X_train, y)
            stack_preds = self.generalizer.predict(X_cv)

        #if self.fwls:
        #    meta, meta_cv = get_dataset('metafeatures', train, predict)
        #    fwls_train = np.hstack((X_train, meta))
        #    fwls_cv = np.hstack((X_cv, meta))
        #    self.generalizer.fit(fwls_train)
        #    fwls_preds = self.generalizer.predict(fwls_cv)

        return mean_preds, stack_preds, fwls_preds        

    def _find_best_subset(self, y, predictions_list):
        """
        Finds the combination of models that produce the best AUC.
        """

        best_subset_indices = range(len(predictions_list))

        pool = multiprocessing.Pool(processes=4)
        partial_compute_subset_auc = partial(compute_subset_auc,
                                             pred_set=predictions_list, y=y)
        best_auc = 0
        best_n = 0
        best_indices = []

        if len(predictions_list) == 1:
            return [1]

        for n in range(int(len(predictions_list)/2), len(predictions_list)):
            cb = itertools.combinations(range(len(predictions_list)), n)
            combination_results = pool.map(partial_compute_subset_auc, cb)
            best_subset_auc, best_subset_indices = max(
                combination_results, key=itemgetter(0))
            print "- best subset auc (%d models): %.4f > %s" % (
                n, best_subset_auc, list(best_subset_indices))
            if best_subset_auc > best_auc:
                best_auc = best_subset_auc
                best_n = n
                best_indices = list(best_subset_indices)
        pool.terminate()

        print "best auc: %.4f" % (best_auc)
        print "best n: %d" % (best_n)
        print "best indices: %s" % (best_indices)
        for i, (model) in enumerate(self.models):
            if i in best_subset_indices:
                print "> model: %s " % (model.__class__.__name__)

        return best_subset_indices
    
def compute_auc(y, y_pred):
        fpr, tpr, _ = roc_curve(y, y_pred)
        return auc(fpr, tpr)

def compute_subset_auc(indices, pred_set, y):
    subset = [vect for i, vect in enumerate(pred_set) if i in indices]
    mean_preds = sp.mean(subset, axis=0)
    mean_auc = compute_auc(y, mean_preds)

    return mean_auc, indices
