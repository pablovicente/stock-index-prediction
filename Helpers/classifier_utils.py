import json
import scipy as sp
import numpy as np

from sklearn.metrics import roc_curve, auc
from sklearn.grid_search import GridSearchCV

from utils import toString

N_TREES = 500
SEED = 42

INITIAL_PARAMS = {
    'LogisticRegression': {'C': 2, 'penalty': 'l2', 'random_state': SEED},
    'RandomForestClassifier': {
        'n_estimators': N_TREES, 'n_jobs': 4,
        'min_samples_leaf': 2, 'bootstrap': False, 'random_state': SEED, 
        'max_depth': 30, 'min_samples_split': 5, 'max_features': .1
    },
    'ExtraTreesClassifier': {
        'n_estimators': N_TREES, 'n_jobs': 3, 'min_samples_leaf': 2,
        'max_depth': 30, 'min_samples_split': 5, 'max_features': .1,
        'bootstrap': False,
    },
    'GradientBoostingClassifier': {
        'n_estimators': N_TREES, 'learning_rate': .08, 'max_features': 7,
        'min_samples_leaf': 1, 'min_samples_split': 3, 'max_depth': 5,
        'random_state': SEED,
    },
    'SVC': {
        'probability': True,
    },
}

PARAM_GRID = {
    'LogisticRegression': {'C': [1.5, 2, 2.5, 3, 3.5, 5, 5.5]},
    'RandomForestClassifier': {
        'n_jobs': [1], 'max_depth': [15, 20, 25, 30, 35, None],
        'min_samples_split': [1, 3, 5, 7],
        'max_features': [3, 8, 11, 15],
    },
    'ExtraTreesClassifier': {'min_samples_leaf': [2, 3],
                             'n_jobs': [1],
                             'min_samples_split': [1, 2, 5],
                             'bootstrap': [False],
                             'max_depth': [15, 20, 25, 30],
                             'max_features': [1, 3, 5, 11]},
    'GradientBoostingClassifier': {'max_features': [4, 5, 6, 7],
                                   'learning_rate': [.05, .08, .1],
                                   'max_depth': [8, 10, 13]},
}


def compute_auc(y, y_pred):
        fpr, tpr, _ = roc_curve(y, y_pred)
        return auc(fpr, tpr)

def compute_subset_auc(indices, pred_set, y):
    subset = [vect for i, vect in enumerate(pred_set) if i in indices]
    mean_preds = sp.mean(subset, axis=0)
    mean_auc = compute_auc(y, mean_preds)

    return mean_auc, indices

def find_params(model, feature_set, trainX, trainY, grid_search=False):
    """
    Return parameter set for the model, either predefined
    or found through grid search.
    """
    model_name = model.__class__.__name__
    params = INITIAL_PARAMS.get(model_name, {})

    try:
        with open('saved_params.json') as f:
            saved_params = json.load(f)
    except IOError:
        saved_params = {}

    if (grid_search and model_name in PARAM_GRID and toString(
            model, feature_set) not in saved_params):

        clf = GridSearchCV(model, PARAM_GRID[model_name], cv=10, n_jobs=6,
                           scoring="roc_auc")
        
        clf.fit(trainX, trainY)
        print "found params (%s > %.4f): %s" % (toString(model, feature_set), clf.best_score_, clf.best_params_)
        params.update(clf.best_params_)
        saved_params[toString(model, feature_set)] = params
        with open('saved_params.json', 'w') as f:
            json.dump(saved_params, f, indent=4, separators=(',', ': '),
                      ensure_ascii=True, sort_keys=True)
    else:
        params.update(saved_params.get(toString(model, feature_set), {}))
        if grid_search:
            print "using params %s: %s" % (toString(model, feature_set), params)

    return params