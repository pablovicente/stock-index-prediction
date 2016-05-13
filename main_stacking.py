N_TREES = 500
SEED = 42

selected_models = [
    "LRC:dataset",
    "LRC:dataset"
    #"LRC:dataset",
    #"LRC:dataset",
    #"RFC:dataset",
    #"RFC:dataset",
    #"RFC:dataset",
    #"RFC:dataset",
    #"RFC:dataset",
    #"GBC:dataset",
    #"GBC:dataset",
    #"LRC:dataset",
    #"GBC:dataset",
    #"GBC:dataset",
    #"RFC:effects_f",  # experimental; added after the competition
]

# Create the models on the fly
models = []
for item in selected_models:
    model_id, data_set = item.split(':')
    model = {'LRC': linear_model.LogisticRegression,
             'GBC': ensemble.GradientBoostingClassifier,
             'RFC': ensemble.RandomForestClassifier,
             'ETC': ensemble.ExtraTreesClassifier}[model_id]()
    model.set_params(random_state=SEED)
    models.append(model)

# Set params    
clf = Stacking.Stacking(models, stack=True, fwls=False,
                           model_selection=True)

colsToRemove = ['Date', 'INDEX_IBEX']
colY = 'INDEX_IBEX'

training_dates = Iteration.Iteration('1993-07-07', '2012-07-06')
testing_dates  = Iteration.Iteration('2012-07-09', '2016-04-20')
training_dates.calculate_indices(dataset)
testing_dates.calculate_indices(dataset)

trainDates = []
testDates = []
trainDates.append(training_dates.lowerIndex)
trainDates.append(training_dates.upperIndex)
testDates.append(testing_dates.lowerIndex)
testDates.append(testing_dates.upperIndex)
    
trainX, trainY, testX, testY = ml_dataset.dataset_to_train_using_dates(dataset, dataset_b, trainDates, testDates, colsToRemove, colY, True)


###  Metrics
print("computing cv score")
mean_auc = 0.0
for i in range(1):        
    cv_preds = clf.fit_predict(trainY, trainX, testX, testY, show_steps=True)

#    fpr, tpr, _ = metrics.roc_curve(y[cv], cv_preds)
#    roc_auc = metrics.auc(fpr, tpr)
#    logger.info("AUC (fold %d/%d): %.5f", i + 1, CONFIG.iter, roc_auc)
#    mean_auc += roc_auc

#    if CONFIG.diagnostics and i == 0:  # only plot for first fold
#        logger.info("plotting learning curve")
#        diagnostics.learning_curve(clf, y, train, cv)
#        diagnostics.plot_roc(fpr, tpr)
#if CONFIG.iter:
#    logger.info("Mean AUC: %.5f",  mean_auc/CONFIG.iter)
