N_TREES = 500
SEED = 42


selected_models = [
    "LRC:dataset",    
    "RFC:dataset",
    "GBC:dataset",
    #"RFC:effects_f",  # experimental; added after the competition
]


# Create the models on the fly
models = []
for item in selected_models:
    model_id, data_set = item.split(':')
    model = {'LRC': linear_model.LogisticRegression,
             'GBC': ensemble.GradientBoostingClassifier,
             'RFC': ensemble.RandomForestClassifier,
             'ETC': ensemble.ExtraTreesClassifier, 
             'SVM': svm.SVC}[model_id]()
    model.set_params(random_state=SEED)
    models.append(model)

training_dates = Iteration.Iteration('1993-08-19', '2012-07-06')
testing_dates  = Iteration.Iteration('2012-07-09', '2016-04-20')
training_dates.calculate_indices(dataset)
testing_dates.calculate_indices(dataset)

trainDates = []
testDates = []
trainDates.append(training_dates.lowerIndex)
trainDates.append(training_dates.upperIndex)
testDates.append(testing_dates.lowerIndex)
testDates.append(testing_dates.upperIndex)
    
trainX, trainY, testX, testY = ml_dataset.dataset_to_train_using_dates(dataset, trainDates, testDates, binary=True)

################
##  Stacking  ##
################
clf = Stacking.Stacking(models, stack=True, fwls=False,
                           model_selection=True)

###  Metrics
print("computing cv score")
mean_auc = 0.0
iter_ = 1
for i in range(iter_):        
    cv_preds = clf.fit_predict(trainY, trainX, testX, testY, show_steps=True)

    fpr, tpr, _ = metrics.roc_curve(testY, cv_preds)
    roc_auc = metrics.auc(fpr, tpr)
    print "AUC (fold %d/%d): %.5f" % (i + 1, iter_, roc_auc)
    mean_auc += roc_auc

    print "Mean AUC: %.5f" % (mean_auc/iter_)

################
##  Boosting  ##
################
    
boosting = Boosting.Boosting(models)

###  Metrics
mean_auc = 0.0
iter_ = 1
for i in range(iter_):        
    boosting.fit_predict(trainX, trainY, testX, testY)

    #fpr, tpr, _ = metrics.roc_curve(testY, cv_preds)
    #roc_auc = metrics.auc(fpr, tpr)
    #print "AUC (fold %d/%d): %.5f" % (i + 1, iter_, roc_auc)
    #mean_auc += roc_auc
#
    #print "Mean AUC: %.5f" % (mean_auc/CONFIG.iter)

##############################################
## PRICE FLOW BETWEEN TWO CONSECUTIVES DAYS ##
##############################################

slice = 30
pylab.figure(1)

ax = plt.subplot(111)

ax.plot(good[:slice], 'g', label='GOOD')
ax.plot(bad[:slice], 'r', label='BAD')
box = ax.get_position()
ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
plt.show()
