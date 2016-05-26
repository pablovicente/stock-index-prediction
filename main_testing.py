#!/usr/bin/python

"""
Performs different types of testing on the time series.

	    Training Period  |   Testing Period
	 ____________________________________________________
1 -   from 1993 to 2010  |   from 2011 to 2016
2 -  three years periods |   one year periods
3 -     certain periods  |   10 days aftes those periods
4 -     certain periods  |   10 days aftes those periods including data in the next period


"""

colsToRemove = ['Date', 'INDEX_IBEX']
colY = 'INDEX_IBEX'

####################
#      Type 1      #
####################


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

## SVM
scv = svm.SVC(kernel='rbf')
scv.fit(trainX, trainY)
print scv.score(testX, testY)
print classification_report(testY, scv.predict(testX))
print roc_auc_score(testY, scv.predict(testX))

####################
#      Type 2      #
####################


iterations = []

iteration_train_dates1 = Iteration.Iteration('1993-07-07', '1997-07-07')
iterations.append(iteration_train_dates1)
iteration_test_dates1  = Iteration.Iteration('1997-07-08', '1998-07-07')
iterations.append(iteration_test_dates1)
iteration_train_dates2 = Iteration.Iteration('1998-07-08', '2002-07-08')
iterations.append(iteration_train_dates2)
iteration_test_dates2  = Iteration.Iteration('2002-07-05', '2003-07-07')
iterations.append(iteration_test_dates2)
iteration_train_dates3 = Iteration.Iteration('2003-07-08', '2007-07-06')
iterations.append(iteration_train_dates3)
iteration_test_dates3  = Iteration.Iteration('2007-07-09', '2008-07-07')
iterations.append(iteration_test_dates3)
iteration_train_dates4 = Iteration.Iteration('2008-07-08', '2012-07-09')
iterations.append(iteration_train_dates4)
iteration_test_dates4  = Iteration.Iteration('2012-07-06', '2013-07-05')
iterations.append(iteration_test_dates4)
iteration_train_dates5 = Iteration.Iteration('2013-07-05', '2015-07-08')
iterations.append(iteration_train_dates5)
iteration_test_dates5  = Iteration.Iteration('2015-07-08', '2016-04-20')
iterations.append(iteration_test_dates5)

iteration_train_dates1.calculate_indices(dataset)
iteration_test_dates1.calculate_indices(dataset)
iteration_train_dates2.calculate_indices(dataset)
iteration_test_dates2.calculate_indices(dataset)
iteration_train_dates3.calculate_indices(dataset)
iteration_test_dates3.calculate_indices(dataset)
iteration_train_dates4.calculate_indices(dataset)
iteration_test_dates4.calculate_indices(dataset)
iteration_train_dates5.calculate_indices(dataset)
iteration_test_dates5.calculate_indices(dataset)


colsToRemove = ['Date', 'INDEX_IBEX']
colY = 'INDEX_IBEX'

for i in range(0, len(iterations), 2):
    print("===========================")    
    print("Iteration %s" % (i))
    print("Training: from %s to %s" % (iterations[i].startDate, iterations[i].endDate))
    print("Testing: from %s to %s" % (iterations[i+1].startDate, iterations[i+1].endDate))    
    trainDates = []
    testDates = []
    trainDates.append(iterations[i].lowerIndex)
    trainDates.append(iterations[i].upperIndex)
    testDates.append(iterations[i+1].lowerIndex)
    testDates.append(iterations[i+1].upperIndex)
    
    trainX, trainY, testX, testY, cols, new_dataset = dataset_to_train_using_dates(dataset, trainDates, testDates, binary=False, shiftFeatures=False, shiftTarget=False)
    print "%s %s %s %s" % (trainX.shape,trainY.shape,testX.shape,testY.shape)
    
    ## SVM
    scv = svm.SVC(kernel='rbf')
    scv.fit(trainX, trainY)
    
    print scv.score(testX, testY)
    print classification_report(testY, scv.predict(testX))
    print roc_auc_score(testY, scv.predict(testX))



####################
#      Type 3      #
####################

colsToRemove = ['Date', 'INDEX_IBEX']
colY = 'INDEX_IBEX'

train_index = []
test_index = []
for i in range(dataset.shape[0]):
    if i % 30 == 0 and i > 0:
        test_index.append(i)
    else:
        train_index.append(i)

train_df = dataset.ix[train_index]        
test_df = dataset.ix[test_index]        

trainX, trainY, testX, testY = ml_dataset.dataset_to_train(train_df, test_df, binary=True)

## SVM
scv = svm.SVC(kernel='rbf')
scv.fit(trainX, trainY)
print scv.score(testX, testY)
print classification_report(testY, scv.predict(testX))
print roc_auc_score(testY, scv.predict(testX))


####################
#      Type 4      #
####################

colsToRemove = ['Date', 'INDEX_IBEX']
colY = 'INDEX_IBEX'

rows = dataset.shape[0]
training_samples = 200
test_samples = 10
min_train = 0
i = 0

while i < rows:
    print("===========================")    
    print("Iteration %s" % (i))
    

    if((i + training_samples + test_samples) < rows): 
        max_train = i + training_samples
        min_test = i + training_samples + 1
        max_test = i + training_samples +test_samples + 1
    else:
        break
    
    train_indices = range(min_train,max_train,1)
    test_indices = range(min_test,max_test,1)

    print("Training: from %s to %s" % (dataset['Date'][min_train], dataset['Date'][max_train]))
    print("Testing: from %s to %s" % (dataset['Date'][min_test], dataset['Date'][max_test]))    
    
    i = i + training_samples + test_samples + 2
    train_df = dataset.ix[train_indices]        
    test_df = dataset.ix[test_indices]        
    
    
    trainX, trainY, testX, testY = ml_dataset.dataset_to_train(train_df, test_df, binary=True)
   
    ## SVM
    scv = svm.SVC(kernel='rbf')
    scv.fit(trainX, trainY)
    
    print scv.score(testX, testY)
    print classification_report(testY, scv.predict(testX))
    print roc_auc_score(testY, scv.predict(testX))
