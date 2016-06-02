#!/usr/bin/python

"""
Performs different types of testing on the time series.

	    Training Period  |   Testing Period
	 ____________________________________________________
1 -   from 1993 to 2010  |   from 2011 to 2016
2 -  three years periods |   one year periods
3 -  three years periods |   one year periods
4 -  three years periods |   one year periods
5 -  three years periods |   one year periods
6 -  three years periods |   one year periods
7 -  three years periods |   one year periodsx

"""

colsToRemove = ['Date', 'INDEX_IBEX']
colY = 'INDEX_IBEX'

####################################################################################################
#                                           Type 1                                                 #
####################################################################################################


dates = [('1993-08-19','2011-07-08','2011-07-11','2016-04-20'), ('1993-08-19','2012-07-06','2012-07-09', '2016-04-20'), ('1993-08-19','2013-07-08','2013-07-09','2016-04-20')]
for date in dates:

    training_dates = Iteration.Iteration(date[0], date[1])
    testing_dates  = Iteration.Iteration(date[2], date[3])
    training_dates.calculate_indices(dataset)
    testing_dates.calculate_indices(dataset)

    trainDates = []
    testDates = []
    trainDates.append(training_dates.lowerIndex)
    trainDates.append(training_dates.upperIndex)
    testDates.append(testing_dates.lowerIndex)
    testDates.append(testing_dates.upperIndex)

    total = (trainDates[1]-trainDates[0]) + (testDates[1]-testDates[0])
    tr = float(trainDates[1]-trainDates[0]) / total * 100.0
    te = float(testDates[1]-testDates[0]) / total * 100.0

    print "==========================="    
    print "Training: from %s to %s" % (training_dates.startDate, training_dates.endDate)
    print "Testing: from %s to %s" % (testing_dates.startDate, testing_dates.endDate)
    print "%.3f %% training %.3f %% testing" % (tr, te)
    print "%d training %d testing" % (trainDates[1]-trainDates[0], testDates[1]-testDates[0])

    trainX, trainY, testX, testY, cols = ml_dataset.dataset_to_train_using_dates(dataset, trainDates, testDates, binary=False, shiftFeatures=False, shiftTarget=False)
    print "%s %s %s %s" % (trainX.shape,trainY.shape,testX.shape,testY.shape)

    ## SVM
    scv = svm.SVC(kernel='rbf')
    scv.fit(trainX, trainY)
    print scv.score(testX, testY)
    print metrics.classification_report(testY, scv.predict(testX))
    print metrics.roc_auc_score(testY, scv.predict(testX))



####################################################################################################
#                                           Type 2                                                 #
####################################################################################################

dates = [('1993-08-19', '2000-08-18', '2000-08-21', '2001-08-20'), ('1995-08-18', '2002-08-19','2002-08-20', '2003-08-20'),
         ('1997-08-19', '2004-08-19',' 2004-08-20', '2005-08-19'), ('1999-08-19', '2006-08-18','2006-08-21', '2007-08-20'),
         ('2001-08-17', '2008-08-19', '2008-08-20', '2009-08-20'), ('2003-08-19', '2010-08-19','2010-08-19', '2011-08-19'),
         ('2005-08-19', '2012-08-17', '2012-08-20', '2013-08-20'), ('2007-08-17', '2015-05-19','2015-05-19', '2016-04-20')]

iterations = []

iteration_train_dates1 = Iteration.Iteration('1993-08-19', '2000-08-18')
iterations.append(iteration_train_dates1)
iteration_test_dates1  = Iteration.Iteration('2000-08-21', '2001-08-20')
iterations.append(iteration_test_dates1)
iteration_train_dates2 = Iteration.Iteration('1995-08-18', '2002-08-19')
iterations.append(iteration_train_dates2)
iteration_test_dates2  = Iteration.Iteration('2002-08-20', '2003-08-20')
iterations.append(iteration_test_dates2)
iteration_train_dates3 = Iteration.Iteration('1997-08-19', '2004-08-19')
iterations.append(iteration_train_dates3)
iteration_test_dates3  = Iteration.Iteration('2004-08-20', '2005-08-19')
iterations.append(iteration_test_dates3)
iteration_train_dates4 = Iteration.Iteration('1999-08-19', '2006-08-18')
iterations.append(iteration_train_dates4)
iteration_test_dates4  = Iteration.Iteration('2006-08-21', '2007-08-20')
iterations.append(iteration_test_dates4)
iteration_train_dates5 = Iteration.Iteration('2001-08-17', '2008-08-19')
iterations.append(iteration_train_dates5)
iteration_test_dates5  = Iteration.Iteration('2008-08-20', '2009-08-20')
iterations.append(iteration_test_dates5)
iteration_train_dates6 = Iteration.Iteration('2003-08-19', '2010-08-19')
iterations.append(iteration_train_dates6)
iteration_test_dates6  = Iteration.Iteration('2010-08-19', '2011-08-19')
iterations.append(iteration_test_dates6)
iteration_train_dates7 = Iteration.Iteration('2005-08-19', '2012-08-17')
iterations.append(iteration_train_dates7)
iteration_test_dates7  = Iteration.Iteration('2012-08-20', '2013-08-20')
iterations.append(iteration_test_dates7)
iteration_train_dates8 = Iteration.Iteration('2007-08-17', '2015-05-19')
iterations.append(iteration_train_dates8)
iteration_test_dates8  = Iteration.Iteration('2015-05-19', '2016-04-20')
iterations.append(iteration_test_dates8)

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
iteration_train_dates6.calculate_indices(dataset)
iteration_test_dates6.calculate_indices(dataset)
iteration_train_dates7.calculate_indices(dataset)
iteration_test_dates7.calculate_indices(dataset)
iteration_train_dates8.calculate_indices(dataset)
iteration_test_dates8.calculate_indices(dataset)

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
    
    trainX, trainY, testX, testY, cols = ml_dataset.dataset_to_train_using_dates(dataset, trainDates, testDates, binary=False, shiftFeatures=False, shiftTarget=False)
    print "%s %s %s %s" % (trainX.shape,trainY.shape,testX.shape,testY.shape)
    
    ## SVM
    scv = svm.SVC(kernel='rbf')
    scv.fit(trainX, trainY)
    
    print scv.score(testX, testY)
    print metrics.classification_report(testY, scv.predict(testX))
    print metrics.roc_auc_score(testY, scv.predict(testX))



####################################################################################################
#                                           Type 3                                                 #
####################################################################################################

dates = [('1998-08-19', '2000-08-18', '2000-08-21', '2001-08-20'), ('2000-08-18', '2002-08-19', '2002-08-20', '2003-08-20'),
         ('2002-08-19', '2004-08-19', '2004-08-20', '2005-08-19'), ('2004-08-19', '2006-08-18', '2006-08-21', '2007-08-20'),
         ('2006-08-17', '2008-08-19', '2008-08-20', '2009-08-20'), ('2008-08-19', '2010-08-19', '2010-08-19', '2011-08-19'),
         ('2010-08-19', '2012-08-17', '2012-08-20', '2013-08-20'), ('2012-08-17', '2015-05-19', '2015-05-19', '2016-04-20')]

iterations = []

iteration_train_dates1 = Iteration.Iteration('1998-08-19', '2000-08-18')
iterations.append(iteration_train_dates1)
iteration_test_dates1  = Iteration.Iteration('2000-08-21', '2001-08-20')
iterations.append(iteration_test_dates1)
iteration_train_dates2 = Iteration.Iteration('2000-08-18', '2002-08-19')
iterations.append(iteration_train_dates2)
iteration_test_dates2  = Iteration.Iteration('2002-08-20', '2003-08-20')
iterations.append(iteration_test_dates2)
iteration_train_dates3 = Iteration.Iteration('2002-08-19', '2004-08-19')
iterations.append(iteration_train_dates3)
iteration_test_dates3  = Iteration.Iteration('2004-08-20', '2005-08-19')
iterations.append(iteration_test_dates3)
iteration_train_dates4 = Iteration.Iteration('2004-08-19', '2006-08-18')
iterations.append(iteration_train_dates4)
iteration_test_dates4  = Iteration.Iteration('2006-08-21', '2007-08-20')
iterations.append(iteration_test_dates4)
iteration_train_dates5 = Iteration.Iteration('2006-08-17', '2008-08-19')
iterations.append(iteration_train_dates5)
iteration_test_dates5  = Iteration.Iteration('2008-08-20', '2009-08-20')
iterations.append(iteration_test_dates5)
iteration_train_dates6 = Iteration.Iteration('2008-08-19', '2010-08-19')
iterations.append(iteration_train_dates6)
iteration_test_dates6  = Iteration.Iteration('2010-08-19', '2011-08-19')
iterations.append(iteration_test_dates6)
iteration_train_dates7 = Iteration.Iteration('2010-08-19', '2012-08-17')
iterations.append(iteration_train_dates7)
iteration_test_dates7  = Iteration.Iteration('2012-08-20', '2013-08-20')
iterations.append(iteration_test_dates7)
iteration_train_dates8 = Iteration.Iteration('2012-08-17', '2015-05-19')
iterations.append(iteration_train_dates8)
iteration_test_dates8  = Iteration.Iteration('2015-05-19', '2016-04-20')
iterations.append(iteration_test_dates8)

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
iteration_train_dates6.calculate_indices(dataset)
iteration_test_dates6.calculate_indices(dataset)
iteration_train_dates7.calculate_indices(dataset)
iteration_test_dates7.calculate_indices(dataset)
iteration_train_dates8.calculate_indices(dataset)
iteration_test_dates8.calculate_indices(dataset)
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
    
    trainX, trainY, testX, testY, cols = ml_dataset.dataset_to_train_using_dates(dataset, trainDates, testDates, binary=False, shiftFeatures=False, shiftTarget=False)
    print "%s %s %s %s" % (trainX.shape,trainY.shape,testX.shape,testY.shape)
    
    ## SVM
    scv = svm.SVC(kernel='rbf')
    scv.fit(trainX, trainY)
    
    print scv.score(testX, testY)
    print metrics.classification_report(testY, scv.predict(testX))
    print metrics.roc_auc_score(testY, scv.predict(testX))



####################################################################################################
#                                           Type 4                                                 #
####################################################################################################
dates = [('1993-08-19', '2000-08-18', '2000-08-21', '2000-09-21'), ('1995-08-18', '2002-08-19', '2002-08-20', '2002-09-20'),
         ('1997-08-19', '2004-08-19', '2002-08-20', '2002-09-20'), ('1999-08-19', '2006-08-18', '2002-08-21', '2002-09-20'),
         ('2001-08-17', '2008-08-19', '2008-08-20', '2008-09-22'), ('2003-08-19', '2010-08-19', '2010-08-19', '2010-09-20'),
         ('2005-08-19', '2012-08-20', '2012-08-20', '2012-09-20'), ('2007-08-20', '2015-09-21', '2015-08-19', '2015-09-21')]


iterations = []

iteration_train_dates1 = Iteration.Iteration('1993-08-19', '2000-08-18')
iterations.append(iteration_train_dates1)
iteration_test_dates1  = Iteration.Iteration('2000-08-21', '2000-09-21')
iterations.append(iteration_test_dates1)
iteration_train_dates2 = Iteration.Iteration('1995-08-18', '2002-08-19')
iterations.append(iteration_train_dates2)
iteration_test_dates2  = Iteration.Iteration('2002-08-20', '2002-09-20')
iterations.append(iteration_test_dates2)
iteration_train_dates3 = Iteration.Iteration('1997-08-19', '2004-08-19')
iterations.append(iteration_train_dates3)
iteration_test_dates3  = Iteration.Iteration('2002-08-20', '2002-09-20')
iterations.append(iteration_test_dates3)
iteration_train_dates4 = Iteration.Iteration('1999-08-19', '2006-08-18')
iterations.append(iteration_train_dates4)
iteration_test_dates4  = Iteration.Iteration('2002-08-21', '2002-09-20')
iterations.append(iteration_test_dates4)
iteration_train_dates5 = Iteration.Iteration('2001-08-17', '2008-08-19')
iterations.append(iteration_train_dates5)
iteration_test_dates5  = Iteration.Iteration('2008-08-20', '2008-09-22')
iterations.append(iteration_test_dates5)
iteration_train_dates6 = Iteration.Iteration('2003-08-19', '2010-08-19')
iterations.append(iteration_train_dates6)
iteration_test_dates6  = Iteration.Iteration('2010-08-19', '2010-09-20')
iterations.append(iteration_test_dates6)
iteration_train_dates7 = Iteration.Iteration('2005-08-19', '2012-08-20')
iterations.append(iteration_train_dates7)
iteration_test_dates7  = Iteration.Iteration('2012-08-20', '2012-09-20')
iterations.append(iteration_test_dates7)
iteration_train_dates8 = Iteration.Iteration('2007-08-20', '2015-09-21')
iterations.append(iteration_train_dates8)
iteration_test_dates8  = Iteration.Iteration('2015-08-19', '2015-09-21')
iterations.append(iteration_test_dates8)

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
iteration_train_dates6.calculate_indices(dataset)
iteration_test_dates6.calculate_indices(dataset)
iteration_train_dates7.calculate_indices(dataset)
iteration_test_dates7.calculate_indices(dataset)
iteration_train_dates8.calculate_indices(dataset)
iteration_test_dates8.calculate_indices(dataset)

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
    
    trainX, trainY, testX, testY, cols = ml_dataset.dataset_to_train_using_dates(dataset, trainDates, testDates, binary=False, shiftFeatures=False, shiftTarget=False)
    print "%s %s %s %s" % (trainX.shape,trainY.shape,testX.shape,testY.shape)
    
    ## SVM
    scv = svm.SVC(kernel='rbf')
    scv.fit(trainX, trainY)
    
    print scv.score(testX, testY)
    print metrics.classification_report(testY, scv.predict(testX))
    print metrics.roc_auc_score(testY, scv.predict(testX))



####################################################################################################
#                                           Type 5                                                 #
####################################################################################################


dates = [('1998-08-19', '2000-08-18', '2000-08-21', '2000-09-20'), ('2000-08-18', '2002-08-19', '2002-08-20', '2002-09-20'),
         ('2002-08-19', '2004-08-19', '2004-08-20', '2004-09-20'), ('2004-08-19', '2006-08-18', '2006-08-21', '2006-09-20'),
         ('2006-08-18', '2008-08-19', '2008-08-20', '2008-09-22'), ('2008-08-19', '2010-08-19', '2010-08-19', '2010-09-20'),
         ('2010-08-19', '2012-08-17', '2012-08-20', '2012-09-20'), ('2012-08-17', '2015-08-19', '2015-08-20', '2015-09-21')]


iterations = []

iteration_train_dates1 = Iteration.Iteration('1998-08-19', '2000-08-18')
iterations.append(iteration_train_dates1)
iteration_test_dates1  = Iteration.Iteration('2000-08-21', '2000-09-20')
iterations.append(iteration_test_dates1)
iteration_train_dates2 = Iteration.Iteration('2000-08-18', '2002-08-19')
iterations.append(iteration_train_dates2)
iteration_test_dates2  = Iteration.Iteration('2002-08-20', '2002-09-20')
iterations.append(iteration_test_dates2)
iteration_train_dates3 = Iteration.Iteration('2002-08-19', '2004-08-19')
iterations.append(iteration_train_dates3)
iteration_test_dates3  = Iteration.Iteration('2004-08-20', '2004-09-20')
iterations.append(iteration_test_dates3)
iteration_train_dates4 = Iteration.Iteration('2004-08-19', '2006-08-18')
iterations.append(iteration_train_dates4)
iteration_test_dates4  = Iteration.Iteration('2006-08-21', '2006-09-20')
iterations.append(iteration_test_dates4)
iteration_train_dates5 = Iteration.Iteration('2006-08-18', '2008-08-19')
iterations.append(iteration_train_dates5)
iteration_test_dates5  = Iteration.Iteration('2008-08-20', '2008-09-22')
iterations.append(iteration_test_dates5)
iteration_train_dates6 = Iteration.Iteration('2008-08-19', '2010-08-19')
iterations.append(iteration_train_dates6)
iteration_test_dates6  = Iteration.Iteration('2010-08-19', '2010-09-20')
iterations.append(iteration_test_dates6)
iteration_train_dates7 = Iteration.Iteration('2010-08-19', '2012-08-17')
iterations.append(iteration_train_dates7)
iteration_test_dates7  = Iteration.Iteration('2012-08-20', '2012-09-20')
iterations.append(iteration_test_dates7)
iteration_train_dates8 = Iteration.Iteration('2012-08-17', '2015-08-19')
iterations.append(iteration_train_dates8)
iteration_test_dates8  = Iteration.Iteration('2015-08-20', '2015-09-21')
iterations.append(iteration_test_dates8)

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
iteration_train_dates6.calculate_indices(dataset)
iteration_test_dates6.calculate_indices(dataset)
iteration_train_dates7.calculate_indices(dataset)
iteration_test_dates7.calculate_indices(dataset)
iteration_train_dates8.calculate_indices(dataset)
iteration_test_dates8.calculate_indices(dataset)


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
    
    trainX, trainY, testX, testY, cols = ml_dataset.dataset_to_train_using_dates(dataset, trainDates, testDates, binary=False, shiftFeatures=False, shiftTarget=False)
    print "%s %s %s %s" % (trainX.shape,trainY.shape,testX.shape,testY.shape)
    
    ## SVM
    scv = svm.SVC(kernel='rbf')
    scv.fit(trainX, trainY)
    
    print scv.score(testX, testY)
    print metrics.classification_report(testY, scv.predict(testX))
    print metrics.roc_auc_score(testY, scv.predict(testX))

####################################################################################################
#                                           Type 6                                                 #
####################################################################################################

trainingPeriodList = [365, 365*2,365*3]
testingPeriodList = [1,5,30,40, 50, 365]
stepList = [100, 200, 300, 400]

for trainingPeriod in trainingPeriodList:
    for testingPeriod in testingPeriodList:
        for step in stepList:

            #print "#######################################"
            #print "\t Training period %s" % (trainingPeriod)
            #print "\t Testing period %s" % (testingPeriod)
            #print "\t Step %s" % (step)
            #print "#######################################\n"

            init = trainingPeriod + testingPeriod

            for index in range(init,dataset.shape[0], step):


                trainDates = []
                testDates = []
                trainDates.append(index - trainingPeriod - testingPeriod)
                trainDates.append(index - testingPeriod)
                testDates.append(index -  testingPeriod + 1)
                testDates.append(index)

                #print "%s %s %s %s" % (trainDates[0], trainDates[1], testDates[0], testDates[1])
#
                #print "==========================="
                #print "Iteration %s" % (i)
                #print "Training: from %s to %s" % (dataset.Date[trainDates[0]], dataset.Date[trainDates[1]])
                #print "Testing: from %s to %s" % (dataset.Date[testDates[0]], dataset.Date[testDates[1]])

                trainX, trainY, testX, testY, cols = ml_dataset.dataset_to_train_using_dates(dataset, trainDates, testDates, binary=False, shiftFeatures=False, shiftTarget=False)
                #print "%s %s %s %s" % (trainX.shape,trainY.shape,testX.shape,testY.shape)

                ## SVM
                scv = svm.SVC(kernel='rbf')
                scv.fit(trainX, trainY)

                #print scv.score(testX, testY)
                #print metrics.classification_report(testY, scv.predict(testX))
                #if testingPeriod > 1:     print metrics.roc_auc_score(testY, scv.predict(testX))


####################################################################################################
#                                           Type 7                                                 #
####################################################################################################

trainingPeriodList = [365, 365*2,365*3]
testingPeriodList = [1,5,10,30,40,40]
stepList = [100, 200, 300, 400]

for trainingPeriod in trainingPeriodList:
    for testingPeriod in testingPeriodList:
        for step in stepList:

            print "#######################################"
            print "\t Training period %s" % (trainingPeriod)
            print "\t Testing period %s" % (testingPeriod)
            print "\t Step %s" % (step)
            print "#######################################\n"

            init = trainingPeriod + testingPeriod
            match = 0
            missmatch = 0
            
            for index in range(init,dataset.shape[0], step):


                trainDates = []
                testDates = []
                trainDates.append(index - trainingPeriod - testingPeriod)
                trainDates.append(index - testingPeriod)
                testDates.append(index -  testingPeriod + 1)
                testDates.append(index)

                #print "%s %s %s %s" % (trainDates[0], trainDates[1], testDates[0], testDates[1])

                #print "==========================="
                #print "Iteration %s" % (i)
                #print "Training: from %s to %s" % (dataset.Date[trainDates[0]], dataset.Date[trainDates[1]])
                #print "Testing: from %s to %s" % (dataset.Date[testDates[0]], dataset.Date[testDates[1]])

                trainX, trainY, testX, testY, cols = ml_dataset.dataset_to_train_using_dates(dataset, trainDates, testDates, binary=False, shiftFeatures=False, shiftTarget=False)
                #print "%s %s %s %s" % (trainX.shape,trainY.shape,testX.shape,testY.shape)

                ## SVM
                scv = svm.SVC(kernel='rbf')
                scv.fit(trainX, trainY)
                predictions = scv.predict(testX)
                
 
                
                if testY[testingPeriod-1] == predictions[testingPeriod-1]: 
                    match = match + 1
                else: 
                    missmatch = missmatch + 1  

                #print scv.score(testX, testY)
                #print metrics.classification_report(testY, scv.predict(testX))
                #if testingPeriod > 1:     print metrics.roc_auc_score(testY, scv.predict(testX))
            print "%.2f hit " % (float(match)/(match+missmatch))


