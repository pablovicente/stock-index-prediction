#!/usr/bin/python

class Iteration:
    
    def __init__(self, startDate, endDate):
        self.startDate = startDate
        self.endDate = endDate
        self.lowerIndex = 0
        self.upperIndex = 0
    
    def calculate_indices(self, dataset):
        self.lowerIndex = dataset.Date[dataset.Date == self.startDate].index[0]
        self.upperIndex = dataset.Date[dataset.Date == self.endDate].index[0]