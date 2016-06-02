#!/usr/bin/python


import Quandl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import sys
from os import listdir
from os.path import isfile, join

def main(argv):
	#if len(sys.argv) != 3:
	#	print "There must be two arguments"
	#	print "Arg 1: Base name of images"
	#	print "Arg 2: number of images"
	# ndaq_data = Quandl.get("WIKI/NDAQ", authtoken="yqJTTM9qspzh2WXXtJ8V")
	
	ndaq_data = pd.read_csv('/Users/Pablo/Desktop/TFM/Raw_Data/WIKI-NDAQ.csv')
	

	

	plt.figure()
	ndaq_data.plot(x='Open')

	print "##############"
	print ndaq_data.shape

if __name__ == "__main__":
   main(sys.argv[1:])	