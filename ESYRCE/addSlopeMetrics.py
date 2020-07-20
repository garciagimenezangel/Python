# -*- coding: utf-8 -*-
"""
Calculate the slope of the metrics calculated in 'addMetrics.py'

INPUT: csv file with ESYRCE identificator (segment number + year) and the metrics 
OUTPUT: csv file with ESYRCE identificator (segment number) and the slope of each metric 
"""

import csv
import geopandas as gpd
import numpy as np
from os.path import expanduser
home = expanduser("~")

# The functions used to calculate the metrics are stored in a different file, to make this script cleaner 
import sys
sys.path.append(home + '\\Documents\\REPOSITORIES\\Python\\ESYRCE\\')
import functions

# INPUT
inputESYRCE = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\z30\\metrics\\flag2.csv'
data = gpd.read_file(inputESYRCE)

# OUTPUT
outFilename = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\z30\\metrics\\slope_flag2.csv'

def customFunction(a):
    a = np.asarray(a)
    for i in range(0, len(a)-1):
        b.append((a[i+1]-a[i]))
    return np.mean(b) 

# Agreggate segments (unique zone and number), using a custom function that calculates the slope of the regression line using 'YEA' 
# as the 'x axis' values and each metric as the 'y axis' values
data = data.groupby(['D1_HUS','D2_NUM']).agg(customFunction)
