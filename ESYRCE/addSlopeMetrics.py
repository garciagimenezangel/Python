# -*- coding: utf-8 -*-
"""
Calculate the slope of the metrics calculated in 'addMetrics.py'

INPUT: csv file with ESYRCE identificator (segment number + year) and the metrics 
OUTPUT: csv file with ESYRCE identificator (segment number) and the slope of each metric 
"""

import pandas as pd
from datetime import datetime
from os.path import expanduser
home = expanduser("~")

# The functions used to calculate the metrics are stored in a different file, to make this script cleaner 
import sys
sys.path.append(home + '\\Documents\\REPOSITORIES\\Python\\ESYRCE\\')
import functions

# INPUT
metrics = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\z28\\metrics\\flag0.csv'
data = pd.read_csv(metrics)

# OUTPUT
outFilename = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\z28\\metrics\\slope_flag0.csv'   

# LOG
logFile = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\logs\\addSlopeMetrics.log'
buffSize = 1
log = open(logFile, "a", buffering=buffSize)
log.write("\n")
log.write("PROCESS addMetrics.py STARTED AT: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'\n')


# Agreggate segments (unique zone and number), using a custom function that calculates the slope of the regression line using 'YEA' 
# as the 'x axis' values and each metric as the 'y axis' values
grouped = data.groupby(['D1_HUS','D2_NUM'])
slopeMetrics = grouped.apply(functions.getEvolutionMetrics)
slopeMetrics.reset_index(inplace=True)

# Save as csv
log.write("Writing file..."+outFilename+'\n')
slopeMetrics.to_csv(outFilename, index=False)
log.write("FINISHED... Data saved... " + outFilename+'\n')
log.close()