# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24

@author: angel.gimenez

Calculate the the following variables using ESYRCE data
# INTENSIFICATION METRICS: 
# - Percentage of semi-natural cover
# - Average of cropfield size
# - Heterogeneity of crops
"""
import dill
import geopandas as gpd
import pandas as pd
import numpy as np
from os.path import expanduser
home = expanduser("~")

import sys
sys.path.append(home + '\\Documents\\REPOSITORIES\\Python\\ESYRCE\\lib\\')
import blockCalculator as bc 

# INPUT
inputESYRCE = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\session_esyrceFiltered_z30_epsg23030_selectedCols_addIntenMetrics.pkl'
dill.load_session(inputESYRCE) # data in dataSel
dataSel['demand'] = np.nan

backupFile = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\session_esyrceFiltered_z30_epsg23030_selectedCols_addIntenMetrics_addDemand.pkl'
stepsSave = 1000000
dataSel = bc.addDemand(dataSel, stepsSave, backupFile)

dill.dump_session(backupFile)
print("FINISHED... Saved session... " + backupFile)


