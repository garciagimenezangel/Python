# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24

@author: angel.gimenez

Calculate demand of ESYRCE polygons and add as a new column to the data.
The script also adds a measure of the demand by block (averaged over the area of the polygons)
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

backupFile = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\session_esyrceFiltered_z30_epsg23030_selectedCols_addIntenMetrics_addDemand.pkl'
stepsSave = 1000000
dataSel = bc.addDemand(dataSel, stepsSave, backupFile)
dataSel = bc.addDemandBlockAvg(dataSel, stepsSave/100, backupFile)

dataSel['block_demand'] = np.nan
blockNrs = np.unique(dataSel.D2_NUM)
contNr = 0
totalNr = len(blockNrs) 
for blockNr in blockNrs:
    selectedInd = dataSel.D2_NUM == blockNr
    block = [dataSel.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
    block = gpd.GeoDataFrame(block)
    blockDemand = calculateDemand(block)
    dataSel.at[selectedInd,'block_demand'] = blockDemand
        
    contNr = contNr+1
    if np.mod(contNr, 100) == 0:
        times = contNr / totalNr 
        print("addDemandBlockAvg...", np.floor(times*100), "percent completed...")
        
    if np.mod(contNr, 10000) == 0:
        times = contNr / totalNr 
        dill.dump_session(backupFile)
        print("Saved session... " + backupFile)
            

dill.dump_session(backupFile)
print("FINISHED... Saved session... " + backupFile)


