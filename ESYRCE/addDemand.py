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

# Add demand average (weighted by area) by block 
dataSel['block_demand'] = np.nan
blockNrs = np.unique(dataSel.D2_NUM)
contNr = 0
totalNr = len(blockNrs) 
for blockNr in blockNrs:
    # The only safe and direct option is to do a selection of the block numer is doing like this: [dataBlockNr.iloc[i] for i in range(0,len(bSelectedInd)) if bSelectedInd.iloc[i]]   
    # However, such operation is very time consuming when searching in the whole dataset, so we make a first selection by using the interval [i0,iM]. 
    ii = np.where(dataSel.D2_NUM == blockNr)
    i0 = ii[0][0]
    iM = ii[0][len(ii[0])-1]
    dataBlockNr = dataSel[i0:(iM+1)]
    # Sanity check: in case the previous approach is not enough -> (iM-i0+1)!=len(ii[0]) -> further selection needed
    if (iM-i0+1)!=len(ii[0]): 
        bSelectedInd = dataBlockNr.D2_NUM == blockNr
        dataBlockNr  = [dataBlockNr.iloc[i] for i in range(0,len(bSelectedInd)) if bSelectedInd.iloc[i]]    
        dataBlockNr  = gpd.GeoDataFrame(dataBlockNr)       
        print("Warning... Further selection needed in Block nr:",blockNr)
    
    years = np.unique(dataBlockNr.YEA)
    for year in years: 
        bSelectedInd  = dataBlockNr.YEA == year
        dataBlockYear = [dataBlockNr.iloc[i] for i in range(0,len(bSelectedInd)) if bSelectedInd.iloc[i]]    
        dataBlockYear = gpd.GeoDataFrame(dataBlockYear)       
        dataSel.block_demand.iloc[dataBlockYear.index] = bc.calculateDemand(dataBlockYear)
        
    contNr = contNr+1
    if np.mod(contNr, 100) == 0:
        times = contNr / totalNr 
        print("addDemandBlockAvg...", np.floor(times*100), "percent completed...")
        
    if np.mod(contNr, 4000) == 0:
        times = contNr / totalNr 
        dill.dump_session(backupFile)
        print("Saved session... " + backupFile)
            
dill.dump_session(backupFile)
print("FINISHED... Saved session... " + backupFile)



    