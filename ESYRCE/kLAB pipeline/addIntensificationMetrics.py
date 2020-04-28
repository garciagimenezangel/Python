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
inputESYRCE = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\session_esyrceFiltered_z30_epsg23030_selectedCols.pkl'
dill.load_session(inputESYRCE) # data in dataSel

dataSel['seminaturalPercentage'] = np.nan
dataSel['avCropfieldSize'] = np.nan
dataSel['heterogeneity'] = np.nan

# Loop plot numbers
blockNrs = np.unique(dataSel.D2_NUM)
totalNr = len(blockNrs) 
contNr = 0
for blockNr in blockNrs:
    
    bSelectedInd = dataSel.D2_NUM == blockNr
    dataBlockNr = [dataSel.iloc[i] for i in range(0,len(bSelectedInd)) if bSelectedInd.iloc[i]]
    dataBlockNr = gpd.GeoDataFrame(dataBlockNr)
    
    years = np.unique(dataBlockNr.YEA)
    for year in years:
        
        bSelectedInd  = dataBlockNr.YEA == year
        dataBlockYear = [dataBlockNr.iloc[i] for i in range(0,len(bSelectedInd)) if bSelectedInd.iloc[i]]    
        dataBlockYear = gpd.GeoDataFrame(dataBlockYear)
    
        # Calculate intensification parameters 
        intensParams    = bc.calculateIntensificationParameters(dataBlockYear)
        seminatuPerc    = intensParams['seminaturalPercentage']
        avCropfieldSize = intensParams['avCropfieldSize']
        heterogeneity   = intensParams['heterogeneity']
        
        # Assign values
        dataSel.seminaturalPercentage.iloc[dataBlockYear.index] = seminatuPerc
        dataSel.avCropfieldSize.iloc[dataBlockYear.index]       = avCropfieldSize
        dataSel.heterogeneity.iloc[dataBlockYear.index]         = heterogeneity
        
    contNr = contNr+1
    if np.mod(contNr, 10) == 0:
        times = contNr / totalNr 
        print("Processing data...", np.floor(times*100), "percent completed...")
    
    if np.mod(contNr, 3000) == 0:
        times = contNr / totalNr 
        dill.dump_session(home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\session_esyrceFiltered_z30_epsg23030_selectedCols_addIntenMetrics.pkl')
        print("Saved session... " + home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\session_esyrceFiltered_z30_epsg23030_selectedCols_addIntenMetrics.pkl')

dill.dump_session(home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\session_esyrceFiltered_z30_epsg23030_selectedCols_addIntenMetrics.pkl')
print("FINISHED... Saved session... " + home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\session_esyrceFiltered_z30_epsg23030_selectedCols_addIntenMetrics.pkl')


