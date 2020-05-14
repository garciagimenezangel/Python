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
    
#    bSelectedInd = dataSel.D2_NUM == blockNr
#    dataBlockNr = [dataSel.iloc[i] for i in range(0,len(bSelectedInd)) if bSelectedInd.iloc[i]]
#    dataBlockNr = gpd.GeoDataFrame(dataBlockNr)
    ii = np.where(dataSel.D2_NUM == blockNr)
    i0 = ii[0][0]
    iM = ii[0][len(ii[0])-1]
    dataBlockNr = dataSel[i0:(iM+1)]
    if (iM-i0+1)!=len(ii[0]): 
        print("Error... Exit loop in Block nr:",blockNr)
        break
    
    years = np.unique(dataBlockNr.YEA)
    for year in years:
        
#        bSelectedInd  = dataBlockNr.YEA == year
#        dataBlockYear = [dataBlockNr.iloc[i] for i in range(0,len(bSelectedInd)) if bSelectedInd.iloc[i]]    
#        dataBlockYear = gpd.GeoDataFrame(dataBlockYear)
        ii = np.where(dataBlockNr.YEA == year)
        i0 = ii[0][0]
        iM = ii[0][len(ii[0])-1]
        dataBlockYear = dataBlockNr[i0:(iM+1)]
        if (iM-i0+1)!=len(ii[0]): 
            print("Error... Exit loop in Block nr:",blockNr,"...Year:",year)
            break
    
        # Calculate intensification parameters 
        intensParams    = bc.calculateIntensificationParameters(dataBlockYear)
        seminatuPerc    = intensParams['seminaturalPercentage']
        avCropfieldSize = intensParams['avCropfieldSize']
        heterogeneity   = intensParams['heterogeneity']/bc.getBlockArea(dataBlockYear)
        
        # Assign values
        dataSel.seminaturalPercentage.iloc[dataBlockYear.index] = seminatuPerc
        dataSel.avCropfieldSize.iloc[dataBlockYear.index]       = avCropfieldSize
        dataSel.heterogeneity.iloc[dataBlockYear.index]         = heterogeneity
        
    contNr = contNr+1
    if np.mod(contNr, 100) == 0:
        times = contNr / totalNr 
        print("Processing data...", np.floor(times*100), "percent completed...")
    
    if np.mod(contNr, 3000) == 0:
        times = contNr / totalNr 
        dill.dump_session(home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\session_esyrceFiltered_z30_epsg23030_selectedCols_addIntenMetrics.pkl')
        print("Saved session... " + home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\session_esyrceFiltered_z30_epsg23030_selectedCols_addIntenMetrics.pkl')

dill.dump_session(home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\session_esyrceFiltered_z30_epsg23030_selectedCols_addIntenMetrics.pkl')
print("FINISHED... Saved session... " + home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\session_esyrceFiltered_z30_epsg23030_selectedCols_addIntenMetrics.pkl')


