# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:56:21 2020

@author: angel.gimenez
"""

import dill
import csv
import numpy as np
import geopandas as gpd
from os.path import expanduser
home = expanduser("~")
pollReliance = home + '\\Google Drive\\PROJECTS\\OBSERV\\Lookup Tables\\Cultivar-Demand.csv'
cropNatArt   = home + '\\Google Drive\\PROJECTS\\OBSERV\\Lookup Tables\\CropNaturalArtificial.csv'
dictPollValues = { ''           :       0,
                   'no increase':       0, 
                   'increase':          0.5,
                   'increase-breeding': 0.5,
                   'increase-seed production': 0.5,
                   'little': 0.05,
                   'modest': 0.25,
                   'great': 0.65,
                   'essential': 0.95}

"""
INPUT: any subset of ESYRCE data
OUTPUT: dictionary with the values of intensification metrics
"""
def calculateIntensificationParameters(dfEsyrce):
    # Read LC codes
    with open(cropNatArt, mode='r') as infile:
        reader    = csv.reader(infile)
        dictCropNatArt = {rows[0]:rows[1] for rows in reader} # key: 'ESYRCE_code; value: 'Crop, Natural or Artificial'
    
    accAreaSeminatural = 0
    accAreaCropFields = 0
    avCropfieldSize = 0
    totalArea = 0
    crops = []
    for index in dfEsyrce.index:
        areaPolygon = dfEsyrce.loc[index].Shape_Area
        code = dfEsyrce.loc[index].D4_GRC       
        try:
            isSeminatural = dictCropNatArt[code] == 'Semi-natural'
            isCropfield   = dictCropNatArt[code] == 'Crop'
        except:
            isSeminatural = False
            isCropfield   = False       
        if isSeminatural: 
            accAreaSeminatural = accAreaSeminatural + areaPolygon
        if isCropfield:   
            accAreaCropFields  = accAreaCropFields + areaPolygon
            crops = np.append(crops, dfEsyrce.loc[index].D5_CUL)
        totalArea = totalArea + areaPolygon
    
    seminaturalPercentage = accAreaSeminatural / totalArea
    if len(crops) > 0:
        avCropfieldSize = accAreaCropFields / len(crops)
    heterogeneity = len(np.unique(crops)) / totalArea
    
    dictOut = {'seminaturalPercentage': seminaturalPercentage, 
               'avCropfieldSize': avCropfieldSize, 
               'heterogeneity': heterogeneity}   
    return dictOut;


"""
INPUT: a block from ESYRCE data (squares of 700x700m or 500x500m) 
OUTPUT: demand averaged over the area of the polygons
"""
def calculateDemand(block):
    # Read crop codes
    with open(pollReliance, mode='r') as infile:
        reader    = csv.reader(infile)
        dictDemand = {rows[0]:rows[1] for rows in reader} # key: 'ESYRCE_code; value: 'Demand'

    total_area = 0
    acc_demand = 0
    for index in block.index:
        try:
            code = block.loc[index].D5_CUL
            assocElts = code.split("-")
            demand = 0
            if len(assocElts) > 0:
                for elt in assocElts: # average over all elements
                    increase = dictDemand[elt]
                    demand = demand + dictPollValues[increase]      
                demand = demand / len(assocElts)
        except:
            demand = 0
        area = block.loc[index].geometry.area
        acc_demand = acc_demand + demand*area  
        total_area = total_area + area
    return acc_demand/total_area;


"""
INPUT: any subset of ESYRCE data
OUTPUT: the input dataframe plus a new column of demand averaged over the area of the polygons
"""
def addDemandBlockAvg(dfEsyrce, stepsSave, backupFile):
    
    dfEsyrce['block_demand'] = np.nan
    blockNrs = np.unique(dfEsyrce.D2_NUM)
    contNr = 0
    totalNr = len(blockNrs) 
    for blockNr in blockNrs:
        selectedInd = dfEsyrce.D2_NUM == blockNr
        block = [dfEsyrce.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
        block = gpd.GeoDataFrame(block)
        blockDemand = calculateDemand(block)
        dfEsyrce.at[selectedInd,'block_demand'] = blockDemand
        
        contNr = contNr+1
        if np.mod(contNr, 100) == 0:
            times = contNr / totalNr 
            print("addDemandBlockAvg...", np.floor(times*100), "percent completed...")
        
        if np.mod(contNr, stepsSave) == 0:
            times = contNr / totalNr 
            dill.dump_session(backupFile)
            print("Saved session... " + backupFile)
    return dfEsyrce;


"""
INPUT: any subset of ESYRCE data
OUTPUT: the input dataframe plus a new column of demand for each polygon
"""
def addDemand(dfEsyrce, stepsSave, backupFile):
    
    dfEsyrce['demand'] = np.nan
    # Read crop codes
    with open(pollReliance, mode='r') as infile:
        reader    = csv.reader(infile)
        dictDemand = {rows[0]:rows[1] for rows in reader} # key: 'ESYRCE_code; value: 'Demand'
    
    contNr = 0
    totalNr = len(dfEsyrce) 
    for index in dfEsyrce.index:       
        try:
            code = dfEsyrce.loc[index].D5_CUL
            assocElts = code.split("-")
            demand = 0
            if len(assocElts) > 0:
                for elt in assocElts: # average over all elements
                    increase = dictDemand[elt]
                    demand = demand + dictPollValues[increase]      
                demand = demand / len(assocElts)
        except:
            demand = 0
            
        dfEsyrce.at[index,'demand'] = demand
    
        contNr = contNr+1
        if np.mod(contNr, 10000) == 0:
            times = contNr / totalNr 
            print("addDemand...", np.floor(times*100), "percent completed...")
    
        if np.mod(contNr, stepsSave) == 0:
            times = contNr / totalNr 
            dill.dump_session(backupFile)
            print("Saved session... " + backupFile)
    return dfEsyrce;


"""
INPUT: a block from ESYRCE data (squares of 700x700m or 500x500m) 
OUTPUT: the area of the block
"""
def getBlockArea(block):
    totalArea = 0
    for index in block.index:
        areaPolygon = block.loc[index].Shape_Area
        totalArea = totalArea + areaPolygon
    return totalArea*1e-6; # m^2 to km^2
