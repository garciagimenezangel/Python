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
pollReliance        = home + '\\Google Drive\\PROJECTS\\OBSERV\\Lookup Tables\\Cultivar-Demand.csv'
cropNatArt          = home + '\\Google Drive\\PROJECTS\\OBSERV\\Lookup Tables\\CropNaturalArtificial.csv'
cropNatArt_detailed = home + '\\Google Drive\\PROJECTS\\OBSERV\\Lookup Tables\\CropNaturalArtificial_detailed.csv'
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
INPUT: subset of ESYRCE data corresponding to one block number
OUTPUT: polygon corresponding to the data of the year with the smallest spatial coverage
"""
def getPolygonToClip(dataBlockNr):
    years = np.unique(dataBlockNr.YEA)
    cont = 0
    for year in years:
        selectedInd   = dataBlockNr.YEA == year
        dataBlockYear = [dataBlockNr.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
        dataBlockYear = gpd.GeoDataFrame(dataBlockYear)
        try:
            dissolved = dataBlockYear.dissolve(by='YEA')    
        except:
            print("Warning: problems dissolving block ", dataBlockNr.iloc[0]['D2_NUM'])
            return gpd.GeoDataFrame()
            
        if (cont == 0):  
            intersection = dissolved
            cont = cont+1
        else:
            intersection = gpd.overlay(intersection, dissolved, how='intersection')
    
    return gpd.GeoDataFrame(intersection.geometry)

"""
INPUT: subset of ESYRCE data corresponding to one block number
OUTPUT: quality flag. 0:ok; 1:size changed, clip to smallest; 2: blocks not aligned; 3: geometry problems dissolving
"""
def getBlockQualityFlag(dataBlockNr):
    flag = 0 
    years = np.unique(dataBlockNr.YEA)
    cont = 0
    for year in years:
        selectedInd   = dataBlockNr.YEA == year
        dataBlockYear = [dataBlockNr.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
        dataBlockYear = gpd.GeoDataFrame(dataBlockYear)
        try:
            dissolved     = dataBlockYear.dissolve(by='YEA')    
            newDissGeo    = dissolved.geometry
            newBBox       = newDissGeo.bounds
        except:
            return 3
            
        if (cont == 0):  
            minBBox = newBBox
            cont = cont+1
        else:
            lastMinX = minBBox.iloc[0][0];
            lastMinY = minBBox.iloc[0][1];            
            newMinX  = newBBox.iloc[0][0];
            newMinY  = newBBox.iloc[0][1];
            lastMaxX = minBBox.iloc[0][2];
            lastMaxY = minBBox.iloc[0][3];            
            newMaxX  = newBBox.iloc[0][2];
            newMaxY  = newBBox.iloc[0][3];      
            if (not(np.isclose(lastMinX, newMinX, rtol=1e-6)) or not(np.isclose(lastMinY, newMinY, rtol=1e-6))):
                return 2
            elif (not(np.isclose(lastMaxX, newMaxX, rtol=1e-6)) or not(np.isclose(lastMaxY, newMaxY, rtol=1e-6))):
                flag = 1
                minBBox = newBBox
            else:
                minBBox = newBBox
                
    return flag

        
"""
INPUT: any subset of ESYRCE data
OUTPUT: dictionary with the values of intensification metrics
"""
def calculateIntensificationParametersDetailed(dfEsyrce):
    # Read LC codes
    with open(cropNatArt_detailed, mode='r') as infile:
        reader    = csv.reader(infile)
        dictCropNatArt = {rows[0]:rows[1] for rows in reader} # key: 'ESYRCE_code; value: 'Crop, Natural or Artificial'
    
    # Metrics to calculate
    accAreaCereal        = 0
    accAreaLegume        = 0
    accAreaTuber         = 0
    accAreaIndustrial    = 0
    accAreaFodder        = 0
    accAreaVegetable     = 0
    accAreaOrnamental    = 0
    accAreaEmptyGreenh   = 0
    accAreaCitrics       = 0
    accAreaFruitTree     = 0
    accAreaVineyard      = 0
    accAreaOlive         = 0
    accAreaOtherWoody    = 0
    accAreaNursery       = 0
    accAreaAssociation   = 0
    accAreaFallow        = 0
    accAreaOrchard       = 0
    accAreaGrasslandNat  = 0
    accAreaPastureMount  = 0
    accAreaPasture       = 0
    accAreaPastureShrub  = 0
    accAreaConifer       = 0
    accAreaBroadleafSlow = 0
    accAreaBroadleafFast = 0
    accAreaPoplar        = 0
    accAreaConiferBroad  = 0
    accAreaShrub         = 0
    accAreaWasteland     = 0
    accAreaSpartizal     = 0
    accAreaWasteToUrbanize = 0
    accAreaImproductive  = 0
    accAreaArtificial    = 0
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
INPUT: a block of one year from ESYRCE data (squares of 700x700m or 500x500m) 
OUTPUT: demand averaged over the area of the polygons
"""
def calculateDemand(blockYear):
    # Read crop codes
    with open(pollReliance, mode='r') as infile:
        reader    = csv.reader(infile)
        dictDemand = {rows[0]:rows[1] for rows in reader} # key: 'ESYRCE_code; value: 'Demand'

    total_area = 0
    acc_demand = 0
    for index in blockYear.index:
        try:
            code = blockYear.loc[index].D5_CUL
            assocElts = code.split("-")
            demand = 0
            if len(assocElts) > 0:
                for elt in assocElts: # average over all elements
                    increase = dictDemand[elt]
                    demand = demand + dictPollValues[increase]      
                demand = demand / len(assocElts)
        except:
            demand = 0
        area = blockYear.loc[index].geometry.area
        acc_demand = acc_demand + demand*area  
        total_area = total_area + area
    return acc_demand/total_area;


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
