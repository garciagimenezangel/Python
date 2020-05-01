# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:56:21 2020

@author: angel.gimenez
"""

import dill
import csv
import numpy as np
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
    heterogeneity = len(np.unique(crops))
    
    dictOut = {'seminaturalPercentage': seminaturalPercentage, 
               'avCropfieldSize': avCropfieldSize, 
               'heterogeneity': heterogeneity}   
    return dictOut;


def calculateDemand(dfEsyrce):
    # Read crop codes
    with open(pollReliance, mode='r') as infile:
        reader    = csv.reader(infile)
        dictDemand = {rows[0]:rows[1] for rows in reader} # key: 'ESYRCE_code; value: 'Demand'
    
    output = 0 
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
        area = dfEsyrce.loc[index].Shape_Area
        output = output + demand*area  
    return output;


def addDemand(dfEsyrce, stepsSave, backupFile):
    
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
            print("Processing data...", np.floor(times*100), "percent completed...")
    
        if np.mod(contNr, stepsSave) == 0:
            times = contNr / totalNr 
            dill.dump_session(backupFile)
            print("Saved session... " + backupFile)

    return dfEsyrce;

