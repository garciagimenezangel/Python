# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:56:21 2020

@author: angel.gimenez
"""

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
        code = dfEsyrce.loc[index].D5_CUL
        if "-" in code:
            assocElts = code.split("-")
            #lenElts = [len(elt) for elt in assocElts]
            
            # Save first element
            try:
                increase = dictDemand[assocElts[0]]
            except:
                increase = ''
                    
        else:
            try:
                increase = dictDemand[code[0:2]]
            except:
                increase = ''
        
        demand = dictPollValues[increase]
        area = dfEsyrce.loc[index].Shape_Area
        output = output + demand*area
    
    return output;


