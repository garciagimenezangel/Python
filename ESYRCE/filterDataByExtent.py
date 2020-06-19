# -*- coding: utf-8 -*-
"""
INPUT: ESYRCE database (.gdb file)
OUTPUT: shapefiles (data is saved every 1000 blocks) with the ESYRCE blocks where 
Notes: sometimes, the area of the blocks changes from one year to another. It seems (from the exploration
of the data in QGIS) that some of the blocks are wrongly referenced (they are shifted in space), and some are 
different because of a change in the survey area (e.g from 500x500m to 700x700m or vice versa). These situations
are flagged to deal with them as good as possible
"""
import dill
import geopandas as gpd
import numpy as np
import pandas as pd
from os.path import expanduser
home = expanduser("~")

import sys
sys.path.append(home + '\\Documents\\REPOSITORIES\\Python\\ESYRCE\\')
import blockCalculator as bc 

# INPUT
inputESYRCE = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\Esyrce2001_2016.gdb'
layer = 'z30'
# load file from local path or saved Python session
data = gpd.read_file(inputESYRCE, layer=layer)
#data = gpd.read_file(home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\z30\\tests\\testNorth.shp')

# OUTPUT
rootFilename = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\' + layer + '\\filtered\\data' # output is splitted in many files-> rootFilename_NUMBER.shp

if layer == 'z28':
    crs = "EPSG:23028"

if layer == 'z30':
    crs = "EPSG:23030"

##################
# Loop block numbers
blockNrs = np.unique(data.D2_NUM)
totalNr = len(blockNrs) 
contNr  = 0
emptyDF = True
contSavedFiles = 0
for blockNr in blockNrs[1:]:
    
    # Select block data
    selectedInd = data.D2_NUM == blockNr
    dataBlockNr = [data.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
    dataBlockNr = gpd.GeoDataFrame(dataBlockNr)
    
    # Three cases seen in the exploration of the data: every year same spatial cover, spatial cover reduced from 700x700m to 500x500m after 2007 and data wrong in 2001, 2002. 
    # Check in which case we are: 0:ok; 1:size changed, clip to smallest; 2: data not aligned, skip; 3: geometry problems dissolving, skip data
    flag = bc.getBlockQualityFlag(dataBlockNr);
    if (flag == 0):
        if emptyDF:
            validData = dataBlockNr
            validData.crs = crs
            emptyDF = False
        else:
            validData = pd.concat([validData, dataBlockNr], ignore_index=True)
    
    elif (flag == 1):
        minPolygon = bc.getPolygonToClip(dataBlockNr);
        intersDataBlockNr = gpd.overlay(dataBlockNr, minPolygon, how='intersection')
        if emptyDF:
            validData = intersDataBlockNr
            validData.crs = crs
            emptyDF = False
        else:
            validData = pd.concat([validData, intersDataBlockNr], ignore_index=True)
    
    contNr = contNr+1
    if np.mod(contNr, 10) == 0:
        times = contNr / totalNr 
        print("Processing data...", np.floor(times*100), "percent completed...")
        
    # If input data is very large, save file every certain number of blocks processed, e.g. 1000
    if np.mod(contNr, 1000) == 0:
        processedFilename = rootFilename + '_' + str(contSavedFiles) + '.shp'
        validData.to_file(filename = processedFilename, driver="ESRI Shapefile")
        print("Saved file:", processedFilename)
        validData = None
        emptyDF = True
        contSavedFiles = contSavedFiles + 1

# To file 
processedFilename = rootFilename + '_' + str(contSavedFiles) + '.shp'
validData.to_file(filename = processedFilename, driver="ESRI Shapefile")
print("Saved file:", processedFilename)

