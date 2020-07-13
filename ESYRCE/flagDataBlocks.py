# -*- coding: utf-8 -*-
"""
The area of the blocks may change from one year to another. 
Some of the blocks are wrongly referenced (they are shifted in space), and some are 
different because of a change in the survey area (e.g from 500x500m to 700x700m or vice versa). 
These situations are flagged to deal with them as good as possible

INPUT: ESYRCE database (2001-2016: gdb file; 2017-2019: shapefile)
OUTPUT: shapefiles for each flag. 0:ok; 1:size changed, clip to smallest; 2: data not aligned
"""

import geopandas as gpd
import numpy as np
import pandas as pd
from os.path import expanduser
home = expanduser("~")

import sys
sys.path.append(home + '\\Documents\\REPOSITORIES\\Python\\ESYRCE\\')
import functions 

# INPUT
inputESYRCE_2001_2016 = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\Esyrce2001_2016.gdb'
inputESYRCE_2017_2019 = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\Esyrce2017_2019\\Z30.shp'
layer = 'z30'
tol  = 1.0 # tolerance, in meters, to check whether the blocks are aligned

# load files from local path, concatenate the data and sort the rows in order to iterate through them faster, using numpy.where (which seems to work better than dataframe.where for very large dataframes like these ones)
data2001_20016 = gpd.read_file(inputESYRCE_2001_2016, layer=layer)
data2017_20019 = gpd.read_file(inputESYRCE_2017_2019)
data = pd.concat([data2001_20016, data2017_20019], ignore_index=True)
data.sort_values(by=['D2_NUM','YEA'], inplace = True)
data.reset_index(drop=True, inplace=True)

# OUTPUT
rootFilename = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\' + layer + '\\flagged\\data' # output is splitted in many files-> rootFilename_NUMBER.shp

if layer == 'z28':
    crs = "EPSG:23028"

if layer == 'z30':
    crs = "EPSG:23030"

data = gpd.read_file(rootFilename+ '_flag2.shp')
data.sort_values(by=['D2_NUM','YEA'], inplace = True)
data.reset_index(drop=True, inplace=True)

##################
# Loop block numbers
blockNrs = np.unique(data.D2_NUM)
totalNr = len(blockNrs) 
contNr  = 0
dataFlag0 = pd.DataFrame()
dataFlag1 = pd.DataFrame()
dataFlag2 = pd.DataFrame()
contSavedFiles = 0
for blockNr in blockNrs[1:]:  #skip first one because it is equal to zero and it has no information
    
    # Select block data (not very elegant way, but using numpy seems to be the fastest option)
    ii = np.where(data.D2_NUM == blockNr) 
    i0 = ii[0][0]
    iM = ii[0][len(ii[0])-1]
    dataBlockNr = data[i0:(iM+1)]
    
    # Three cases seen in the exploration of the data: every year same spatial cover, spatial cover reduced from 700x700m to 500x500m after 2007 and data wrong in some years (I think that 2001, 2002). 
    # Check in which case we are: 0:ok; 1:size changed, clip to smallest; 2: data not aligned; 3: geometry problems dissolving
    flag = functions.getBlockQualityFlag(dataBlockNr, tol)
    if (flag == 0):
        dataFlag0 = pd.concat([dataFlag0, dataBlockNr], ignore_index=True)
    
    elif (flag == 1):
        minPolygon = functions.getPolygonToClip(dataBlockNr);
        try:
            intersDataBlockNr = gpd.overlay(dataBlockNr, minPolygon, how='intersection')
            dataFlag1 = pd.concat([dataFlag1, intersDataBlockNr], ignore_index=True)
        except:
            print("Warning (flagDataBlocks): block skipped because of problems performing intersection ", dataBlockNr.iloc[0]['D2_NUM'])
    
    elif (flag == 2):
        dataFlag2 = pd.concat([dataFlag2, dataBlockNr], ignore_index=True)       
    
    contNr = contNr+1
    if np.mod(contNr, 10) == 0:
        times = contNr / totalNr 
        print("Processing data...", np.floor(times*100), "percent completed...")
        

## To files
processedFilename0 = rootFilename + '_flag0.shp'
processedFilename1 = rootFilename + '_flag1.shp'
processedFilename2 = rootFilename + '_flag2.shp'
dataFlag0.to_file(filename = processedFilename0, driver="ESRI Shapefile")
dataFlag1.to_file(filename = processedFilename1, driver="ESRI Shapefile")
dataFlag2.to_file(filename = processedFilename2, driver="ESRI Shapefile")
print("Saved file:", processedFilename0)
print("Saved file:", processedFilename1)
print("Saved file:", processedFilename2)

