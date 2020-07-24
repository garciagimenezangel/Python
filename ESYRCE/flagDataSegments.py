# -*- coding: utf-8 -*-
"""
The data from ESYRCE needs some preprocessing because of the following reasons:
The area of the segments may change from one year to another. 
Some of the segments are wrongly referenced (they are shifted in space), and some are 
different because of a change in the survey area (e.g from 500x500m to 700x700m or vice versa). 
These situations are flagged to deal with them as good as possible

INPUT: ESYRCE database (2001-2016: gdb file; 2017-2019: shapefile)
OUTPUT: shapefiles for each flag. 0:ok; 1:size changed, clip to smallest; 2: data not aligned
"""

import pandas as pd
import geopandas as gpd
import numpy as np
from datetime import datetime
from os.path import expanduser
home = expanduser("~")

import sys
sys.path.append(home + '/Documents/REPOSITORIES/Python/ESYRCE/')
import functions 

# INPUT
inputESYRCE_2001_2016 = home + '/Documents/DATA/OBServ/ESYRCE/Esyrce2001_2016.gdb'
inputESYRCE_2017_2019 = home + '/Documents/DATA/OBServ/ESYRCE/Esyrce2017_2019/Z30.shp'
layer = 'z30'
tol  = 10.0 # tolerance, in meters, to check whether the segments are aligned

# OUTPUT
rootFilename = home + '/Documents/DATA/OBServ/ESYRCE/PROCESSED/'+layer+'/flagged/data'
processedFilename0 = rootFilename + '_flag0.shp'
processedFilename1 = rootFilename + '_flag1.shp'
processedFilename2 = rootFilename + '_flag2.shp'
print("PROCESS flagDataSegments.py STARTED AT: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'\n')

# load files from local path, concatenate the data and sort the rows in order to iterate through them faster, using numpy.where (which seems to work better than dataframe.where for very large dataframes like these ones)
data2001_20016 = gpd.read_file(inputESYRCE_2001_2016, layer=layer)
data2017_20019 = gpd.read_file(inputESYRCE_2017_2019)
data = pd.concat([data2001_20016, data2017_20019], ignore_index=True)
data = data.dropna(thresh=1)
data = data.where(data['D1_HUS'] != 0)
data = data.where(data['D2_NUM'] != 0)
data.sort_values(by=['D1_HUS','D2_NUM','YEA'], inplace = True)
data.reset_index(drop=True, inplace=True)

##################
# Loop zones
dataFlag0 = pd.DataFrame()
dataFlag1 = pd.DataFrame()
dataFlag2 = pd.DataFrame()
zoneNrs = np.unique(data.D1_HUS)
zoneNrs = zoneNrs[~np.isnan(zoneNrs)]
for zoneNr in zoneNrs:
    # Select zone data 
    ii = np.where(data.D1_HUS == zoneNr) 
    i0 = ii[0][0]
    iM = ii[0][len(ii[0])-1]
    dataZoneNr = data[i0:(iM+1)]
    if (iM-i0+1)!=len(ii[0]): # sanity check
        print("Error... Exit loop in zone nr:"+str(zoneNr)+'\n') # sanity check
        break
    
    # Loop segments in zone
    segmentNrs = np.unique(dataZoneNr.D2_NUM)
    totalNr = len(segmentNrs)
    contNr  = 0
    contSavedFiles = 0
    for segmentNr in segmentNrs:  #skip first one because it is equal to zero and it has no information
    
        # Select segment data (not very elegant way, but using numpy seems to be the fastest option)
        ii = np.where(dataZoneNr.D2_NUM == segmentNr) 
        i0 = ii[0][0]
        iM = ii[0][len(ii[0])-1]
        dataSegmentNr = dataZoneNr[i0:(iM+1)]
        if (iM-i0+1)!=len(ii[0]): # sanity check
            print("Error... Exit loop in Segment nr:"+str(segmentNr)+'\n') # sanity check
            break
        
        # Three cases seen in the exploration of the data: every year same spatial cover, spatial cover reduced from 700x700m to 500x500m after 2007 and data wrong in some years (I think that 2001, 2002). 
        # Check in which case we are: 0:ok; 1:size changed, clip to smallest; 2: data not aligned; 3: geometry problems dissolving
        flag = functions.getSegmentQualityFlag(dataSegmentNr, tol)
        if (flag == 0):
            dataFlag0 = pd.concat([dataFlag0, dataSegmentNr], ignore_index=True)
            
        elif (flag == 1):
            minPolygon = functions.getPolygonToClip(dataSegmentNr);
            try:
                intersDataSegmentNr = gpd.overlay(dataSegmentNr, minPolygon, how='intersection')
                dataFlag1 = pd.concat([dataFlag1, intersDataSegmentNr], ignore_index=True)
            except:
                print("Warning (flagDataSegments): segment skipped because of problems performing intersection "+str(dataSegmentNr.iloc[0]['D2_NUM'])+'\n')
    
        elif (flag == 2):
            dataFlag2 = pd.concat([dataFlag2, dataSegmentNr], ignore_index=True)       
    
        contNr = contNr+1
        if np.mod(contNr, 10) == 0:
            times = contNr / totalNr 
            print("Processing data Zone..."+str(int(zoneNr))+" Percentage completed..."+str(np.floor(times*100))+'\n')
        

## To shapefiles
if layer == 'z28': crs = "EPSG:23028"
if layer == 'z30': crs = "EPSG:23030"
print("Writing file..."+processedFilename0+'\n')
dataFlag0 = gpd.GeoDataFrame(dataFlag0)
dataFlag0.crs = crs
dataFlag0.to_file(filename = processedFilename0, driver="ESRI Shapefile")
print("Saved file: "+processedFilename0+'\n')
print("Writing file..."+processedFilename1+'\n')
dataFlag1 = gpd.GeoDataFrame(dataFlag1)
dataFlag1.crs = crs
dataFlag1.to_file(filename = processedFilename1, driver="ESRI Shapefile")
print("Saved file: "+processedFilename1+'\n')
print("Writing file..."+processedFilename2+'\n')
dataFlag2 = gpd.GeoDataFrame(dataFlag2)
dataFlag2.crs = crs
dataFlag2.to_file(filename = processedFilename2, driver="ESRI Shapefile")
print("Saved file: "+processedFilename2+'\n')

