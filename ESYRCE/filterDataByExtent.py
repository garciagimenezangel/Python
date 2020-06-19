# -*- coding: utf-8 -*-
"""
INPUT: ESYRCE database (.gdb file)
OUTPUT: shapefiles (data is saved every 1000 blocks) with the ESYRCE blocks where 
"""
import dill
import geopandas as gpd
import numpy as np
import pandas as pd
from os.path import expanduser
home = expanduser("~")

# INPUT
inputESYRCE = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\Esyrce2001_2016.gdb'
layer = 'z30'
# load file from local path or saved Python session
#data = gpd.read_file(inputESYRCE, layer=layer)
data = 

# OUTPUT
rootFilename = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\' + layer + '\\filtered\\data'

if layer == 'z28':
    crs = "EPSG:23028"

if layer == 'z30':
    crs = "EPSG:23030"

##################
# Select plots
validPlots = np.array([0])

# Loop block numbers
blockNrs = np.unique(data.D2_NUM)
totalNr = len(blockNrs) 
contNr  = 0
emptyDF = True
contSavedFiles = 0
for blockNr in blockNrs:
    
    ii = np.where(data.D2_NUM == blockNr)
    i0 = ii[0][0]
    iM = ii[0][len(ii[0])-1]
    dataBlockNr = data[i0:(iM+1)]
    if (iM-i0+1)!=len(ii[0]): 
        print("Warning Block nr:",blockNr," excluded. Indices not consecutive")
        validNr = False
    
    if validNr:
        # Loop years: check every year has the same spatial extent
        years = np.unique(dataBlockNr.YEA)
        cont  = 0
        validNr = True
        for year in years:
            
            selectedInd   = dataBlockNr.YEA == year
            dataBlockYear = [dataBlockNr.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
            dataBlockYear = gpd.GeoDataFrame(dataBlockYear)
            try:
                dissolved     = dataBlockYear.dissolve(by='YEA')    
                newDissGeo    = dissolved.geometry
                newBBox       = newDissGeo.bounds
            except:
                print("Warning Block nr:",blockNr," excluded. Geometry problems")
                validNr = False
                break
            
            if (cont == 0):  
                lastBBox = newBBox
                cont = cont+1
            else:
                condition = np.isclose(lastBBox.iloc[0].reset_index(drop=True), newBBox.iloc[0].reset_index(drop=True), rtol=1e-6) 
                if all(condition):
                    lastBBox = newBBox
                    cont = cont+1
                else:
                    print("Warning Block nr:",blockNr," excluded. It does not meet the required conditions")
                    validNr = False
                    break
        
    if validNr: 
        if emptyDF:
            validData = dataBlockNr
            validData.crs = crs
            emptyDF = False
        else:
            validData = pd.concat([validData, dataBlockNr], ignore_index=True) 
    
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

