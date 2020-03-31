# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import geopandas as gpd
import numpy as np
import pandas as pd

# load file from local path
layer = "z28"
data = gpd.read_file('..\\..\\..\\DATA\\Observ\\LandCover\\ESYRCE\\Esyrce2001_2016.gdb', layer=layer)

if layer == "z28":
    crs = "EPSG:32628"

if layer == "z30":
    crs = "EPSG:32630"

##################
# Select plots
validPlots = np.array([0])

# Loop plot numbers
plotNrs = np.unique(data.D2_NUM)
totalNr = len(plotNrs) 
contNr  = 0
emptyDF = True
for plotNr in plotNrs:
    
    selectedInd = data.D2_NUM == plotNr
    dataPlotsNr = [data.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
    dataPlotsNr = gpd.GeoDataFrame(dataPlotsNr)
    
    # Loop years: check every year has the same spatial extent
    years = np.unique(dataPlotsNr.YEA)
    cont  = 0
    validNr = True
    for year in years:
        
        selectedInd   = dataPlotsNr.YEA == year
        dataPlotsYear = [dataPlotsNr.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
        dataPlotsYear = gpd.GeoDataFrame(dataPlotsYear)
        dissolved     = dataPlotsYear.dissolve(by='YEA')    
        newDissGeo    = dissolved.geometry
        newBBox       = newDissGeo.bounds
        
        if (cont == 0):  
            lastBBox = newBBox
            cont = cont+1
        else:
            condition = np.isclose(lastBBox.iloc[0].reset_index(drop=True), newBBox.iloc[0].reset_index(drop=True), rtol=1e-6) 
            if all(condition):
                lastBBox = newBBox
                cont = cont+1
            else:
                validNr = False
                break
        
    if validNr: 
        if emptyDF:
            validData = dataPlotsNr
            validData.crs = crs
            emptyDF = False
        else:
            validData = pd.concat([validData, dataPlotsNr], ignore_index=True) 
    
    contNr = contNr+1
    if np.mod(contNr, 10) == 0:
        times = contNr / totalNr 
        print("Processing data...", np.floor(times*100), "percent completed...")

# To file
validData.to_file(filename="..\\..\\..\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\validData.shp", driver="ESRI Shapefile")

