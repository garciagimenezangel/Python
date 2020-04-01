# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import geopandas as gpd
import numpy as np

from os.path import expanduser
home = expanduser("~")

# INPUT
layer = "z28"
inputFile = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\esyrceProcessed_' + layer + '.shp'

# Read file
processedData = gpd.read_file(inputFile)

if layer == 'z28':
    crs = "EPSG:32628"

if layer == 'z30':
    crs = "EPSG:32630"
    
# To files, by year
years = np.unique(processedData.YEA)
for year in years:
    selectedInd   = processedData.YEA == year
    validDataYear = [processedData.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
    validDataYear = gpd.GeoDataFrame(validDataYear)
    validDataYear.crs = crs
    validDataYear.to_file(filename = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\esyrceProcessed_'+layer+'_'+str(year)+".shp", driver="ESRI Shapefile")
