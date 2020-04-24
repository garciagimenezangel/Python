# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import geopandas as gpd
import numpy as np

from os.path import expanduser
home = expanduser("~")

import sys
sys.path.append(home + '\\Documents\\REPOSITORIES\\Python\\ESYRCE\\lib\\')
import rasterize

# INPUT
resolution = 100
layer = "z28"
inputFile = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\esyrceProcessed_' + layer + '.shp'

# Read file
processedData = gpd.read_file(inputFile)

if layer == 'z28':
    crs  = "EPSG:23028"
    epsg = 23028

if layer == 'z30':
    crs  = "EPSG:23030"
    epsg = 23030
    
# To files, by year
years = np.unique(processedData.YEA)
for year in years:
    selectedInd   = processedData.YEA == year
    validDataYear = [processedData.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
    validDataYear = gpd.GeoDataFrame(validDataYear)
    validDataYear.crs = crs
    shapefile = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\detailCode_'+layer+'_'+str(year)+".shp"
    validDataYear.to_file(filename = shapefile, driver="ESRI Shapefile")
    
    # Rasterize
    rasterfile = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\detailCode_'+layer+'_'+str(year)+".tiff"
    field = 'detailcode'
    rasterize.rasterize(shapefile, epsg, rasterfile, field, resolution)
