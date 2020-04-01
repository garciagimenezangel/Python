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
inputFile = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\processedData_'+layer+'.shp'

# Read file
processedData = gpd.read_file(inputFile)

# To files, by year
years = np.unique(processedData.YEA)
for year in years:
    selectedInd   = processedData.YEA == year
    validDataYear = [processedData.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
    validDataYear = gpd.GeoDataFrame(validDataYear)
    validDataYear.to_file(filename = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\processedData_'+layer+'_'+str(year)+".shp", driver="ESRI Shapefile")
