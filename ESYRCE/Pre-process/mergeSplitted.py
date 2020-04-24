# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import geopandas as gpd
import pandas as pd
import glob
from os.path import expanduser
home = expanduser("~")

# INPUT
layer = 'z30'
rootFilename = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\esyrceFiltered_' + layer

# OUTPUT
outFilename = rootFilename + "_merged.shp"

# Get filtered files
listFiltered = glob.glob(rootFilename+"*.shp")

# Concat dataframes
data = gpd.read_file(listFiltered[0])
print("Read file:", listFiltered[0])
frames = [data]
for file in listFiltered[1:]:
    data = gpd.read_file(file)
    frames.append(data)
    print("Read file:", file)
result = pd.concat(frames)

# To file 
result.to_file(filename = outFilename, driver="ESRI Shapefile")
print("Saved file:", outFilename)

