# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24

@author: angel.gimenez

Divide a field of a geodataframe by the area of the ESYRCE block
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import glob
from os.path import expanduser
home = expanduser("~")

import sys
sys.path.append(home + '\\Documents\\REPOSITORIES\\Python\\ESYRCE\\lib\\')
import blockCalculator as bc 

# INPUT
rootFilename = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\z30\\dissolved\\'
field = 'heterogene'
layer = 'z30'

# OUTPUT
outDir =  home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\z30\\'

# Get filtered files
files = glob.glob(rootFilename+"*.shp")

# Loop plot numbers
for file in files:
    data = gpd.read_file(file)
    dataSel = [data.iloc[i] for i in range(0,len(data)) if data.iloc[i].geometry.area < 500000]
    indLargeArea = [i for i in range(0,len(data)) if data.iloc[i].geometry.area >= 500000]
    # Some few blocks may have larger area because the original data may come with equal D2_NUM for blocks that are actually different. Rule out those blocks because the block metrics depending on area will be wrong.
    if len(indLargeArea)>0:
        print("Warning file "+file+": some blocks ruled out because dissolved regions are not valid. Number ruled out:",len(indLargeArea)," out of: ",len(data))
    dataSel = gpd.GeoDataFrame(dataSel)
    dataSel[field] = 1e6 * dataSel[field] / dataSel.area
    dataSel.crs = data.crs
    dataSel = dataSel.to_file(filename = outDir + str(int(dataSel.iloc[0].YEA)) + '.shp', driver="ESRI Shapefile")



