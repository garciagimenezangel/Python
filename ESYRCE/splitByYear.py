# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import geopandas as gpd
import numpy as np

# load file from local path
validData = gpd.read_file("..\\..\\..\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\validData.shp")

# To files, by year
years = np.unique(validData.YEA)
for year in years:
    selectedInd   = validData.YEA == year
    validDataYear = [validData.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
    validDataYear = gpd.GeoDataFrame(validDataYear)
    validDataYear.to_file(filename="..\\..\\..\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\validData"+str(year)+".shp", driver="ESRI Shapefile")
