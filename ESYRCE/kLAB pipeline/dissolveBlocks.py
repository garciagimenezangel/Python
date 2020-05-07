# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import dill
import geopandas as gpd
import numpy as np
import glob

from os.path import expanduser
home = expanduser("~")

# INPUT
layer = "z30"
inputFolder = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\z28_yearly\\'
files = glob.glob(inputFolder+'*.shp')

if layer == 'z28':
    crs  = "EPSG:23028"
    epsg = 23028

if layer == 'z30':
    crs  = "EPSG:23030"
    epsg = 23030
    
# To files, by year
for file in files:
    dataYear = gpd.read_file(file)
    dataYear = dataYear.dissolve(by='D2_NUM')
    dataYear.to_file(filename = file, driver="ESRI Shapefile")
    print("File: "+file+" dissolved")
    
#    # Rasterize
#    rasterfile = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\'+layer+'_'+str(year)+".tiff"
#    field = 'detailcode'
#    rasterize.rasterize(shapefile, epsg, rasterfile, field, resolution)
