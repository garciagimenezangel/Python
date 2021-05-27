# -*- coding: utf-8 -*-
"""
Created on Thu May 27 11:56:20 2021

@author: angel
"""
import geopandas as gpd
import pandas as pd
from os.path import expanduser
import glob
home = expanduser("~")

inputESYRCE  = home + '\\DATA\\ESYRCE\\PROCESSED - local testing\\z30\\flagged\\all\\'
flag0 = glob.glob(inputESYRCE + "data_flag0*.shp")
gdf0 = pd.concat([
    gpd.read_file(shp)
    for shp in flag0
]).pipe(gpd.GeoDataFrame)

flag1 = glob.glob(inputESYRCE + "data_flag1*.shp")
gdf1 = pd.concat([
    gpd.read_file(shp)
    for shp in flag1
]).pipe(gpd.GeoDataFrame)

flag2 = glob.glob(inputESYRCE + "data_flag2*.shp")
gdf2 = pd.concat([
    gpd.read_file(shp)
    for shp in flag2
]).pipe(gpd.GeoDataFrame)

len0 = gdf0.groupby(['D1_HUS','D2_NUM']).size()
len1 = gdf1.groupby(['D1_HUS','D2_NUM']).size()
len2 = gdf2.groupby(['D1_HUS','D2_NUM']).size()
print("Flag 0: ", len(len0))
print("Flag 1: ", len(len1))
print("Flag 2: ", len(len2))