# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:29:06 2020

@author: angel.gimenez
"""
import numpy as np
import geopandas as gpd

# file from local path
mallard = gpd.read_file('.\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Esyrce2001_2016.gdb', layer="z30")



selected = np.array([])
index = 0
n= len(sandias)
for i in range(0,n-1):
    for j in range(0,n-1):
        if i == j: continue
        target = sandias.iloc[i]
        field  = sandias.iloc[j] 
        condition1 = (target["X_COORD"] <  field["X_COORD"]+10000) and (target["X_COORD"] > field["X_COORD"]-10000)
        condition2 = (target["Y_COORD"] <  field["Y_COORD"]+10000) and (target["Y_COORD"] > field["Y_COORD"]-10000)
        condition3 = (target["YEA"]     == field["YEA"])
        condition4 = (target["X_COORD"] != 0)
        condition5 = (target["D2_NUM"] != field["D2_NUM"])
        if (condition1 and condition2 and condition3 and condition4 and condition5):
            selected = np.append(selected, [i])

selected = np.unique(selected)
