# -*- coding: utf-8 -*-
"""
Created on Mon Mar 23 15:21:11 2020

@author: angel.gimenez
"""
import dill   
       
# file                 
filename = 'C:\\Users\\angel.gimenez\\Documents\\REPOSITORIES\\Python\\ESYRCE\\gdbLayerZ30_LoadedSession.pkl'
dill.load_session(filename)

import numpy as np

listPar = mallard["D2_NUM"]
listParcelas = np.unique(listPar)

nMax   = len(listParcelas)
nFiles = 10
for i in range(1, nFiles+1):
    min =  int( (i-1) * nMax/nFiles )
    max =  int( i     * nMax/nFiles )    
    parcelas = listParcelas[min:max]
    
    # Select indices that are part of 'parcelas'
    selInd = np.empty([1,0], dtype=int)
    for j in range(0, len(mallard)):
        if mallard["D2_NUM"].iloc[j] in parcelas:
            selInd = np.append(selInd, j)
    
    dataExp = mallard.iloc[selInd]
    dataExp.to_file(filename="C:\\Users\\angel.gimenez\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Splitted\\esyrce"+str(i)+".shp", driver="ESRI Shapefile")
    