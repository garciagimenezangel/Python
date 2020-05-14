# -*- coding: utf-8 -*-
"""
Created on Thu May 14 11:27:43 2020

@author: angel.gimenez
"""

import dill
import geopandas as gpd
import numpy as np
from os.path import expanduser
home = expanduser("~")

# INPUT
inputFile = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\z30\\merged.shp'
data = gpd.read_file(inputFile)

# Compute centroids and add to data
centroids = data.centroid
data.centroids = np.nan
data['centroids'] = centroids

# Create new geodataframe with the desired columns
centroids = gpd.GeoDataFrame()
centroids['geometry']      = data['centroids']
centroids['seminatural']   = data['seminatura']
centroids['fieldsize']     = data['avCropfiel']
centroids['heterogeneity'] = data['heterogene']
centroids['demand']        = data['block_dema']
centroids['year']          = data['YEA']

# Evolution of seminatural habitat percentage
seminatural = gpd.GeoDataFrame()
uniquePts = centroids.geometry.unique()
seminatural['geometry'] = uniquePts
for i in range(2001,2017): seminatural[str(i)]=np.nan
contNr  = 0
totalNr = len(seminatural)
for indexPt in seminatural.index:
    pt = seminatural.at[indexPt,'geometry']
    indices = [i for i in range(0,len(centroids)) if np.isclose(pt.x, centroids.at[i,'geometry'].x, atol=1e0) and np.isclose(pt.y, centroids.at[i,'geometry'].y, atol=1e0)]    
    dataYear = centroids.iloc[indices]
    for indexYr in indices:
        year = str(dataYear.at[indexYr,'year'])
        seminatural.at[indexPt, year] = dataYear.at[indexYr,'seminatural']
        
    contNr = contNr+1
    if np.mod(contNr, 100) == 0:
        times = contNr / totalNr 
        print("Creating time series seminatural habitat percentage...", np.floor(times*100), "percent completed...")

# Evolution of fieldsize (average size of crop fields)
fieldsize = gpd.GeoDataFrame()
fieldsize['geometry'] = uniquePts
for i in range(2001,2017): fieldsize[str(i)]=np.nan
contNr  = 0
totalNr = len(fieldsize)
for indexPt in fieldsize.index:
    pt = fieldsize.at[indexPt,'geometry']
    indices = [i for i in range(0,len(centroids)) if np.isclose(pt.x, centroids.at[i,'geometry'].x, atol=1e0) and np.isclose(pt.y, centroids.at[i,'geometry'].y, atol=1e0)]    
    dataYear = centroids.iloc[indices]
    for indexYr in indices:
        year = str(dataYear.at[indexYr,'year'])
        fieldsize.at[indexPt, year] = dataYear.at[indexYr,'fieldsize']
        
    contNr = contNr+1
    if np.mod(contNr, 100) == 0:
        times = contNr / totalNr 
        print("Creating time series fieldsize average...", np.floor(times*100), "percent completed...")

# Evolution of heterogeneity (number of different crops per km^2)
heterogeneity = gpd.GeoDataFrame()
heterogeneity['geometry'] = uniquePts
for i in range(2001,2017): heterogeneity[str(i)]=np.nan
contNr  = 0
totalNr = len(heterogeneity)
for indexPt in heterogeneity.index:
    pt = heterogeneity.at[indexPt,'geometry']
    indices = [i for i in range(0,len(centroids)) if np.isclose(pt.x, centroids.at[i,'geometry'].x, atol=1e0) and np.isclose(pt.y, centroids.at[i,'geometry'].y, atol=1e0)]    
    dataYear = centroids.iloc[indices]
    for indexYr in indices:
        year = str(dataYear.at[indexYr,'year'])
        heterogeneity.at[indexPt, year] = dataYear.at[indexYr,'heterogeneity']
        
    contNr = contNr+1
    if np.mod(contNr, 100) == 0:
        times = contNr / totalNr 
        print("Creating time series heterogeneity...", np.floor(times*100), "percent completed...")

# Evolution of demand (pollinator's demand averaged over area)
demand = gpd.GeoDataFrame()
demand['geometry'] = uniquePts
for i in range(2001,2017): demand[str(i)]=np.nan
contNr  = 0
totalNr = len(demand)
for indexPt in demand.index:
    pt = demand.at[indexPt,'geometry']
    indices = [i for i in range(0,len(centroids)) if np.isclose(pt.x, centroids.at[i,'geometry'].x, atol=1e0) and np.isclose(pt.y, centroids.at[i,'geometry'].y, atol=1e0)]    
    dataYear = centroids.iloc[indices]
    for indexYr in indices:
        year = str(dataYear.at[indexYr,'year'])
        demand.at[indexPt, year] = dataYear.at[indexYr,'demand']
        
    contNr = contNr+1
    if np.mod(contNr, 100) == 0:
        times = contNr / totalNr 
        print("Creating time series demand...", np.floor(times*100), "percent completed...")

backupFile = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\timeSeries.pkl'
dill.dump_session(backupFile)
print("FINISHED... Saved session... " + backupFile)




