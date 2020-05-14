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
    indSameX  = np.where(np.isclose(pt.x, centroids['geometry'].x, atol=1e0))
    dataSameX = centroids.iloc[indSameX]
    indSameY  = np.where(np.isclose(pt.y, dataSameX['geometry'].y, atol=1e0))
    dataYear  = dataSameX.iloc[indSameY]
#    indices = [i for i in range(0,len(centroids)) if np.isclose(pt.x, centroids.at[i,'geometry'].x, atol=1e0) and np.isclose(pt.y, centroids.at[i,'geometry'].y, atol=1e0)]    
#    dataYear = centroids.iloc[indices]
    for index, row in dataYear.iterrows():
        year = str(row['year'])
        seminatural.at[indexPt, year] = row['seminatural']
        
    contNr = contNr+1
    if np.mod(contNr, 100) == 0:
        times = contNr / totalNr 
        print("Creating time series seminatural habitat percentage...", np.floor(times*100), "percent completed...")
backupFile = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\timeSeries.pkl'
dill.dump_session(backupFile)
print("Time series seminatural habitat percentage FINISHED... " + backupFile)

# Evolution of fieldsize (average size of crop fields)
fieldsize = gpd.GeoDataFrame()
fieldsize['geometry'] = uniquePts
for i in range(2001,2017): fieldsize[str(i)]=np.nan
contNr  = 0
totalNr = len(fieldsize)
for indexPt in fieldsize.index:
    pt = fieldsize.at[indexPt,'geometry']
    indSameX  = np.where(np.isclose(pt.x, centroids['geometry'].x, atol=1e0))
    dataSameX = centroids.iloc[indSameX]
    indSameY  = np.where(np.isclose(pt.y, dataSameX['geometry'].y, atol=1e0))
    dataYear  = dataSameX.iloc[indSameY]
#    indices = [i for i in range(0,len(centroids)) if np.isclose(pt.x, centroids.at[i,'geometry'].x, atol=1e0) and np.isclose(pt.y, centroids.at[i,'geometry'].y, atol=1e0)]    
#    dataYear = centroids.iloc[indices]
    for index, row in dataYear.iterrows():
        year = str(row['year'])
        fieldsize.at[indexPt, year] = row['fieldsize']
        
    contNr = contNr+1
    if np.mod(contNr, 100) == 0:
        times = contNr / totalNr 
        print("Creating time series fieldsize average...", np.floor(times*100), "percent completed...")
backupFile = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\timeSeries.pkl'
dill.dump_session(backupFile)
print("Time series fieldsize average FINISHED... " + backupFile)

# Evolution of heterogeneity (number of different crops per km^2)
heterogeneity = gpd.GeoDataFrame()
heterogeneity['geometry'] = uniquePts
for i in range(2001,2017): heterogeneity[str(i)]=np.nan
contNr  = 0
totalNr = len(heterogeneity)
for indexPt in heterogeneity.index:
    pt = heterogeneity.at[indexPt,'geometry']
    indSameX  = np.where(np.isclose(pt.x, centroids['geometry'].x, atol=1e0))
    dataSameX = centroids.iloc[indSameX]
    indSameY  = np.where(np.isclose(pt.y, dataSameX['geometry'].y, atol=1e0))
    dataYear  = dataSameX.iloc[indSameY]
#    indices = [i for i in range(0,len(centroids)) if np.isclose(pt.x, centroids.at[i,'geometry'].x, atol=1e0) and np.isclose(pt.y, centroids.at[i,'geometry'].y, atol=1e0)]    
#    dataYear = centroids.iloc[indices]
    for index, row in dataYear.iterrows():
        year = str(row['year'])
        heterogeneity.at[indexPt, year] = row['heterogeneity']
        
    contNr = contNr+1
    if np.mod(contNr, 100) == 0:
        times = contNr / totalNr 
        print("Creating time series heterogeneity...", np.floor(times*100), "percent completed...")
backupFile = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\timeSeries.pkl'
dill.dump_session(backupFile)
print("Time series heterogeneity FINISHED... " + backupFile)

# Evolution of demand (pollinator's demand averaged over area)
demand = gpd.GeoDataFrame()
demand['geometry'] = uniquePts
for i in range(2001,2017): demand[str(i)]=np.nan
contNr  = 0
totalNr = len(demand)
for indexPt in demand.index:
    pt = demand.at[indexPt,'geometry']
    indSameX  = np.where(np.isclose(pt.x, centroids['geometry'].x, atol=1e0))
    dataSameX = centroids.iloc[indSameX]
    indSameY  = np.where(np.isclose(pt.y, dataSameX['geometry'].y, atol=1e0))
    dataYear  = dataSameX.iloc[indSameY]
#    indices = [i for i in range(0,len(centroids)) if np.isclose(pt.x, centroids.at[i,'geometry'].x, atol=1e0) and np.isclose(pt.y, centroids.at[i,'geometry'].y, atol=1e0)]    
#    dataYear = centroids.iloc[indices]
    for index, row in dataYear.iterrows():
        year = str(row['year'])
        demand.at[indexPt, year] = row['demand']
        
    contNr = contNr+1
    if np.mod(contNr, 100) == 0:
        times = contNr / totalNr 
        print("Creating time series demand...", np.floor(times*100), "percent completed...")
backupFile = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\timeSeries.pkl'
dill.dump_session(backupFile)
print("Time series demand FINISHED... " + backupFile)




