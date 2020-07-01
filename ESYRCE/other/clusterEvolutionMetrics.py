# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:24:41 2020

@author: angel.gimenez

1. Cluster time-series in defined groups (types of trajectory)
2. Calculate slope of the time-series using a linear regression.
"""

import dill
import geopandas as gpd
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt
from os.path import expanduser
home = expanduser("~")
import sys
sys.path.append(home + '\\Documents\\REPOSITORIES\\Python\\ESYRCE\\lib\\')
import blockCalculator as bc 
from sklearn import linear_model

# INPUT
session = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\z30\\sessions\\timeSeries.pkl'
dill.load_session(session)
crs  = "EPSG:23030"

# OUTPUT
rootDir = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\z30\\time-series\\'

def calculateSlopes(df):
    regr = linear_model.LinearRegression()
    slopes = np.array([])
    for index in df.index:
        X = np.array(range(2001,2017), dtype=int)
        Y = np.array(df.loc[index][1:17], dtype=float)
        valid = np.invert(np.isnan(Y))
        X = X[valid]
        Y = Y[valid]
        slope = np.nan
        if X.size > 2:
            regr.fit(X.reshape(-1,1),Y.reshape(-1,1))
            slope = regr.coef_[0][0]
        slopes = np.append(slopes, slope)   
    df['slope'] = slopes
    return df
        
####################
# Group years in three intervals: early(2001-2006), middle(2007-2011), late(2012-2016)
# For the metrics, for example group seminatural proportion values in three intervals: low(0-1/3), medium(1/3-2/3), high(2/3-1)
# These groupings leave 27 possible trajectories: e.g. early-low, middle-high, late-medium -> group trajectories into these 27 bins
def calculateTrajectoryBins(df, thr1=1/3, thr2=2/3):
    df['trajClass']='999'
    for index in df.index:
        # early interval
        earlyVals = df.loc[index][1:7]
        earlyAvg  = pd.Series.mean(earlyVals)
        if (math.isnan(earlyAvg)): earlyInterval = -1
        elif (earlyAvg < thr1)    : earlyInterval = 0    
        elif (earlyAvg < thr2)    : earlyInterval = 1
        else                     : earlyInterval = 2
        # middle interval
        middleVals = df.loc[index][7:12]
        middleAvg  = pd.Series.mean(middleVals)
        if (math.isnan(middleAvg)): middleInterval = -1
        elif (middleAvg < thr1)    : middleInterval = 0    
        elif (middleAvg < thr2)    : middleInterval = 1
        else                      : middleInterval = 2
        # late interval
        lateVals = df.loc[index][12:17]
        lateAvg  = pd.Series.mean(lateVals)
        if (math.isnan(lateAvg)): lateInterval = -1
        elif (lateAvg < thr1)    : lateInterval = 0    
        elif (lateAvg < thr2)    : lateInterval = 1
        else                    : lateInterval = 2
        
        # Deal with -1 values
        if (earlyInterval == -1):
            if(middleInterval == -1):
                if(lateInterval != -1):
                    earlyInterval  = lateInterval
                    middleInterval = lateInterval
            else:
                earlyInterval = middleInterval
        if (middleInterval == -1):
            if (lateInterval == -1):
                middleInterval = earlyInterval
                lateInterval   = earlyInterval
            else:
                middleInterval = np.ceil((earlyInterval + lateInterval)/2)
        if (lateInterval == -1):
            lateInterval = middleInterval
        
        # Create trajectory
        df.at[index,'trajClass'] = str(int(earlyInterval))+str(int(middleInterval))+str(int(lateInterval))
        
    return df

####################
# Save files by trajectory
def saveTrajectories(df, name):
    uniqueTraj = np.unique(df['trajClass'])
    for traj in uniqueTraj:
        trajIloc = np.where(df['trajClass'] == traj)
        trajElts = df.iloc[trajIloc]
        fileName = rootDir+name+"\\"+"traj"+traj+".shp"
        trajElts.crs = crs
        trajElts.to_file(filename = fileName, driver="ESRI Shapefile")    
        print("Saved Trajectory: "+traj+" in file: " + fileName)

def saveToShapefile(df, name):
    fileName = rootDir+name+".shp"
    df.crs = crs
    df.to_file(filename = fileName, driver="ESRI Shapefile")
    
###########################################
# Seminatural
seminatural = calculateTrajectoryBins(seminatural)
seminatural = calculateSlopes(seminatural)
saveToShapefile(seminatural, 'seminatural')
saveTrajectories(seminatural, 'seminatural')

# Demand
demand = calculateTrajectoryBins(demand, 0.25, 0.5)
demand = calculateSlopes(demand)
saveToShapefile(demand, 'demand')
saveTrajectories(demand, 'demand')

# Heterogeneity
heterogeneity = calculateTrajectoryBins(heterogeneity, 5, 10)
heterogeneity = calculateSlopes(heterogeneity)
saveToShapefile(heterogeneity, 'heterogeneity')
saveTrajectories(heterogeneity, 'heterogeneity')

# Field size
fieldsize = calculateTrajectoryBins(fieldsize, 5000, 20000)
fieldsize = calculateSlopes(fieldsize)
saveToShapefile(fieldsize, 'fieldsize')
saveTrajectories(fieldsize, 'fieldsize')

