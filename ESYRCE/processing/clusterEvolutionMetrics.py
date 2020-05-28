# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:24:41 2020

@author: angel.gimenez

Cluster time-series in defined groups.
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

# INPUT
session = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\z30\\sessions\\timeSeries.pkl'
dill.load_session(session)
crs  = "EPSG:23030"

# OUTPUT
rootDir = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\z30\\time-series trajectories\\'

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
        
        # Creatr trajectory
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

    
###########################################
# Seminatural
seminatural = calculateTrajectoryBins(seminatural)
saveTrajectories(seminatural, 'seminatural')

# Demand
demand = calculateTrajectoryBins(demand, 0.25, 0.5)
saveTrajectories(demand, 'demand')

# Heterogeneity
heterogeneity = calculateTrajectoryBins(heterogeneity, 5, 10)
saveTrajectories(heterogeneity, 'heterogeneity')

# Field size
fieldsize = calculateTrajectoryBins(fieldsize, 10000, 50000)
saveTrajectories(fieldsize, 'fieldsize')

