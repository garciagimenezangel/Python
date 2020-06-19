# -*- coding: utf-8 -*-
"""
Created on Thu May 14 15:24:41 2020

@author: angel.gimenez

Plot each dataframe (seminatural, fieldsize, heterogeneity, demand),
with different colors for different Comunidades Autónomas
"""

import dill
import geopandas as gpd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from os.path import expanduser
home = expanduser("~")

# INPUT
session = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\timeSeries.pkl'
dill.load_session(session)

# Hitograms
bins = np.array([-40000, -10000,-1000,-100, 100, 1000, 10000, 40000])
plt.hist(fieldsize.slope,bins=bins)

bins = np.array([-50,-20,-10,-1,-0.1,0.1, 1, 10, 20,50])
plt.hist(heterogeneity.slope,bins=bins)

bins = np.array([-0.5,-0.1,-0.01,-0.001, 0.001, 0.01,0.1,0.5])
plt.hist(seminatural.slope,bins=bins)

bins = np.array([-0.1,-0.05,-0.01,-0.001, 0.001, 0.01,0.05,0.1])
plt.hist(demand.slope,bins=bins)