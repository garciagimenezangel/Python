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
dill.loadSession(session)

# Data: seminatural, fieldsize, heterogeneity, demand
# Pintar con colores por comunidad autónoma

# seminatural
def getColor(point, polygons, palette):
    i = np.where(point.within(polygons))
    return(palette[i])    

# polygons = ... sacar de shapefile de com.aut.
# palette = ... hacer paleta de colores de dimensión igual que polygons
points = seminatural['geometry']
df = seminatural.T
df = pd.DataFrame(df)
df = df.drop(['geometry'], axis=0)
df.plot(color=[getColor(points[i],polygons, palette) for i in range(0:len(df.columns)) ], legend=False)
plt.show()

a=seminatural[0:10]
b=a.T
b = b.drop(['geometry'], axis=0)
b = pd.DataFrame(b)
b.plot(legend=False)