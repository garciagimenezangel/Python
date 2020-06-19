# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 15:41:29 2020

@author: angel.gimenez
"""


import dill
import geopandas as gpd
import numpy as np
import pandas as pd
from os.path import expanduser
home = expanduser("~")


# INPUT
inputESYRCE = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\session_esyrceFiltered_z30_epsg23030.pkl'

# load session with data loaded
dill.load_session(inputESYRCE)

# Select columns
selectedCols = ['D1_HUS','D2_NUM','D3_PAR','D4_GRC','D5_CUL','YEA','Shape_Leng','Shape_Area','geometry']
dataSel = data[selectedCols]
dill.dump_session(home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\session_esyrceFiltered_z30_epsg23030_selectedCols.pkl')

