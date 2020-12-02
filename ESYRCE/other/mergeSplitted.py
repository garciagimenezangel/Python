# -*- coding: utf-8 -*-
"""
INPUT: root folder with shapefiles (for example, output files from script 'flagDataSegments')
OUTPUT: one shapefile merging all the input files
"""
import pandas as pd
import numpy as np
import glob
from os.path import expanduser
home = expanduser("~")

# INPUT
root = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\z30\\metrics\\demand\\'

# OUTPUT
outFilename = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\z30\\metrics\\demand\\demandCorrected_2001-2019.csv'

# Get filtered files
listFiles = glob.glob(root+"data*.csv")

# Concat dataframes
#data = gpd.read_file(listFiles[0])

frames = []
for file in listFiles:
    #data = gpd.read_file(file)
    data = pd.read_csv(file)
#    x = file.split("data_flag")[1]
#    flag = x.split("_")[0]
#    data['flag'] = np.repeat(flag, len(data))
    frames.append(data)
    print("Read file:", file)
result = pd.concat(frames)

# To file 
#result.to_file(filename = outFilename, driver="ESRI Shapefile")
result.to_csv(outFilename, index=False)
print("Saved file:", outFilename)

