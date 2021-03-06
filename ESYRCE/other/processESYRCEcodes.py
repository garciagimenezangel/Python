# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 15:29:30 2020

@author: angel.gimenez
"""


# -*- coding: utf-8 -*-
"""
The goal of this script is to process the column D5_CUL in ESYRCE datasets, to keep
all the relevant information that we will need in the near future to calculate
 pollinators' supply and demand, and agricultural intensification metrics.

Why is that needed? D5_CUL includes complex ways of encoding information that 
are difficult to diggest by k.LAB (or any other modeling environment). For example, 
complementary codes are added, sometimes, at the end of the code; association of 
cultivars are indicated using a hyphen as separator (e.g. "BT-CL-CW-TB").

INPUT
The input of the script is:
    1) a shapefile with a column D5_CUL (among others) associated with areas 
    surveyed in ESYRCE, where land cover type is stored using the ESYRCE encoding

OUTPUT
The original dataset + 2 new columns with the following information: 
    1) simple land cover codes, named 'detailcode'
    2) complementary codes, named 'compcode'

PROCESSING
When reading the ESYRCE codes in D5_CUL, these are the following possibilities:
    1) Simple codes (e.g. "TD"). They are saved in the column 'detailcode' 
    as they are. 'compcode' remains empty. 
    2) Strings with simple and complementary codes (e.g. "BTI": "BT" + "I"). The
    first two characters are saved in 'detailcode' and the 3rd is saved in 
    'compcode'. 
    3) Combination of codes (e.g. "MN-LI" or "MN1-LI1"). In 'detailcode',
    it is saved the first cultivar involved ("MN" in the previous examples).
    If every individual code includes the same complementary code (e.g. "1" in 
    "MN1-LI1"), it is saved in the column 'compcode'. 
"""

import geopandas as gpd
import numpy as np
import csv
from os.path import expanduser
home = expanduser("~")

# INPUT
layer = 'z28'
inputESYRCE = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\esyrceFiltered_' + layer + '.shp'
cropCODES   = home + '\\Google Drive\\PROJECTS\\OBSERV\\Lookup Tables\\cropCODES.csv'
        
# OUTPUT
processedFile = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\esyrceProcessed_' + layer + '.shp'

# load file from local path
data = gpd.read_file(inputESYRCE)

# Read crop codes
with open(cropCODES, mode='r') as infile:
    reader    = csv.reader(infile)
    dictCodes = {rows[0]:rows[1] for rows in reader}

if layer == 'z28':
    crs = "EPSG:32628"

if layer == 'z30':
    crs = "EPSG:32630"
    
# 3 new columns
data['detailcode'] = np.NaN
data['compcode'] = ""

for index in data.index:
    code = data.loc[index].D5_CUL
    if "-" in code:
        assocElts = code.split("-")
        lenElts = [len(elt) for elt in assocElts]
        
        # Save first element
        try:
            dictCode = dictCodes[assocElts[0]]
        except:
            dictCode = 0
        data.at[index, 'detailcode'] = dictCode
                
        # Check if the elements have a complementary code == 3 characters
        if all(np.isclose(lenElts,3)):
            thirdElts = [i[2] for i in assocElts] # Get third character
            if all(x==thirdElts[0] for x in thirdElts): # make sure that the complementary code is the same in every element
                # Save the complementary code
                data.at[index, 'compcode'] = thirdElts[0]
    else:
        try:
            dictCode = dictCodes[code[0:2]]
        except:
            dictCode = 0
        data.at[index, 'detailcode'] = dictCode
        if len(code) == 3: # check whether it has a complementary code
            data.at[index, 'compcode'] = code[2]

data.crs = crs
data.to_file(filename=processedFile, driver="ESRI Shapefile")


