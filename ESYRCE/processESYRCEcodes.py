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
    2) a CSV lookup table with an associated value of demand [0-1] for every crop

OUTPUT
The original dataset + 3 new columns with the following information: 
    1) simple land cover codes, named 'processed_code'
    2) complementary codes, named 'complementary_code'
    3) pollinators' demand, named 'pollinators_demand'

PROCESSING
When reading the ESYRCE codes in D5_CUL, these are the following possibilities:
    1) Simple codes (e.g. "TD"). They are saved in the column 'processed_code' 
    as they are. 'complementary_code' remains empty. 'pollinators_demand' is 
    filled using the CSV lookup table.
    2) Strings with simple and complementary codes (e.g. "BTI": "BT" + "I"). The
    first two characters are saved in 'processed_code' and the 3rd is saved in 
    'complementary_code'. 'pollinators_demand' is filled using the CSV lookup table
    for the extracted simple code.
    3) Combination of codes (e.g. "MN-LI" or "MN1-LI1"). In 'processed_code',
    it is saved "C" + number of cultivars involved ("C2" in the previous examples).
    If every individual code includes the same complementary code (e.g. "1" in 
    "MN1-LI1"), it is saved in the column 'complementary_code'. For the pollinators'
    demand, the average demand for all the cultivars involved is calculated using
    the lookup table, and the result is saved in 'pollinators_demand'.
"""

import geopandas as gpd
import numpy as np

# INPUT
inputESYRCE = '..\\..\\..\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\validData.shp'
#inputDemand = 

# OUTPUT
processedFile = '..\\..\\..\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\processedData.shp'

# load file from local path
data = gpd.read_file(inputESYRCE)

# Two new columns
data['code'] = ""
data['complementary_code'] = ""

for index in data.index:
    code = data.loc[index].D5_CUL
    if "-" in code:
        assocElts = code.split("-")
        lenElts = [len(elt) for elt in assocElts]
        
        # Save with the CODE: C + #ELEMENTS
        data.at[index, 'code'] = "C"+str(len(assocElts))
                
        # Check if the elements have a complementary code == 3 characters
        if all(np.isclose(lenElts,3)):
            thirdElts = [i[2] for i in assocElts] # Get third character
            if all(x==thirdElts[0] for x in thirdElts): # make sure that the complementary code is the same in every element
                # Save the complementary code
                data.at[index, 'complementary_code'] = thirdElts[0]
    else:
        data.at[index, 'code'] = code[0:2]
        if len(code) == 3: # check whether it has a complementary code
            data.at[index, 'complementary_code'] = code[2]
                
data.to_file(filename=processedFile, driver="ESRI Shapefile")


