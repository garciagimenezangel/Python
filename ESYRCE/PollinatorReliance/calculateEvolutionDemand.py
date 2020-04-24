# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:37:49 2020

@author: angel.gimenez
"""
import geopandas as gpd
import pandas as pd
import numpy as np
import blockCalculator as bc
from os.path import expanduser
home = expanduser("~")

# INPUT
inputESYRCE = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\esyrceFiltered_z30_merged.shp'
    
# load file from local path
data = gpd.read_file(inputESYRCE)

colsDemandByYear =  ['blockNr', 'year', 'demand']
colsDemandDiff   =  ['blockNr', 'demandDiff']
DfDemandByYear   = pd.DataFrame(columns = colsDemandByYear)
DfDemandDiff     = pd.DataFrame(columns = colsDemandDiff)

# Loop plot numbers
blockNrs = np.unique(data.D2_NUM)
for blockNr in blockNrs:
    
    selectedInd = data.D2_NUM == blockNr
    dataBlockNr = [data.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
    dataBlockNr = gpd.GeoDataFrame(dataBlockNr)
    
    # Create dataframes with the initial and end years
    years   = np.unique(dataBlockNr.YEA)
    yearIni = years[0]
    yearEnd = years[len(years)-1]    
    if yearIni == yearEnd: continue
    selectedIni  = dataBlockNr.YEA == yearIni
    selectedEnd  = dataBlockNr.YEA == yearEnd
    dataBlockIni = [dataBlockNr.iloc[i] for i in range(0,len(selectedIni)) if selectedIni.iloc[i]]    
    dataBlockEnd = [dataBlockNr.iloc[i] for i in range(0,len(selectedEnd)) if selectedEnd.iloc[i]]
    dataBlockIni = gpd.GeoDataFrame(dataBlockIni)
    dataBlockEnd = gpd.GeoDataFrame(dataBlockEnd)
    
    # Calculate pollinators' demand in the initial and end years
    demandIni = bc.calculateDemand(dataBlockIni)
    demandEnd = bc.calculateDemand(dataBlockEnd)
    demandDiff = (demandEnd - demandIni) / (yearEnd - yearIni)
    
    # Save data in dataframes
    DfDemandByYear.loc[len(DfDemandByYear)] = [blockNr, yearIni, demandIni]
    DfDemandByYear.loc[len(DfDemandByYear)] = [blockNr, yearEnd, demandEnd]
    DfDemandDiff.loc[len(DfDemandDiff)]     = [blockNr, demandDiff]
    
    