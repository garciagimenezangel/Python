# -*- coding: utf-8 -*-
"""
Calculate metrics from ESYRCE data:
1) Percentage of the following land cover types:
D4_GRC	D5_CUL	descripción
CE	    *	    cereal grano
LE	    *	    leguminosas grano
TU	    *	    tubérculos consumo humano
IN	    *	    industriales
FO	    *	    forrajeras
HO	    *	    hortalizas
FL	    *	    flores y hornamentales
TC	    *	    invernaderos vacíos
CI	    *	    cítricos
FR	    *	    frutales no cítricos
VI	    *	    viñedo
OL	    *	    olivar
OC	    *	    otros cultivos leñosos
VV	    *	    viveros
AS	    *	    asociaciones de cultivo
TC	    Ba	    barbecho
TC	    Hu	    huerto familiar
PP	    PR	    prado natural
PP	    PH	    prado de alta montaña
PP	    PS	    pastizal
PP	    PM	    pastizal matorral
SF	    CO	    coníferas
SF	    FL	    frondosas de crecimiento lento
SF	    CP	    chopo
SF	    FR	    frondosas de crecimiento rápido
SF	    CF	    coníferas y frondosas
SF	    ML	    matorral
OS	    ER	    erial
OS	    ES	    espartizal
OS	    BL	    baldío
IM	    *	    rocas, pedregales, graveras, arenales, dunas, playas, torrenteras, nieves perpetuas, etc.
NA	    *	    producción de energía, industria, transportes, almacenamiento, construcción, zonas de ocio, edificaciones
AG	    *	    aguas interiores
MO	    * 	    mar, lagunas litorales, estuarios, etc.

2) Average cropfield size: average size in every block of the fields identified as crops

3) Heterogeneity: number of different crop types within a block, per km^2

4) Demand: demand in a block, averaged over the area of the polygons

INPUT: 
    - one shapefile with the ESYRCE data, and csv's: 
    - table with values of crops' demand of pollinators
    - table of classification of ESYRCE codes as crop or not
    - land cover types to measure their percentage
    
OUTPUT: csv file with the ESYRCE data and the new metrics added as columns
"""

import dill
import csv
import geopandas as gpd
import numpy as np
from os.path import expanduser
home = expanduser("~")

# The functions used to calculate the metrics are stored in a different file, to make this script cleaner 
import sys
sys.path.append(home + '\\Documents\\REPOSITORIES\\Python\\ESYRCE\\')
import functions

# INPUT
#import dill
#session = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\z30\\sessions\\dataSel.pkl'
#dill.load_session(session) # data in dataSel
inputESYRCE = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\z30\\filtered\\merged.shp'
data = gpd.read_file(inputESYRCE)
tableCultivarDemand = home + '\\Google Drive\\PROJECTS\\OBSERV\\Lookup Tables\\ESYRCE\\Cultivar-Demand.csv'
tableIsCrop         = home + '\\Google Drive\\PROJECTS\\OBSERV\\Lookup Tables\\ESYRCE\\isCrop.csv'

# Land cover types (associating to esyrce codes, add more or remove if needed), to calculate proportion in every block for each year
landCoverTypes = {'cerealGrain':        ['CE','*'],     
                  'legumeGrain':        ['LE','*'],    
                  'tuber':              ['TU','*'],  
                  'industrial':         ['IN','*'],     
                  'fodder':             ['FO','*'],     
                  'vegetable':          ['HO','*'],     
                  'ornamental':         ['FL','*'],     
                  'citric':             ['CI','*'],     
                  'fruitNoCitric':      ['FR','*'],   
                  'vineyard':           ['VI','*'],  
                  'olive':              ['OL','*'],  
                  'otherWoodyCrop':     ['OC','*'],  
                  'nursery':            ['VV','*'],  
                  'association':        ['AS','*'],  
                  'fallow':             ['TC','BA'],  
                  'emptyGreenh':        ['TC','VA'],     
                  'orchard':            ['TC','HU'],  
                  'posio':              ['TC','PO'],  
                  'naturalMeadow':      ['PP','PR'],  
                  'highMountainMeadow': ['PP','PH'],
                  'pastureGrassland':   ['PP','PS'],
                  'pastureShrub':       ['PP','PM'],
                  'conifers':           ['SF','CO'],
                  'broadleafFast':      ['SF','FR'],
                  'broadleafSlow':      ['SF','FL'],
                  'poplar':             ['SF','CP'],
                  'mixedForest':        ['SF','CF'],
                  'shrub':              ['SF','ML'],
                  'wasteland':          ['OS','ER'],
                  'spartizal':          ['OS','ES'],
                  'abandoned':          ['OS','BL'],
                  'improductive':       ['OS','IM'],
                  'notAgri':            ['OS','NA']
                 }                           

# Some land cover types can be identified with other combination of codes (lamentable pero cierto). 
# I define another dictionary to deal with this issue
alternatCodes = {'improductive': ['IM','*'], 
                 'notAgri':      ['NA','*']}

# OUTPUT
outFilename = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\z30\\filtered\\metrics.shp'

# Backup session
backupSession = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\z30\\sessions\\backup.pkl'

# Init new columns
for x in landCoverTypes.keys(): 
    data[x] = np.nan
data['avgFieldSize'] = np.nan
data['heterogeneity'] = np.nan
data['demand'] = np.nan

# Define dictionaries from tables
# Dictionary to associate codes with crop category
with open(tableIsCrop, mode='r') as infile:
    reader     = csv.reader(infile)
    dictIsCrop = {rows[0]:rows[1] for rows in reader} # keys: esyrce codes; values: 'YES' or 'NO'
    
# Dictionary to associate crop codes with demand
with open(tableCultivarDemand, mode='r') as infile:
    reader             = csv.reader(infile)
    dictCultivarDemand = {rows[0]:rows[1] for rows in reader} # key: 'esyrce codes; value: demand estimation (see dictDemandValues defined at the beginning of this file)

# Loop plot numbers
blockNrs = np.unique(data.D2_NUM)
totalNr = len(blockNrs) 
contNr = 0
for blockNr in blockNrs:
    
    # Get the rows corresponding to blockNr. This is not the safest way (it doesn't work if the rows are not sorted out by number of block, which seems not to be the case), 
    # but it is the fastest way. A sanity check is included to make sure that this way works reasonably well. If the "Error..." message is prompted too much times, then 
    # this way of getting the rows for the blockNr must be replaced by the slow way (see filterDataByExtent.py as an example)
    ii = np.where(data.D2_NUM == blockNr) 
    i0 = ii[0][0]
    iM = ii[0][len(ii[0])-1]
    dataBlockNr = data[i0:(iM+1)]
    if (iM-i0+1)!=len(ii[0]): # sanity check
        print("Error... Exit loop in Block nr:",blockNr) # sanity check
        break
    
    years = np.unique(dataBlockNr.YEA)
    for year in years:
        
        # Get the rows corresponding to a particular year. As above-mentioned, not the safest way, but the fastest (to my knowledge), so a sanity check is needed.
        ii = np.where(dataBlockNr.YEA == year)
        i0 = ii[0][0]
        iM = ii[0][len(ii[0])-1]
        dataBlockYear = dataBlockNr[i0:(iM+1)]
        if (iM-i0+1)!=len(ii[0]): # sanity check
            print("Error... Exit loop in Block nr:",blockNr,"...Year:",year)  
            break
    
        # Calculate metrics
        landCoverProportion = functions.calculateLandCoverProportion(dataBlockYear, landCoverTypes, alternatCodes)
        avgFieldSize        = functions.calculateAvgFieldSize(dataBlockYear, dictIsCrop)
        heterogeneity       = functions.calculateHeterogeneity(dataBlockYear, dictIsCrop)
        demand              = functions.calculateDemand(dataBlockYear, dictCultivarDemand)
        
        # Assign values
        for x in landCoverTypes.keys(): 
            data[x].loc[dataBlockYear.index] = landCoverProportion[x]
        data.avgFieldSize.loc[dataBlockYear.index]  = avgFieldSize
        data.heterogeneity.loc[dataBlockYear.index] = heterogeneity
        data.demand.loc[dataBlockYear.index]        = demand
        
    contNr = contNr+1
    if np.mod(contNr, 100) == 0:
        times = contNr / totalNr 
        print("Processing data...", np.floor(times*100), "percent completed...")
    
    if np.mod(contNr, 3000) == 0:
        times = contNr / totalNr 
        dill.dump_session(backupSession)
        print("Saved session... " + backupSession)

data.to_file(filename = outFilename, driver="ESRI Shapefile")
print("FINISHED... Data saved... " + outFilename)


