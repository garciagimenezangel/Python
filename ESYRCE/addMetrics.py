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

2) Average cropfield size: average size in every segment of the fields identified as crops

3) Heterogeneity: number of different crop types within a segment, per km^2

4) Demand: demand in a segment, averaged over the area of the polygons
    
5) Soil maintenance (for woody crops, fallows, cereal, sunflower, fodder corn and fodder cereals): 
DE_CS
For woody crops and fallows:
LT    laboreo tradicional
LM    laboreo mínimo
CE    cubiertas vegetales espontáneas
CS    cubiertas vegetales sembradas
CP    cubiertas inertes
SM    sin mantenimiento
NL    no laboreo (caso de cultivos leñosos)
For cereals, sunflower, fodder corn and fodder cereals:
D     siembra directa
N     siembra tradicional

6) Average yield of crops (see variable 'cropCodes') 
    

INPUT: 
    - one shapefile with the ESYRCE data, and csv's: 
    - table with values of crops' demand of pollinators
    - table of classification of ESYRCE codes as crop or not
    - land cover types to measure their percentage
    - soil management techniques
    - sowing technique options
    - crop codes that have available measurements of yield 
    
OUTPUT: csv file with ESYRCE identificator (segment number + year) and the metrics 
"""

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
inputESYRCE = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\z28\\flagged\\data_flag0.shp'
data = gpd.read_file(inputESYRCE)

# OUTPUT
outFilename = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\z30\\metrics\\flag0.csv'
logFile = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\logs\\addMetrics.log'
log = open(logFile, "w")

# Select columns, sort and reset indices
data.Shape_Area = data.geometry.area
data = data[['D1_HUS','D2_NUM','D3_PAR','D4_GRC','D5_CUL','D9_RTO','DE_CS','YEA','Shape_Area']]
data = data.dropna(thresh=1)
data = data.where(data['D1_HUS'] != 0)
data = data.where(data['D2_NUM'] != 0)
data.sort_values(by=['D1_HUS','D2_NUM','YEA'], inplace = True)
data.reset_index(drop=True, inplace=True)

# Define dictionaries:
tableCultivarDemand = home + '\\Google Drive\\PROJECTS\\OBSERV\\Lookup Tables\\ESYRCE\\Cultivar-Demand.csv'
tableIsCrop         = home + '\\Google Drive\\PROJECTS\\OBSERV\\Lookup Tables\\ESYRCE\\isCrop.csv'

# Land cover types (associating to esyrce codes, add more or remove if needed), to calculate proportion in every segment for each year
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

# Soil codes. Apply to land cover types of woody crops and fallow: 
soilCodes = {'traditional':   ['LT'],
             'minimal':       ['LM'],
             'spontVegCover': ['CE'],
             'sowedVegCover': ['CS'],
             'inertCover':    ['CP'],
             'noMainten':     ['SM'],
             'noTillage':     ['NL']}

# Sowing codes. Apply to cereals, sunflower, fodder corn and fodder cereals: 
sowCodes = { 'directSowing':  ['D'],
             'traditSowing':  ['N']} 

# Crop codes. Crops that might have a yield estimation (only some of the fields, have a yield estimation, and theoretically this is done only for these crops, according to the ESYRCE manual 2017)
cropCodes = {'hardWheat':     ['TD'],
             'softWheat':     ['TB'],
             'barleyTwoRows': ['C2'],
             'barleySixRows': ['C6'],
             'oat':           ['AV'],
             'rye':           ['CN'],
             'rice':          ['AR'],
             'maize':         ['MA'],
             'quinoa':        ['QN'],
             'frijol':        ['JS'],
             'fabaBean':      ['HS'],
             'lentil':        ['LE'],
             'chickpea':      ['GA'],
             'pea':           ['GS'],
             'ervil':         ['YE'],
             'potato':        ['PT'],
             'sugarbeet':     ['RM'],
             'cotton':        ['AD'],
             'sunflower':     ['GI'],
             'soybean':       ['SO'],
             'rapeseed':      ['CZ'],
             'industTomato':  ['TI'],
             'fodderMaize':   ['MF'],
             'garlic':        ['AJ'],
             'artichoke':     ['AC'],
             'eggplant':      ['BE'],
             'zucchini':      ['CB'],
             'onion':         ['CL'],
             'strawberry':    ['FN'],
             'greenPea':      ['GV'],
             'greenBean':     ['HV'],
             'kidneyBean':    ['JV'],
             'melon':         ['MO'],
             'cucumber':      ['PI'],
             'sweetPepper':   ['PQ'],
             'watermelon':    ['SA'],
             'carrot':        ['CT'],
             'orange':        ['NR'],
             'clementine':    ['MR'],
             'lemon':         ['LI'],
             'apple':         ['MN'],
             'pear':          ['PE'],
             'apricot':       ['AB'],
             'cherry':        ['CE'],
             'peach':         ['ME'],
             'plum':          ['CR'],
             'banana':        ['PL'],
             'almond':        ['AM'],
             'walnut':        ['AE'],
             'whiteGrapeSeedless':['V1'],
             'whiteGrape':    ['V2'],
             'redGrapeSeedless':['V3'],
             'redGrape':      ['V4'],
             'transfGrape':   ['VT'],
             'oliveTable':    ['OM'],
             'olive':         ['OD'],
             'oliveMill':     ['OT']}

# TODO: ¿cómo hago para luego guardar varianzas del yield tb?

# Init new columns with NaN data
for x in landCoverTypes.keys(): data[x] = np.repeat(np.nan, len(data))
for x in soilCodes.keys():      data[x] = np.repeat(np.nan, len(data))
for x in sowCodes.keys():       data[x] = np.repeat(np.nan, len(data))
for x in cropCodes.keys():      data[x] = np.repeat(np.nan, len(data))
data['avgFieldSize']                    = np.repeat(np.nan, len(data))
data['heterogeneity']                   = np.repeat(np.nan, len(data))
data['demand']                          = np.repeat(np.nan, len(data))

# Define dictionaries from tables
# Dictionary to associate codes with crop category
with open(tableIsCrop, mode='r') as infile:
    reader     = csv.reader(infile)
    dictIsCrop = {rows[0]:rows[1] for rows in reader} # keys: esyrce codes; values: 'YES' or 'NO'


# Dictionary to associate crop codes with demand
with open(tableCultivarDemand, mode='r') as infile:
    reader             = csv.reader(infile)
    dictCultivarDemand = {rows[0]:rows[1] for rows in reader} # key: 'esyrce codes; value: demand estimation (see dictDemandValues defined at the beginning of this file)


##################
# Loop zones
zoneNrs = np.unique(data.D1_HUS)
for zoneNr in zoneNrs:
    # Select zone data 
    ii = np.where(data.D1_HUS == zoneNr) 
    i0 = ii[0][0]
    iM = ii[0][len(ii[0])-1]
    dataZoneNr = data[i0:(iM+1)]
    if (iM-i0+1)!=len(ii[0]): # sanity check
        log.write("Error... Exit loop in zone nr:", zoneNr) # sanity check
        break
    
    # Loop plot numbers
    segmentNrs = np.unique(dataZoneNr.D2_NUM)
    totalNr = len(segmentNrs) 
    contNr = 0
    for segmentNr in segmentNrs:
    
        # Get the rows corresponding to segmentNr. This is not the safest way (it doesn't work if the rows are not sorted out by number of segment, which seems not to be the case), 
        # but it is the fastest way. A sanity check is included to make sure that this way works reasonably well. If the "Error..." message is prompted too much times, then 
        # this way of getting the rows for the segmentNr must be replaced by the slow way (see filterDataByExtent.py as an example)
        ii = np.where(dataZoneNr.D2_NUM == segmentNr) 
        i0 = ii[0][0]
        iM = ii[0][len(ii[0])-1]
        dataSegmentNr = dataZoneNr[i0:(iM+1)]
        if (iM-i0+1)!=len(ii[0]): # sanity check
            log.write("Error... Exit loop in Segment nr:", segmentNr) # sanity check
            break
    
        years = np.unique(dataSegmentNr.YEA)
        for year in years:
        
            # Get the rows corresponding to a particular year. As above-mentioned, not the safest way, but the fastest (to my knowledge), so a sanity check is needed.
            ii = np.where(dataSegmentNr.YEA == year)
            i0 = ii[0][0]
            iM = ii[0][len(ii[0])-1]
            dataSegmentYear = dataSegmentNr[i0:(iM+1)]
            if (iM-i0+1)!=len(ii[0]): # sanity check
                log.write("Error... Exit loop in Segment nr:", segmentNr, "...Year:",year)  
                break
        
            # Calculate metrics
            landCoverProportion = functions.calculateLandCoverProportion(dataSegmentYear, landCoverTypes, alternatCodes, log)
            soilTechnProportion = functions.calculateSoilTechniqueProportion(dataSegmentYear, soilCodes, sowCodes, log) 
            sowTechnProportion  = functions.calculateSoilTechniqueProportion(dataSegmentYear, sowCodes, soilCodes, log) 
            cropYield           = functions.calculateCropYield(dataSegmentYear, cropCodes, log)
            avgFieldSize        = functions.calculateAvgFieldSize(dataSegmentYear, dictIsCrop, log)
            heterogeneity       = functions.calculateHeterogeneity(dataSegmentYear, dictIsCrop, log)
            demand              = functions.calculateDemand(dataSegmentYear, dictCultivarDemand, log)
        
            # Assign values
            for x in landCoverTypes.keys(): data.loc[dataSegmentYear.index, x] = np.repeat(landCoverProportion[x], len(dataSegmentYear))              
            for x in soilCodes.keys():      data.loc[dataSegmentYear.index, x] = np.repeat(soilTechnProportion[x], len(dataSegmentYear))
            for x in sowCodes.keys():       data.loc[dataSegmentYear.index, x] = np.repeat(sowTechnProportion[x], len(dataSegmentYear))
            for x in cropYield.keys():      data.loc[dataSegmentYear.index, x] = np.repeat(cropYield[x], len(dataSegmentYear))
            data.loc[dataSegmentYear.index, 'avgFieldSize']                    = np.repeat(avgFieldSize, len(dataSegmentYear))
            data.loc[dataSegmentYear.index, 'heterogeneity']                   = np.repeat(heterogeneity, len(dataSegmentYear))
            data.loc[dataSegmentYear.index, 'demand']                          = np.repeat(demand, len(dataSegmentYear))
    
        contNr = contNr+1
        if np.mod(contNr, 100) == 0:
            times = contNr / totalNr 
            log.write("Processing data Zone...", int(zoneNr), "Percentage completed...", np.floor(times*100))

# Group by number of the segment and year, drop not useful columns, and save to csv
data = data.drop(columns=['D3_PAR','D4_GRC','D5_CUL','D9_RTO','DE_CS','Shape_Area'])
data = data.groupby(['D1_HUS','D2_NUM','YEA']).first().reset_index()
log.write("Writing file...", outFilename)
data.to_csv(outFilename, index=False)
log.write("FINISHED... Data saved... " + outFilename)
log.close()