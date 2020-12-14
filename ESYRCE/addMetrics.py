# -*- coding: utf-8 -*-
import csv
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime
from os.path import expanduser
import glob
home = expanduser("~")

# The functions used to calculate the metrics are stored in a different file, to make this script cleaner 
import sys
#sys.path.append(home + '\\git\\Python\\ESYRCE\\')
sys.path.append(home + '/git/Python/ESYRCE/')
import functions

"""
Calculate metrics from ESYRCE data.

INPUT: 
    - folder with shapefiles with the ESYRCE data, and csv's: 
    - table with values of crops' demand of pollinators
    - table of classification of ESYRCE codes as crop or not
    - table of classification of ESYRCE codes as seminatural area or not
    - land cover types to measure their percentage
    - soil management techniques
    - sowing technique options
    - crop codes that have available measurements of yield 
    
OUTPUT: csv file with ESYRCE identificator (segment number + year) and the metrics 
"""

##############
## SETTINGS ##
##############
# Metrics available
getLandCoverProportion     = True # Percentage of the land cover types (see variable 'landCoverTypes' below)
getSoilTechniqueProportion = True # Soil maintenance technique proportion (see variable 'soilCodes' below)
getSowTechniqueProportion  = True # Sowing technique proportion (direct or traditional)
getCropYield               = True # Average and variance of the yield of each crop within the segments (see variable 'cropCodes' below) 
getAvgFieldSize            = True  # Average size of the fields identified as crops (in the table 'tableIsCrop' below) 
getAvgSeminatSize          = True  # Average size of the fields identified as seminatural area (in the table 'tableIsSeminatural' below) 
getHeterogeneity           = True  # Heterogeneity, as number of crop types per unit area
getDemand                  = True  # Average demand, weighted by the area of the polygons 
getSegmentArea             = True  # Total area of the segments
getSegmentAreaWithoutWater = True  # Area of the segments, ignoring water 
getEdgeDensity             = True  # Density of edges (length/area)
getEdgeDensitySeminatural  = True  # Density of edges from seminatural area (length/area)
getEdgeDensityCropfields   = True  # Density of edges from crop fields (length/area)

# Final output
finalFilename = "test2"

# INPUT folder
#inputESYRCE = home + '\\DATA\\ESYRCE\\PROCESSED - local testing\\z30\\flagged\\test2\\'
inputESYRCE = home + '/DATA/OBServ/ESYRCE/PROCESSED/z30/flagged/'

# OUTPUT folder
#outFolder = home + '\\DATA\\ESYRCE\\PROCESSED - local testing\\z30\\metrics\\test2\\'
outFolder = home + '/DATA/OBServ/ESYRCE/PROCESSED - local testing/z30/metrics/'

# Log file
#logFile = home + '\\DATA\\ESYRCE\\PROCESSED - local testing\\logs\\addMetrics.log'
logFile = home + '/DATA/OBServ/ESYRCE/PROCESSED/logs/addMetrics.log'
buffSize = 1
log = open(logFile, "a", buffering=buffSize)
log.write("\n")
log.write("PROCESS addMetrics.py STARTED AT: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'\n')

# Define dictionaries:
#tableCultivarDemand = 'G:\\My Drive\\PROJECTS\\OBSERV\\Lookup Tables\\ESYRCE\\Cultivar-Demand.csv'
#tableIsCrop         = 'G:\\My Drive\\PROJECTS\\OBSERV\\Lookup Tables\\ESYRCE\\isCrop.csv'
#tableIsSeminatural  = 'G:\\My Drive\\PROJECTS\\OBSERV\\Lookup Tables\\ESYRCE\\isSeminatural.csv'
tableCultivarDemand = home + '/lookup/Cultivar-Demand.csv'
tableIsCrop         = home + '/lookup/isCrop.csv'
tableIsSeminatural  = home + '/lookup/isSeminatural.csv'

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
                  'oliveTrees':         ['OL','*'],  
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

# Define dictionaries from tables
# Dictionary to associate codes with crop category
with open(tableIsCrop, mode='r') as infile:
    reader     = csv.reader(infile)
    dictIsCrop = {rows[0]:rows[1] for rows in reader} # keys: esyrce codes; values: 'YES' or 'NO'
    
# Dictionary to associate codes with seminatural category
with open(tableIsSeminatural, mode='r') as infile:
    reader     = csv.reader(infile)
    dictIsSeminatural = {rows[0]:rows[1] for rows in reader} # keys: esyrce codes; values: 'YES' or 'NO'

# Dictionary to associate crop codes with demand
# In ubuntu you may need to change encoding of this file: iconv -f ISO-8859-1 -t utf-8 Cultivar-Demand.csv > Cultivar-Demand-utf8.csv
with open(tableCultivarDemand, mode='r') as infile:
    reader   = csv.reader(infile)       
    dictCultivarDemand = {rows[0]:rows[1] for rows in reader} # key: 'esyrce codes; value: demand estimation (see dictDemandValues defined at the beginning of this file)


##################
# Loop files
for file in glob.glob(inputESYRCE + "*.shp"):
    
    # OUTPUT
    filename = file.split(inputESYRCE)[1]
    filename = filename.split(".shp")[0]
    outFilename = outFolder+filename+".csv"

    log.write("Processing file..."+filename+'\n')
    
    # Read data
    data = gpd.read_file(file)
    
    # Modify or create useful columns
    data.Shape_Area = data.geometry.area
    data.Shape_Leng = data.geometry.length
    data['isSeminatural'] = [functions.isSeminatural(i, dictIsSeminatural) for i in data.D5_CUL] 
    data['isCropfield']   = [functions.isCropfield(i, dictIsCrop) for i in data.D4_GRC] 

    # Select columns, sort and reset indices
    data = data[['D1_HUS','D2_NUM','D3_PAR','D4_GRC','D5_CUL','D9_RTO','DE_CS','YEA','Shape_Area','Shape_Leng','isSeminatural','isCropfield']]
    data = data.dropna(thresh=1)
    data = data.where(data['D1_HUS'] != 0)
    data = data.where(data['D2_NUM'] != 0)
    data.sort_values(by=['D1_HUS','D2_NUM','YEA'], inplace = True)
    data.reset_index(drop=True, inplace=True)
    
    # Init new columns with NaN data
    if getLandCoverProportion:     
        for x in landCoverTypes.keys(): data[x] = np.repeat(np.nan, len(data))
    if getSoilTechniqueProportion: 
        for x in soilCodes.keys():      data[x] = np.repeat(np.nan, len(data))
    if getSowTechniqueProportion:  
        for x in sowCodes.keys():       data[x] = np.repeat(np.nan, len(data))
    if getCropYield:               
        for x in cropCodes.keys():      
            data[x] = np.repeat(np.nan, len(data))
            data['var_'+x] = np.repeat(np.nan, len(data)) 
    if getAvgFieldSize:                 data['avgFieldSize']   = np.repeat(np.nan, len(data))
    if getAvgSeminatSize:               data['avgSeminatSize'] = np.repeat(np.nan, len(data))
    if getHeterogeneity:                data['heterogeneity']  = np.repeat(np.nan, len(data))
    if getDemand:                       data['demand']         = np.repeat(np.nan, len(data))    
    if getSegmentArea:                  data['segArea']        = np.repeat(np.nan, len(data))
    if getSegmentAreaWithoutWater:      data['segAreaNoWater'] = np.repeat(np.nan, len(data))
    if getEdgeDensity:                  data['edgeDensity']    = np.repeat(np.nan, len(data))
    if getEdgeDensitySeminatural:       data['edgeDenSeminat'] = np.repeat(np.nan, len(data))
    if getEdgeDensityCropfields:        data['edgeDenFields']  = np.repeat(np.nan, len(data))

    
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
            log.write("Error... Exit loop in zone nr:"+str(zoneNr)+'\n') # sanity check
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
                log.write("Error... Exit loop in Segment nr:"+str(segmentNr)+'\n') # sanity check
                break
        
            years = np.unique(dataSegmentNr.YEA)
            for year in years:
            
                # Get the rows corresponding to a particular year. As above-mentioned, not the safest way, but the fastest (to my knowledge), so a sanity check is needed.
                ii = np.where(dataSegmentNr.YEA == year)
                i0 = ii[0][0]
                iM = ii[0][len(ii[0])-1]
                dataSegmentYear = dataSegmentNr[i0:(iM+1)]
                if (iM-i0+1)!=len(ii[0]): # sanity check
                    log.write("Error... Exit loop in Segment nr:"+ str(segmentNr)+ "...Year:"+str(year)+'\n')  
                    break
            
                # Calculate metrics and assign values in 'data'
                if getLandCoverProportion:     
                    landCoverProportion = functions.calculateLandCoverProportion(dataSegmentYear, landCoverTypes, alternatCodes, log)
                    for x in landCoverTypes.keys(): data.loc[dataSegmentYear.index, x] = np.repeat(landCoverProportion[x], len(dataSegmentYear)) 
                if getSoilTechniqueProportion: 
                    soilTechnProportion = functions.calculateSoilTechniqueProportion(dataSegmentYear, soilCodes, sowCodes, log) 
                    for x in soilCodes.keys():      data.loc[dataSegmentYear.index, x] = np.repeat(soilTechnProportion[x], len(dataSegmentYear))
                if getSowTechniqueProportion:  
                    sowTechnProportion  = functions.calculateSoilTechniqueProportion(dataSegmentYear, sowCodes, soilCodes, log) 
                    for x in sowCodes.keys():       data.loc[dataSegmentYear.index, x] = np.repeat(sowTechnProportion[x], len(dataSegmentYear))
                if getCropYield:               
                    cropYield           = functions.calculateCropYield(dataSegmentYear, cropCodes, log)
                    varYield            = functions.calculateVarianceYield(dataSegmentYear, cropCodes, cropYield, log)
                    for x in cropYield.keys():      data.loc[dataSegmentYear.index, x] = np.repeat(cropYield[x], len(dataSegmentYear))
                    for x in cropYield.keys():      data.loc[dataSegmentYear.index, 'var_'+x] = np.repeat(varYield[x], len(dataSegmentYear))                    
                if getAvgFieldSize:                 
                    avgFieldSize        = functions.calculateAvgFieldSize(dataSegmentYear, log)
                    data.loc[dataSegmentYear.index, 'avgFieldSize'] = np.repeat(avgFieldSize, len(dataSegmentYear))
                if getAvgSeminatSize:               
                    avgSeminatSize      = functions.calculateAvgSeminaturalSize(dataSegmentYear, log)
                    data.loc[dataSegmentYear.index, 'avgSeminatSize'] = np.repeat(avgSeminatSize, len(dataSegmentYear))
                if getHeterogeneity:                
                    heterogeneity       = functions.calculateHeterogeneity(dataSegmentYear, log)
                    data.loc[dataSegmentYear.index, 'heterogeneity'] = np.repeat(heterogeneity, len(dataSegmentYear))
                if getDemand: 
                    demand              = functions.calculateDemand(dataSegmentYear, dictCultivarDemand, log)
                    data.loc[dataSegmentYear.index, 'demand'] = np.repeat(demand, len(dataSegmentYear))
                if getSegmentArea:                  
                    segmentArea         = functions.calculateSegmentArea(dataSegmentYear)
                    data.loc[dataSegmentYear.index, 'segArea'] = np.repeat(segmentArea, len(dataSegmentYear))
                if getSegmentAreaWithoutWater:     
                    segmentAreaNoWater  = functions.calculateSegmentAreaWithoutWater(dataSegmentYear, log)
                    data.loc[dataSegmentYear.index, 'segAreaNoWater'] = np.repeat(segmentAreaNoWater, len(dataSegmentYear))
                if getEdgeDensity:                 
                    edgeDensity         = functions.calculateEdgeDensity(dataSegmentYear, log)
                    data.loc[dataSegmentYear.index, 'edgeDensity'] = np.repeat(edgeDensity, len(dataSegmentYear))
                if getEdgeDensitySeminatural:                 
                    edgeDenSeminat      = functions.calculateEdgeDensitySeminatural(dataSegmentYear, log)
                    data.loc[dataSegmentYear.index, 'edgeDenSeminat'] = np.repeat(edgeDenSeminat, len(dataSegmentYear))
                if getEdgeDensityCropfields:                 
                    edgeDenFields      = functions.calculateEdgeDensityFields(dataSegmentYear, log)
                    data.loc[dataSegmentYear.index, 'edgeDenFields'] = np.repeat(edgeDenFields, len(dataSegmentYear))


            contNr = contNr+1
            if np.mod(contNr, 100) == 0:
                times = contNr / totalNr 
                log.write("Processing File..."+filename+" Data Zone..."+str(int(zoneNr))+" Percentage completed..."+str(np.floor(times*100))+'\n')
    
    # Group by number of the segment and year, drop not useful columns, and save to csv
    data = data.drop(columns=['D3_PAR','D4_GRC','D5_CUL','D9_RTO','DE_CS','Shape_Area','Shape_Leng','isSeminatural','isCropfield'])
    data = data.groupby(['D1_HUS','D2_NUM','YEA']).first().reset_index()
    log.write("Writing file..."+outFilename+'\n')
    data.to_csv(outFilename, index=False)
    log.write("Data saved... " + outFilename+'\n')

# Merge csv's
outFilename  = outFolder+finalFilename+".csv"
csvs         = [i for i in glob.glob(outFolder + "*.csv")]
combined_csv = pd.concat([pd.read_csv(f) for f in csvs ])
combined_csv.to_csv( outFilename, index=False)
log.write("Data saved... " + outFilename+'\n')
    
##################
log.close()
