# -*- coding: utf-8 -*-
import csv
import geopandas as gpd
import pandas as pd
import numpy as np
from datetime import datetime
from os.path import expanduser
import glob
import sys
home = expanduser("~")

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
getAvgFieldSize            = True # Average size of the fields identified as crops (in the table 'tableIsCrop' below) 
getAvgSeminatSize          = True # Average size of the fields identified as seminatural area (in the table 'tableIsSeminatural' below) 
getAvgFieldSizeDiss        = True # Average size of the fields identified as crops (in the table 'tableIsCrop' below) dissolving by 'isCropfield' and 'isSeminatural'
getAvgSeminatSizeDiss      = True # Average size of the fields identified as seminatural area (in the table 'tableIsSeminatural' below) dissolving by 'isCropfield' and 'isSeminatural'
getHeterogeneity           = True # Heterogeneity, as number of crop types per unit area
getDemand                  = True # Average demand, weighted by the area of the polygons 
getSegmentArea             = True # Total area of the segments
getSegmentAreaWithoutWater = True # Area of the segments, ignoring water 
getEdgeDensity             = True # Density of edges (length/area)
getEdgeDensitySeminatural  = True # Density of edges from seminatural area (length/area)
getEdgeDensityCropfields   = True # Density of edges from crop fields (length/area)
getEdgeDensDissolved       = True # Density of edges (total) dissolving by 'isCropfield' and 'isSeminatural'
getEdgeDensitySeminatDiss  = True # Density of edges (seminatural) dissolving by 'isCropfield' and 'isSeminatural'
getEdgeDensityCropDiss     = True # Density of edges (cropfields) dissolving by 'isCropfield' and 'isSeminatural'
getEdgeDensityOtherDiss    = True # Density of edges (others) dissolving by 'isCropfield' and 'isSeminatural'


# Final output
finalFilename = "test"

# Paths
#inputESYRCE         = home + '\\DATA\\ESYRCE\\PROCESSED - local testing\\z30\\flagged\\test2\\'
#outFolder           = home + '\\DATA\\ESYRCE\\PROCESSED - local testing\\z30\\metrics\\test2\\'
#logFile             = home + '\\DATA\\ESYRCE\\PROCESSED - local testing\\logs\\addMetrics.log'
#tableCultivarDemand = 'G:\\My Drive\\PROJECTS\\OBSERV\\Lookup Tables\\ESYRCE\\Cultivar-Demand.csv'
#tableIsCrop         = 'G:\\My Drive\\PROJECTS\\OBSERV\\Lookup Tables\\ESYRCE\\isCrop.csv'
#tableIsSeminatural  = 'G:\\My Drive\\PROJECTS\\OBSERV\\Lookup Tables\\ESYRCE\\isSeminatural.csv'
#functionsFolder     = home + '\\git\\Python\\ESYRCE\\'
inputESYRCE         = home + '/DATA/OBServ/ESYRCE/PROCESSED/z30/flagged/'
outFolder           = home + '/DATA/OBServ/ESYRCE/PROCESSED/z30/metrics/'
logFile             = home + '/DATA/OBServ/ESYRCE/PROCESSED/logs/addMetrics.log'
tableCultivarDemand = home + '/lookup/Cultivar-Demand.csv'
tableIsCrop         = home + '/lookup/isCrop.csv'
tableIsSeminatural  = home + '/lookup/isSeminatural.csv'
functionsFolder     = home + '/git/Python/ESYRCE/'


# The functions used to calculate the metrics are stored in a different file, to make this script cleaner 
sys.path.append(functionsFolder)
import functions

# Log
buffSize = 1
log = open(logFile, "a", buffering=buffSize)
log.write("\n")
log.write("PROCESS addMetrics.py STARTED AT: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'\n')

# Land cover types (associating to esyrce codes, add more or remove if needed), to calculate proportion in each segment and average yield
# Notes: 
# - Association of crops (e.g. MN-PE) are not considered as such (too many combinations). 
#   We only take the 1st two characters of the attribute D5_CUL (e.g. 'MN-PE' will be treated as 'MN').
landCoverTypes = {'hardWheat':          'TD',
                  'softWheat':          'TB', 
                  'barleyTwoRows':      'C2',
                  'barleySixRows':      'C6',
                  'oat':                'AV',
                  'rye':                'CN',
                  'triticale':          'TT',
                  'rice':               'AR',
                  'maize':              'MA',
                  'sorghum':            'SR',
                  'mixCerealGrain':     'MC',
                  'otherCerealGrain':   'CX',
                  'quinoa':             'QN',
                  'frijol':             'JS',
                  'fabaBean':           'HS',
                  'lentil':             'LE',
                  'chickpea':           'GA',
                  'pea':                'GS',
                  'commonVetch':        'VE',
                  'lupine':             'AT',
                  'carob':              'AL',
                  'otherLegumeGrain':   'LX',
                  'ervil':              'YE',
                  'potato':             'PT',
                  'sweetPotato':        'BT',
                  'yellowNutsedge':     'CY',
                  'otherTuber':         'TX',
                  'sugarcane':          'CA',
                  'sugarbeet':          'RM',
                  'cotton':             'AD',
                  'sunflower':          'GI',
                  'soybean':            'SO',
                  'rapeseed':           'CZ',
                  'flax':               'LN',
                  'peanut':             'CC',
                  'camelina':           'KM',
                  'safflower':          'KR',
                  'otherOleaginous':    'OX',
                  'tobacco':            'TA',
                  'industTomato':       'TI',
                  'capsicumPaprika':    'PD',
                  'condiment':          'CD',
                  'hops':               'LU',
                  'finesHerbes':        'AA',
                  'otherIndustrial':    'IX',
                  'maizeFodder':        'MF',
                  'alfalfa':            'AF',
                  'vetchFodder':        'VF',
                  'otherFodder':        'FV',
                  'grasslandPolifite':  'PP',
                  'turnipFodder':       'NF',
                  'beetFodder':         'RF',
                  'collard':            'CS',
                  'otherWeedFodder':    'RX',
                  'betaVulgarisCicla':  'AZ',
                  'garlic':             'AJ',
                  'artichoke':          'AC',
                  'celery':             'AP',
                  'eggplant':           'BE',
                  'pumpkin':            'CW',
                  'zucchini':           'CB',
                  'agaricusBisporus':   'HP',
                  'mushroom':           'ST',
                  'onion':              'CL',
                  'broccoli':           'CI',
                  'cabbage':            'CM',
                  'cauliflower':        'CK',
                  'endive':             'EL',
                  'aspargus':           'EP',
                  'spinach':            'EI',
                  'sweetCorn':          'MD',
                  'strawberry':         'FN',
                  'rapini':             'GE',
                  'greenPea':           'GV',
                  'greenBean':          'HV',
                  'kidneyBean':         'JV',
                  'lettuce':            'LC',
                  'redCabbage':         'LO',
                  'melon':              'MO',
                  'cucumber':           'PI',
                  'leek':               'PU',
                  'beetTable':          'RW',
                  'sweetPepper':        'PQ',
                  'watermelon':         'SA',
                  'carrot':             'CT',
                  'otherVegetable':     'HX',
                  'emptyGarden':        'VH',
                  'ornamental':         'FO',     
                  'orange':             'NR',
                  'clementine':         'MR',
                  'lemon':              'LI',
                  'bitterOrange':       'NG',
                  'grapefruit':         'PA',
                  'otherCitrics':       'AX',
                  'apple':              'MN',
                  'pear':               'PE',
                  'quince':             'MB',
                  'loquat':             'NI',
                  'apricot':            'AB',
                  'cherry':             'CE',
                  'peach':              'ME',
                  'plum':               'CR',
                  'fig':                'HI',
                  'cherimoya':          'CH',
                  'avocado':            'AU',
                  'banana':             'PL',
                  'persimmon':          'CQ',
                  'papaya':             'PY',
                  'pineapple':          'PÑ',
                  'kiwi':               'KW',
                  'barbaryFig':         'CU',
                  'mango':              'MG',
                  'pomegranate':        'GR',
                  'almond':             'AM',
                  'walnut':             'NU',
                  'hazelnut':           'AE',
                  'chestnut':           'CJ',
                  'redRaspberry':       'FB',
                  'pistachio':          'PX',
                  'otherFruitNoCitric': 'FX',
                  'whiteGrapeSeedless': 'V1',
                  'whiteGrape':         'V2',
                  'redGrapeSeedless':   'V3',
                  'redGrape':           'V4',
                  'transfGrape':        'VT',
                  'oliveTable':         'OM',
                  'olive':              'OD',
                  'oliveMill':          'OT',
                  'carobTree':          'AO',
                  'otherOtherWoody':    'NX',
                  'nursery':            'VV',
                  'populus':            'CP',
                  'pawlonia':           'PZ',
                  'quercusIlexTruffle': 'ET',
                  'fallow':             'BA',  
                  'emptyGreenh':        'VA',     
                  'orchard':            'HU',  
                  'posio':              'PO',  
                  'naturalMeadow':      'PR',  
                  'highMountainMeadow': 'PH',
                  'pastureGrassland':   'PS',
                  'pastureShrub':       'PM',
                  'conifers':           'CO',
                  'broadleafFast':      'FR',
                  'broadleafSlow':      'FL',
                  'poplar':             'CP',
                  'mixedForest':        'CF',
                  'shrub':              'ML',
                  'wasteland':          'ER',
                  'spartizal':          'ES',
                  'abandoned':          'BL',
                  'improductive':       'IM',
                  'notAgri':            'NA'
                 }                           


# Soil codes. Apply to land cover types of woody crops and fallow: 
soilCodes = {'traditional':   'LT',
             'minimal':       'LM',
             'spontVegCover': 'CE',
             'sowedVegCover': 'CS',
             'inertCover':    'CP',
             'noMainten':     'SM',
             'noTillage':     'NL'}

# Sowing codes. Apply to cereals, sunflower, fodder corn and fodder cereals: 
sowCodes = { 'directSowing':  'D',
             'traditSowing':  'N'} 

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
    data['aggClass'] = [functions.setAggregatedClass(data.loc[i], dictIsSeminatural, dictIsCrop) for i in data.index]
    data = data.loc[(data['aggClass'] != "Exception")]
    
    # Select columns, sort and reset indices
    data = data[['D1_HUS','D2_NUM','D3_PAR','D4_GRC','D5_CUL','D9_RTO','DE_CS','YEA','Shape_Area','Shape_Leng','aggClass','geometry']]
    data = data.dropna(thresh=1)
    data = data.where(data['D1_HUS'] != 0)
    data = data.where(data['D2_NUM'] != 0)
    data.sort_values(by=['D1_HUS','D2_NUM','YEA'], inplace = True)
    data.reset_index(drop=True, inplace=True)

    # Init new columns with NaN data
    if getLandCoverProportion:     
        for x in landCoverTypes.keys(): 
            data['prop_'+x] = np.repeat(np.nan, len(data))
    if getCropYield:               
        for x in landCoverTypes.keys():      
            data['yield_'+x] = np.repeat(np.nan, len(data))
            data['var_'+x]  = np.repeat(np.nan, len(data)) 
    if getSoilTechniqueProportion: 
        for x in soilCodes.keys():      data[x] = np.repeat(np.nan, len(data))
    if getSowTechniqueProportion:  
        for x in sowCodes.keys():       data[x] = np.repeat(np.nan, len(data))
    if getAvgFieldSize:                 data['avgFieldSize']       = np.repeat(np.nan, len(data))
    if getAvgSeminatSize:               data['avgSeminatSize']     = np.repeat(np.nan, len(data))
    if getAvgFieldSizeDiss:             data['avgFieldSizeDiss']   = np.repeat(np.nan, len(data))
    if getAvgSeminatSizeDiss:           data['avgSeminatSizeDiss'] = np.repeat(np.nan, len(data))
    if getHeterogeneity:                data['heterogeneity']      = np.repeat(np.nan, len(data))
    if getDemand:                       data['demand']             = np.repeat(np.nan, len(data))    
    if getSegmentArea:                  data['segArea']            = np.repeat(np.nan, len(data))
    if getSegmentAreaWithoutWater:      data['segAreaNoWater']     = np.repeat(np.nan, len(data))
    if getEdgeDensity:                  data['edgeDensity']        = np.repeat(np.nan, len(data))
    if getEdgeDensitySeminatural:       data['edgeDenSeminat']     = np.repeat(np.nan, len(data))
    if getEdgeDensityCropfields:        data['edgeDenFields']      = np.repeat(np.nan, len(data))
    if getEdgeDensDissolved:            data['edgeDensityDiss']    = np.repeat(np.nan, len(data))
    if getEdgeDensitySeminatDiss:       data['edgeDenSemiDiss']    = np.repeat(np.nan, len(data))
    if getEdgeDensityCropDiss:          data['edgeDenFielDiss']    = np.repeat(np.nan, len(data))
    if getEdgeDensityOtherDiss:         data['edgeDenOtherDiss']   = np.repeat(np.nan, len(data))

    
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
                
                # Get dissolved segment if necessary
                if getEdgeDensDissolved or getEdgeDensitySeminatDiss or getEdgeDensityCropDiss or getEdgeDensityOtherDiss or getAvgFieldSizeDiss or getAvgSeminatSizeDiss:
                    dataSegmYearDiss = dataSegmentYear.dissolve(by=['aggClass'])
                    dataSegmYearDiss['aggClass'] = np.array(dataSegmYearDiss.index)
                    dataSegmYearDiss.Shape_Area = dataSegmYearDiss.geometry.area
                    dataSegmYearDiss.Shape_Leng = dataSegmYearDiss.geometry.length
                    dataSegmYearDiss.sort_values(by=['D1_HUS','D2_NUM','YEA'], inplace = True)
                    dataSegmYearDiss.reset_index(drop=True, inplace=True)
                    
                # Calculate metrics and assign values in 'data'
                if getLandCoverProportion:     
                    landCoverProportion = functions.calculateLandCoverProportion(dataSegmentYear, landCoverTypes, log)
                    for x in landCoverTypes.keys(): data.loc[dataSegmentYear.index, 'prop_'+x] = np.repeat(landCoverProportion[x], len(dataSegmentYear)) 
                if getCropYield:               
                    cropYield           = functions.calculateCropYield(dataSegmentYear, landCoverTypes, log)
                    varYield            = functions.calculateVarianceYield(dataSegmentYear, landCoverTypes, cropYield, log)
                    for x in cropYield.keys():      data.loc[dataSegmentYear.index, 'yield_'+x] = np.repeat(cropYield[x], len(dataSegmentYear))
                    for x in cropYield.keys():      data.loc[dataSegmentYear.index, 'var_'+x] = np.repeat(varYield[x], len(dataSegmentYear))  
                if getSoilTechniqueProportion: 
                    soilTechnProportion = functions.calculateSoilTechniqueProportion(dataSegmentYear, soilCodes, sowCodes, log) 
                    for x in soilCodes.keys():      data.loc[dataSegmentYear.index, x] = np.repeat(soilTechnProportion[x], len(dataSegmentYear))
                if getSowTechniqueProportion:  
                    sowTechnProportion  = functions.calculateSoilTechniqueProportion(dataSegmentYear, sowCodes, soilCodes, log) 
                    for x in sowCodes.keys():       data.loc[dataSegmentYear.index, x] = np.repeat(sowTechnProportion[x], len(dataSegmentYear))                  
                if getAvgFieldSize:                 
                    avgFieldSize        = functions.calculateAvgFieldSize(dataSegmentYear, log)
                    data.loc[dataSegmentYear.index, 'avgFieldSize'] = np.repeat(avgFieldSize, len(dataSegmentYear))
                if getAvgSeminatSize:               
                    avgSeminatSize      = functions.calculateAvgSeminaturalSize(dataSegmentYear, log)
                    data.loc[dataSegmentYear.index, 'avgSeminatSize'] = np.repeat(avgSeminatSize, len(dataSegmentYear))
                if getAvgFieldSizeDiss:                 
                    avgFieldSizeDiss    = functions.calculateAvgFieldSize(dataSegmYearDiss, log)
                    data.loc[dataSegmentYear.index, 'avgFieldSizeDiss'] = np.repeat(avgFieldSizeDiss, len(dataSegmentYear))
                if getAvgSeminatSizeDiss:               
                    avgSeminatSizeDiss  = functions.calculateAvgSeminaturalSize(dataSegmYearDiss, log)
                    data.loc[dataSegmentYear.index, 'avgSeminatSizeDiss'] = np.repeat(avgSeminatSizeDiss, len(dataSegmentYear))                    
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
                if getEdgeDensDissolved:
                    edgeDensityDiss    = functions.calculateEdgeDensity(dataSegmYearDiss, log)
                    data.loc[dataSegmentYear.index, 'edgeDensityDiss'] = np.repeat(edgeDensityDiss, len(dataSegmentYear))   
                if getEdgeDensitySeminatDiss:      
                    edgeDenSemiDiss      = functions.calculateEdgeDensitySeminatural(dataSegmYearDiss, log)
                    data.loc[dataSegmentYear.index, 'edgeDenSemiDiss'] = np.repeat(edgeDenSemiDiss, len(dataSegmentYear))  
                if getEdgeDensityCropDiss:      
                    edgeDenFielDiss      = functions.calculateEdgeDensityFields(dataSegmYearDiss, log)
                    data.loc[dataSegmentYear.index, 'edgeDenFielDiss'] = np.repeat(edgeDenFielDiss, len(dataSegmentYear))                    
                if getEdgeDensityOtherDiss:        
                    edgeDenOtherDiss     = functions.calculateEdgeDensityOther(dataSegmYearDiss, log)
                    data.loc[dataSegmentYear.index, 'edgeDenOtherDiss'] = np.repeat(edgeDenOtherDiss, len(dataSegmentYear)) 

            contNr = contNr+1
            if np.mod(contNr, 100) == 0:
                times = contNr / totalNr 
                log.write("Now: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'\n')
                log.write("Processing File..."+filename+" Data Zone..."+str(int(zoneNr))+" Percentage completed..."+str(np.floor(times*100))+'\n')

    # Group by number of the segment and year, drop not useful columns, and save to csv
    data = data.drop(columns=['D3_PAR','D4_GRC','D5_CUL','D9_RTO','DE_CS','Shape_Area','Shape_Leng','aggClass','geometry'])
    data = data.groupby(['D1_HUS','D2_NUM','YEA']).first().reset_index()
    log.write("Now: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S")+'\n')
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
