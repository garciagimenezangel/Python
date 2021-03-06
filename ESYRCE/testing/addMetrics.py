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
getLandCoverControlPoints  = True  # Get Land Cover at control points, to register trajectories (e.g. maize->barley->spartizal->maize)
getLandCoverProportion     = False # Percentage of the land cover types (see variable 'landCoverTypes' below)
getSoilTechniqueProportion = False # Soil maintenance technique proportion (see variable 'soilCodes' below)
getSowTechniqueProportion  = False # Sowing technique proportion (direct or traditional)
getCropYield               = False # Average and variance of the yield of each crop within the segments (see variable 'cropCodes' below) 
getAvgSize                 = False # Average size of the polygons (water ignored)
getAvgFieldSize            = False # Average size of the fields identified as crops 
getAvgSeminatSize          = False # Average size of the fields identified as seminatural area 
getAvgOtherSize            = False # Average size of the fields identified as other
getAvgSizeDiss             = False # Average size of the polygons (water ignored) dissolving by group
getAvgFieldSizeDiss        = False # Average size of the fields identified as crops dissolving by group
getAvgSeminatSizeDiss      = False # Average size of the fields identified as seminatural area dissolving by group
getAvgOtherSizeDiss        = False # Average size of the fields identified as other dissolving by group
getHeterogeneity           = False # Heterogeneity, as number of crop types per unit area
getDemand                  = False # Average demand, weighted by the area of the polygons 
getSegmentArea             = False # Total area of the segments
getSegmentAreaWithoutWater = False # Area of the segments, ignoring water 
getEdgeDensity             = False # Density of edges (length/area)
getEdgeDensitySeminatural  = False # Density of edges from seminatural area (length/area)
getEdgeDensityCropfields   = False # Density of edges from crop fields (length/area)
getEdgeDensityOther        = False # Density of edges from other landcover types (length/area)
getEdgeDensDissolved       = False # Density of edges (total) dissolving by 'isCropfield' and 'isSeminatural'
getEdgeDensitySeminatDiss  = False # Density of edges (seminatural) dissolving by 'isCropfield' and 'isSeminatural'
getEdgeDensityCropDiss     = False # Density of edges (cropfields) dissolving by 'isCropfield' and 'isSeminatural'
getEdgeDensityOtherDiss    = False # Density of edges (others) dissolving by 'isCropfield' and 'isSeminatural'
getSystemProportion        = False  # Percentage of each crop system: dry, water scarce (normally irrigated but dry because of water scarcity), irrigation or greenhouse

# Final output
finalFilename = "landCoverTransitions"

# Paths
inputESYRCE         = home + '\\DATA\\ESYRCE\\PROCESSED - local testing\\z30\\flagged\\test3\\'
outFolder           = home + '\\DATA\\ESYRCE\\PROCESSED - local testing\\z30\\metrics\\test3\\'
logFile             = home + '\\DATA\\ESYRCE\\PROCESSED - local testing\\logs\\addMetrics.log'
EuskadiSegmentsCsv  = home + '\\DATA\\ESYRCE\\landCoverChange\\D1HUS_D2NUM_flag012_Euskadi.csv'
centroidPtsShp      = home + '\\DATA\\ESYRCE\\landCoverChange\\centroids_HUS_NUM.shp'
tableCultivarDemand = 'G:\\My Drive\\PROJECTS\\OBSERV\\Lookup Tables\\ESYRCE\\Cultivar-Demand.csv'
tableIsCropSeminat  = 'G:\\My Drive\\PROJECTS\\OBSERV\\Lookup Tables\\ESYRCE\\isCropSeminatural.csv'
functionsFolder     = home + '\\git\\Python\\ESYRCE\\'
#inputESYRCE         = home + '/DATA/ESYRCE/PROCESSED/z30/flagged/'
#outFolder           = home + '/DATA/ESYRCE/PROCESSED/z30/metrics/'
#logFile             = home + '/DATA/ESYRCE/PROCESSED/logs/addMetrics.log'
#EuskadiSegmentsCsv  = home + '/DATA/ESYRCE/landCoverChange/D1HUS_D2NUM_flag012_Euskadi.csv'
#centroidPtsShp      = home + '/DATA/ESYRCE/landCoverChange/centroids_HUS_NUM.shp'
#tableCultivarDemand = home + '/lookup/Cultivar-Demand.csv'
#tableIsCropSeminat  = home + '/lookup/isCropSeminatural.csv'
#functionsFolder     = home + '/git/Python/ESYRCE/'


# The functions used to calculate the metrics are stored in a different file, to make this script cleaner 
sys.path.append(functionsFolder)
import functions

# Log
buffSize = 1
log = open(logFile, "a", buffering=buffSize)

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
                  'broadBean':          'HV',
                  'greenBean':          'JV',
                  'lettuce':            'LC',
                  'redCabbage':         'LO',
                  'melon':              'MO',
                  'cucumber':           'PI',
                  'leek':               'PU',
                  'beetTable':          'RW',
                  'sweetPepper':        'PQ',
                  'watermelon':         'SA',
                  'tomato':             'TO',
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

landCoverTypes_reverse = {}
for k, v in landCoverTypes.items():
    landCoverTypes_reverse[v] = k

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

# Crop system codes: dry, water scarce (normally irrigated but dry because of water scarcity), irrigated or greenhouse
systemCodes = {'dry': 'S',
               'waterScarce': 'F',
               'irrigated': 'R',
               'greenhouse': 'I'}

# Define dictionaries from tables
# Dictionary to associate codes with crop category
with open(tableIsCropSeminat, mode='r', encoding='latin-1') as infile:
    reader     = csv.reader(infile)
    dictIsCrop = {rows[0]:rows[1] for rows in reader} # keys: esyrce codes; values: 'YES' or 'NO'

# Dictionary to associate codes with seminatural category
with open(tableIsCropSeminat, mode='r', encoding='latin-1') as infile:
    reader     = csv.reader(infile)
    dictIsSeminatural = {rows[0]:rows[2] for rows in reader} # keys: esyrce codes; values: 'YES' or 'NO'
    
# Dictionary to associate crop codes with demand
# In ubuntu you may need to change encoding of this file: iconv -f ISO-8859-1 -t utf-8 Cultivar-Demand.csv > Cultivar-Demand-utf8.csv
with open(tableCultivarDemand, mode='r', encoding='latin-1') as infile:
    reader   = csv.reader(infile)       
    dictCultivarDemand = {rows[0]:rows[1] for rows in reader} # key: 'esyrce codes; value: demand estimation (see dictDemandValues defined at the beginning of this file)

# Control points for land cover change
if getLandCoverControlPoints:      
    centroidPts = gpd.read_file(centroidPtsShp).drop_duplicates()
    centroidPts['D2_NUM'] = centroidPts['D2_NUM'].round(decimals=0).astype('int64')
    
##################
# Loop files
file = glob.glob(inputESYRCE + "*.shp")[0]
    
    # OUTPUT
filename = file.split(inputESYRCE)[1]
filename = filename.split(".shp")[0]
outFilename = outFolder+filename+".csv"

# Read data
data = gpd.read_file(file)
   
# Modify or create useful columns
data.Shape_Area = data.geometry.area
data.Shape_Leng = data.geometry.length
data['aggClass'] = [functions.getAggregatedClass(data.loc[i], dictIsSeminatural, dictIsCrop) for i in data.index]
data = data.loc[(data['aggClass'] != "Exception")]
EuskadiSegments = pd.read_csv(EuskadiSegmentsCsv).drop_duplicates().round(decimals=0).astype('int64')
data['D2_NUM']  = data['D2_NUM'].round(decimals=0).astype('int64')
data['isEuskadi'] = [functions.isEuskadiSegment(data.loc[i], EuskadiSegments) for i in data.index]

# Select columns, remove duplicates (detected many times for 2019 data), sort and reset indices
data = data[['D1_HUS','D2_NUM','D3_PAR','D4_GRC','D5_CUL','D7_SRI','D9_RTO','DE_CS','YEA','Shape_Area','Shape_Leng','aggClass','isEuskadi','geometry']]
data = data.dropna(thresh=1)
data = data.loc[data['D1_HUS'] != 0]
data = data.loc[data['D2_NUM'] != 0]
data = data.groupby(['D1_HUS','D2_NUM','YEA','D3_PAR'], as_index=False).first() 
data.sort_values(by=['D1_HUS','D2_NUM','YEA'], inplace = True)
data.reset_index(drop=True, inplace=True)
data = gpd.GeoDataFrame(data, geometry=data.geometry)

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
if getSystemProportion:             
    for x in systemCodes.keys():    data[x] = np.repeat(np.nan, len(data))
if getAvgSize:                      data['avgSize']            = np.repeat(np.nan, len(data))
if getAvgFieldSize:                 data['avgFieldSize']       = np.repeat(np.nan, len(data))
if getAvgSeminatSize:               data['avgSeminatSize']     = np.repeat(np.nan, len(data))
if getAvgOtherSize:                 data['avgOtherSize']       = np.repeat(np.nan, len(data))
if getAvgFieldSizeDiss:             data['avgFieldSizeDiss']   = np.repeat(np.nan, len(data))
if getAvgSizeDiss:                  data['avgSizeDiss']        = np.repeat(np.nan, len(data))
if getAvgSeminatSizeDiss:           data['avgSeminatSizeDiss'] = np.repeat(np.nan, len(data))
if getAvgOtherSizeDiss:             data['avgOtherSizeDiss']   = np.repeat(np.nan, len(data))
if getHeterogeneity:                data['heterogeneity']      = np.repeat(np.nan, len(data))
if getDemand:                       data['demand']             = np.repeat(np.nan, len(data))    
if getSegmentArea:                  data['segArea']            = np.repeat(np.nan, len(data))
if getSegmentAreaWithoutWater:      data['segAreaNoWater']     = np.repeat(np.nan, len(data))
if getEdgeDensity:                  data['edgeDensity']        = np.repeat(np.nan, len(data))
if getEdgeDensitySeminatural:       data['edgeDenSeminat']     = np.repeat(np.nan, len(data))
if getEdgeDensityCropfields:        data['edgeDenFields']      = np.repeat(np.nan, len(data))
if getEdgeDensityOther:             data['edgeDenOther']       = np.repeat(np.nan, len(data))
if getEdgeDensDissolved:            data['edgeDensityDiss']    = np.repeat(np.nan, len(data))
if getEdgeDensitySeminatDiss:       data['edgeDenSemiDiss']    = np.repeat(np.nan, len(data))
if getEdgeDensityCropDiss:          data['edgeDenFielDiss']    = np.repeat(np.nan, len(data))
if getEdgeDensityOtherDiss:         data['edgeDenOtherDiss']   = np.repeat(np.nan, len(data))
if getLandCoverControlPoints:      
    data['lccp1'] = np.repeat(np.nan, len(data))
    data['lccp2'] = np.repeat(np.nan, len(data))
    data['lccp3'] = np.repeat(np.nan, len(data))
    data['lccp4'] = np.repeat(np.nan, len(data))
    data['lccp5'] = np.repeat(np.nan, len(data))
    data['lccp6'] = np.repeat(np.nan, len(data))
    data['lccp7'] = np.repeat(np.nan, len(data))
    data['lccp8'] = np.repeat(np.nan, len(data))
    data['lccp9'] = np.repeat(np.nan, len(data))

##################
# Loop zones
zoneNr = np.unique(data.D1_HUS)[0]

# Select zone data 
ii = np.where(data.D1_HUS == zoneNr) 
i0 = ii[0][0]
iM = ii[0][len(ii[0])-1]
dataZoneNr = data[i0:(iM+1)]
if (iM-i0+1)!=len(ii[0]): # sanity check
    log.write("Error... Exit loop in zone nr:"+str(zoneNr)+'\n') # sanity check

# Loop plot numbers
segmentNr = np.unique(dataZoneNr.D2_NUM)[0]

# Get the rows corresponding to segmentNr. This is not the safest way (it doesn't work if the rows are not sorted out by number of segment, which seems not to be the case), 
# but it is the fastest way. A sanity check is included to make sure that this way works reasonably well. If the "Error..." message is prompted too much times, then 
# this way of getting the rows for the segmentNr must be replaced by the slow way (see filterDataByExtent.py as an example)
ii = np.where(dataZoneNr.D2_NUM == segmentNr) 
i0 = ii[0][0]
iM = ii[0][len(ii[0])-1]
dataSegmentNr = dataZoneNr[i0:(iM+1)]
if (iM-i0+1)!=len(ii[0]): # sanity check
    log.write("Error... Exit loop in Segment nr:"+str(segmentNr)+'\n') # sanity check

year = np.unique(dataSegmentNr.YEA)[0]

# Get the rows corresponding to a particular year. As above-mentioned, not the safest way, but the fastest (to my knowledge), so a sanity check is needed.
ii = np.where(dataSegmentNr.YEA == year)
i0 = ii[0][0]
iM = ii[0][len(ii[0])-1]
dataSegmentYear = dataSegmentNr[i0:(iM+1)]
if (iM-i0+1)!=len(ii[0]): # sanity check
    log.write("Error... Exit loop in Segment nr:"+ str(segmentNr)+ "...Year:"+str(year)+'\n')  
                


