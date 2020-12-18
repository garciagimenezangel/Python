# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:56:21 2020

@author: angel.gimenez
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from sklearn import linear_model
import glob

dictDemandValues = { ''           :     0,
                   'unknown':           0, 
                   'no increase':       0, 
                   'increase':          0.5,
                   'increase-breeding': 0,
                   'increase-seed production': 0,
                   'little': 0.05,
                   'modest': 0.25,
                   'great':  0.65,
                   'essential': 0.95}


"""
INPUT: subset of ESYRCE data corresponding to one segment number
OUTPUT: polygon corresponding to the data of the year with the smallest spatial coverage
"""
def getPolygonToClip(dataSegmentNr, log):
    years = np.unique(dataSegmentNr.YEA)
    cont = 0
    for year in years:
        ii = np.where(dataSegmentNr.YEA == year)
        i0 = ii[0][0]
        iM = ii[0][len(ii[0])-1]
        dataSegmentYear = dataSegmentNr[i0:(iM+1)]
        try:
            dissolved = dataSegmentYear.dissolve(by='YEA')    
        except:
            log.write("Problems dissolving segment "+str(dataSegmentNr.iloc[0]['D2_NUM'])+'\n')
            return gpd.GeoDataFrame()
            
        if (cont == 0):  
            intersection = dissolved
            cont = cont+1
        else:
            try:
                intersection = gpd.overlay(intersection, dissolved, how='intersection')
            except:
                log.write("Problems performing intersection "+str(dataSegmentNr.iloc[0]['D2_NUM'])+'\n')
                return gpd.GeoDataFrame()
    polygon = gpd.GeoDataFrame(intersection.geometry)
    polygon.crs = dataSegmentNr.crs
    return gpd.GeoDataFrame(intersection.geometry)


"""
INPUT: subset of ESYRCE data corresponding to one segment number
OUTPUT: quality flag. 0:ok; 1:size changed, clip to smallest; 2: segments not aligned; 3: geometry problems dissolving
"""
def getSegmentQualityFlag(dataSegmentNr, tol):
    flag = 0 
    years = np.unique(dataSegmentNr.YEA)
    cont = 0
    for year in years:
        ii = np.where(dataSegmentNr.YEA == year)
        i0 = ii[0][0]
        iM = ii[0][len(ii[0])-1]
        dataSegmentYear = dataSegmentNr[i0:(iM+1)]
        try:
            dissolved     = dataSegmentYear.dissolve(by='YEA')    
            newDissGeo    = dissolved.geometry
            newBBox       = newDissGeo.bounds
        except:
            return 3
            
        if (cont == 0):  
            minBBox = newBBox
            cont = cont+1
        else:
            lastMinX = minBBox.iloc[0][0];
            lastMinY = minBBox.iloc[0][1];            
            newMinX  = newBBox.iloc[0][0];
            newMinY  = newBBox.iloc[0][1];
            lastMaxX = minBBox.iloc[0][2];
            lastMaxY = minBBox.iloc[0][3];            
            newMaxX  = newBBox.iloc[0][2];
            newMaxY  = newBBox.iloc[0][3];      
            if (not(np.isclose(lastMinX, newMinX, atol=tol)) or not(np.isclose(lastMinY, newMinY, atol=tol))):
                return 2
            elif (not(np.isclose(lastMaxX, newMaxX, atol=tol)) or not(np.isclose(lastMaxY, newMaxY, atol=tol))):
                flag = 1
                minBBox = newBBox
            else:
                minBBox = newBBox
                
    return flag
        

"""
INPUT: 
    - a subset of ESYRCE data corresponding to a segment for a particular year 
    - dictionary of land cover types with associated ESYRCE codes
    
OUTPUT: dictionary with the proportion of each land cover type within the segment

Note: water is ignored in the calculations
"""
def calculateLandCoverProportion(dataSegmentYear, landCoverTypes, log):
    
    # Read codes from dictionary landCoverTypes
    keys     = list(landCoverTypes.keys())
    lcCul    = list(landCoverTypes.values())
    lcGrc    = list(['IM','NA'])    
    
    # Initialize variables to store accumulated area values 
    lcAcc = np.zeros(len(lcCul)) # accumulated area of each land cover type
    totalArea = 0
        
    # Iterate through the polygons in dataSegmentYear
    for index in dataSegmentYear.index:
        # Ignore water codes
        try:
            if isWater(dataSegmentYear.loc[index]): continue    
        except Exception as e:
            log.write(str(e))
            continue   
        
        # area of the polygon
        areaPolygon = dataSegmentYear.loc[index].Shape_Area
        
        # First, see if D4_GRC correspond to IM or NA. If not, then use D5_CUL
        polyGrc = dataSegmentYear.loc[index].D4_GRC
        if polyGrc in lcGrc:
            if polyGrc in lcCul:
                ind = lcCul.index(polyGrc)
                lcAcc[ind] = lcAcc[ind] + areaPolygon
                totalArea = totalArea + areaPolygon 
        # If not found with D4_GRC, use 2 first characteres of D5_CUL 
        else: 
            try:
                polyCul = dataSegmentYear.loc[index].D5_CUL[0:2]
            except:
                log.write("Problem with land cover codes:"+str(dataSegmentYear.loc[index].D2_NUM)+
                      "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                      "...Year:"+str(dataSegmentYear.loc[index].YEA)+'\n')
                continue            
            
            if polyCul in lcCul:
                ind = lcCul.index(polyCul)
                lcAcc[ind] = lcAcc[ind] + areaPolygon
                totalArea = totalArea + areaPolygon 
            else: 
                log.write("Warning: Index not found in calculateLandCoverPercentages. Parcel IGNORED"+
                  "...Segment:" +str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"  +str(dataSegmentYear.loc[index].YEA)+
                  "...D5_CUL:"+str(polyCul)+'\n')

    if totalArea != 0:
        values = lcAcc/totalArea
    else:
        values = np.zeros(len(keys))
        
    # Output dictionary. Key: land cover type; value: accumulated area in the segment
    return dict((keys[ind], values[ind]) for ind in range(0,len(keys)))          

    

"""
INPUT: 
    - a subset of ESYRCE data corresponding to a segment for a particular year 
    - dictionary of soil techniques with associated ESYRCE codes. 
    - dictionary of soil techniques to be ignored (in order to use the same function for soil maintenance and sowing techniques). 
    
OUTPUT: dictionary with the porportion of each soil management technique within the segment

Note: proportion is computed only with regard to codes present in the input dictionary. If no code present, then returns 0.
"""
def calculateSoilTechniqueProportion(dataSegmentYear, soilCodes, ignoreCodes, log):
        
    # Read codes from dictionary landCoverTypes
    keys     = list(soilCodes.keys())
    codesCS  = list(soilCodes.values())
    
    # Initialize variables to store accumulated area values 
    soilAcc = np.zeros(len(codesCS)) # accumulated area of each soil management technique
    totalArea = 0
    
    # Iterate through the polygons in dataSegmentYear
    for index in dataSegmentYear.index:
        # area of the polygon
        areaPolygon = dataSegmentYear.loc[index].Shape_Area
        
        # soil management technique code
        polyCS = str(dataSegmentYear.loc[index].DE_CS)
    
        # If code found (polyCS != None), add area to the corresponding soil management technique
        if (polyCS != 'nan') and (polyCS != 'None'):
            # Codes to be ignored
            if polyCS in list(ignoreCodes.values()): continue
    
            # identify soil technique index
            if polyCS in codesCS:
                ind = codesCS.index(polyCS)
                soilAcc[ind] = soilAcc[ind] + areaPolygon
                totalArea    = totalArea + areaPolygon
            else: 
                log.write("Index not found in calculateSoilTechniqueProportion. Parcel IGNORED"+
                  "...Segment:" +str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"  +str(dataSegmentYear.loc[index].YEA)+
                  "...DE_CS:"+str(polyCS)+'\n')
                        
    if totalArea != 0:
        values = soilAcc/totalArea
    else:
        values = np.zeros(len(keys))
        
    # Output dictionary. Key: soil management technique; value: accumulated area in the segment
    return dict((keys[ind], values[ind]) for ind in range(0,len(keys)))              


"""
INPUT: a subset of ESYRCE data corresponding to a segment for a particular year 
OUTPUT: average size of the polygons (in hectares)
"""
def calculateAvgSize(dataSegmentYear, log):
    
    # Iterate through the polygons in dataSegmentYear
    accArea     = 0
    nParcels = 0
    for index in dataSegmentYear.index:  
        # Ignore water codes
        try:
            if isWater(dataSegmentYear.loc[index]): continue    
        except Exception as e:
            log.write(str(e))
            continue  
        
        nParcels = nParcels + 1
        accArea  = accArea + dataSegmentYear.loc[index].Shape_Area
    
    if nParcels > 0:
        return accArea * 1e-4 / nParcels # convert m^2 into hectares
    else:
        return 0


"""
INPUT: a subset of ESYRCE data corresponding to a segment for a particular year 
OUTPUT: average size of the crop fields (in hectares)
"""
def calculateAvgFieldSize(dataSegmentYear, log):
    
    # Iterate through the polygons in dataSegmentYear
    accArea     = 0
    nCropfields = 0
    for index in dataSegmentYear.index:  
        if isCropfield(dataSegmentYear.loc[index]):   
            nCropfields = nCropfields + 1
            accArea  = accArea + dataSegmentYear.loc[index].Shape_Area
    
    if nCropfields > 0:
        return accArea * 1e-4 / nCropfields # convert m^2 into hectares
    else:
        return 0
    

"""
INPUT: a subset of ESYRCE data corresponding to a segment for a particular year 
OUTPUT: average size of the seminatural patches (in hectares)
"""
def calculateAvgSeminaturalSize(dataSegmentYear, log):   
    
    # Iterate through the polygons in dataSegmentYear
    accArea     = 0
    nSeminaturalPatches = 0
    for index in dataSegmentYear.index:  
        if isSeminatural(dataSegmentYear.loc[index]):   
            nSeminaturalPatches = nSeminaturalPatches + 1
            accArea  = accArea + dataSegmentYear.loc[index].Shape_Area
            
    if nSeminaturalPatches > 0:
        return accArea * 1e-4 / nSeminaturalPatches # convert m^2 into hectares
    else:
        return 0    
    

"""
INPUT: a subset of ESYRCE data corresponding to a segment for a particular year     
OUTPUT: average size of the seminatural patches (in hectares)
"""
def calculateAvgOtherSize(dataSegmentYear, log):   
    
    # Iterate through the polygons in dataSegmentYear
    accArea     = 0
    nOther = 0
    for index in dataSegmentYear.index:  
        if isOther(dataSegmentYear.loc[index]):   
            nOther = nOther + 1
            accArea  = accArea + dataSegmentYear.loc[index].Shape_Area
            
    if nOther > 0:
        return accArea * 1e-4 / nOther # convert m^2 into hectares
    else:
        return 0  
    

"""
INPUT: 
    - a subset of ESYRCE data corresponding to a segment for a particular year 
    - dictionary to find out whether a given ESYRCE code correspond to a crop or not
    
OUTPUT: number of crop types, per hectare
"""
def calculateHeterogeneity(dataSegmentYear, log):   
    
    # Iterate through the polygons in dataSegmentYear
    crops     = []
    totalArea = 0
    for index in dataSegmentYear.index:            
        # Ignore water codes
        try:
            if isWater(dataSegmentYear.loc[index]): continue    
        except Exception as e:
            log.write(str(e))
            continue  

        if isCropfield(dataSegmentYear.loc[index]):   
            crops = np.append(crops, dataSegmentYear.loc[index].D5_CUL)
        
        # add up area of the polygon (convert m^2 into hectares)
        totalArea = totalArea + dataSegmentYear.loc[index].Shape_Area*1e-4

    if totalArea > 0:
        return len(np.unique(crops)) / totalArea
    else:
        return 0
 
    
"""
INPUT: 
    - a subset of ESYRCE data corresponding to a segment for a particular year 
    - dictionary with indicators of demand for the crops 
    
OUTPUT: demand value for the segment, using an average weighted by the area of the polygons 
"""
def calculateDemand(dataSegmentYear, dictCultivarDemand, log):

    # Iterate through the polygons in dataSegmentYear
    totalArea = 0
    accDemand = 0
    for index in dataSegmentYear.index:                  
        polyCul = dataSegmentYear.loc[index].D5_CUL
        # area of the polygon
        areaPolygon = dataSegmentYear.loc[index].Shape_Area        
        # Ignore water codes
        try:
            if isWater(dataSegmentYear.loc[index]): continue    
        except Exception as e:
            log.write(str(e))
            continue         
        
        # Calculate demand. If association of cultivars, calculate average
        try:
            assocElts = polyCul.split("-")
        except:
            log.write("Problem with land cover codes:"+str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"+str(dataSegmentYear.loc[index].YEA)+'\n')
            continue
        demand    = 0
        if len(assocElts) > 0:
            for elt in assocElts: # average over all elements
                if elt in dictCultivarDemand:
                    increase = dictCultivarDemand[elt]
                    demand = demand + dictDemandValues[increase]               
            demand = demand / len(assocElts)
            
        accDemand = accDemand + demand*areaPolygon  
        totalArea = totalArea + areaPolygon
        
    if totalArea > 0:
        return accDemand / totalArea
    else:
        return 0


"""
INPUT: 
    - a subset of ESYRCE data corresponding to a segment for a particular year 
    - a dictionary of crop codes in ESYRCE 
    
OUTPUT: average yield of each crop within the segment 
"""
def calculateCropYield(dataSegmentYear, landCoverTypes, log):
    # Read codes from dictionary landCoverTypes
    keys     = list(landCoverTypes.keys())
    lcCul    = list(landCoverTypes.values())
    yieldVal = np.repeat(np.nan, len(keys)) 

    # Initialize variables to store accumulated yields and areas
    yieldAcc = np.zeros(len(lcCul)) # accumulated weighted yield (by area) of each crop type
    areaAcc  = np.zeros(len(lcCul)) # accumulated area of each crop type
    
    # Iterate through the polygons in dataSegmentYear
    for index in dataSegmentYear.index:
        
        # yield (if 0 or None, ignore segment)
        fieldYield = dataSegmentYear.loc[index].D9_RTO
        if(fieldYield is None): continue
        if(fieldYield == 0):    continue 

        # area of the polygon
        areaPolygon = dataSegmentYear.loc[index].Shape_Area
            
        # landcover codes (2 first characteres)
        try:
            polyCul = dataSegmentYear.loc[index].D5_CUL[0:2]
        except:
            log.write("Problem with land cover codes:"+str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"+str(dataSegmentYear.loc[index].YEA)+'\n')
            continue            
    
        # identify landcover index
        if polyCul in lcCul:
            ind = lcCul.index(polyCul)
            yieldAcc[ind] = yieldAcc[ind] + fieldYield*areaPolygon
            areaAcc[ind]  = areaAcc[ind]  + areaPolygon
            
        else: 
            log.write("Index not found in calculateCropYield. Parcel IGNORED"+
                  "...Segment:" +str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"  +str(dataSegmentYear.loc[index].YEA)+
                  "...D5_CUL:"+str(polyCul)+'\n')
            
    for ind in range(0,len(keys)):
        if (areaAcc[ind] > 0): yieldVal[ind] = yieldAcc[ind]/areaAcc[ind]
        
    # Output dictionary. Key: crop type; value: weighted average of the yield
    return dict((keys[ind], yieldVal[ind]) for ind in range(0,len(keys)))          


"""
INPUT: 
    - a subset of ESYRCE data corresponding to a segment for a particular year 
    - a dictionary of crop codes in ESYRCE 
    - weighted mean of each crop type
    
OUTPUT: weighted variance of the yield of each crop within the segment 
"""
def calculateVarianceYield(dataSegmentYear, landCoverTypes, weightedMeans, log):
    # Read codes from dictionary landCoverTypes
    keys     = list(landCoverTypes.keys())
    lcCul    = list(landCoverTypes.values())
    means    = list(weightedMeans.values())
    variance = np.repeat(np.nan, len(keys)) 

    # Initialize variables to store accumulated yields and areas
    sumSqAcc = np.zeros(len(lcCul)) # accumulated weighted (by area) sum of squares [area_i(x_i - mean)^2] for each crop type
    areaAcc  = np.zeros(len(lcCul)) # accumulated area of each crop type
    
    # Iterate through the polygons in dataSegmentYear
    for index in dataSegmentYear.index:       
        # yield (if 0 or None, ignore segment)
        fieldYield = dataSegmentYear.loc[index].D9_RTO
        if(fieldYield is None): continue
        if(fieldYield == 0):    continue 
    
        # area of the polygon
        areaPolygon = dataSegmentYear.loc[index].Shape_Area
        
        # landcover codes (2 first characteres)
        try:
            polyCul = dataSegmentYear.loc[index].D5_CUL[0:2]
        except:
            log.write("Problem with land cover codes:"+str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"+str(dataSegmentYear.loc[index].YEA)+'\n')
            continue            
    
        # identify landcover index and accumulate weighted variance and area
        if polyCul in lcCul:
            ind = lcCul.index(polyCul)
            sumSqAcc[ind] = sumSqAcc[ind] + areaPolygon*(fieldYield-means[ind])**2
            areaAcc[ind]  = areaAcc[ind]  + areaPolygon
            
        else: 
            log.write("Index not found in calculateCropYield. Parcel IGNORED"+
                  "...Segment:" +str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"  +str(dataSegmentYear.loc[index].YEA)+
                  "...D5_CUL:"+str(polyCul)+'\n')
            
    for ind in range(0,len(keys)):
        if (areaAcc[ind] > 0): variance[ind] = sumSqAcc[ind]/areaAcc[ind]
        
    # Output dictionary. Key: crop type; value: weighted variance of the yield
    return dict((keys[ind], variance[ind]) for ind in range(0,len(keys)))          


"""
INPUT: 
    - a subset of ESYRCE data corresponding to a segment for a particular year 
    
OUTPUT: edge density (meters per hectare), defined as (see manual FRAGSTATS) the 
sum of the lengths (m) of all edge segments, divided by the total area (hectares).

Notes: 
    - Edges of the segment are not considered. 
    - Water polygons are skipped.
"""
def calculateEdgeDensity(dataSegmentYear, log):   
    # Iterate through the polygons in dataSegmentYear
    accArea       = 0
    accEdgeLength = 0
    for index in dataSegmentYear.index:      
        # Ignore water codes
        try:
            if isWater(dataSegmentYear.loc[index]): continue    
        except Exception as e:
            log.write(str(e))
            continue              
        accArea = accArea + dataSegmentYear.loc[index].Shape_Area             # area
        accEdgeLength = accEdgeLength + dataSegmentYear.loc[index].Shape_Leng # perimeter
    if accArea > 0:
        return accEdgeLength / (accArea*1e-4) # m2 to hectares
    else:
        return 0   
    

"""
INPUT: 
    - a subset of ESYRCE data corresponding to a segment for a particular year 
    
OUTPUT: edge density of seminatural areas (meters per hectare), defined as 
(see manual FRAGSTATS) the sum of the lengths (m) of all edge segments, 
divided by the total area (hectares).

Notes: 
    - Edges of the segment are not considered. 
    - Water polygons are skipped.
"""
def calculateEdgeDensitySeminatural(dataSegmentYear, log):   
    
    # Iterate through the polygons in dataSegmentYear
    accArea       = 0
    accEdgeLength = 0
    for index in dataSegmentYear.index:   
        # Ignore water codes
        try:
            if isWater(dataSegmentYear.loc[index]): continue    
        except Exception as e:
            log.write(str(e))
            continue  
        accArea  = accArea + dataSegmentYear.loc[index].Shape_Area  # area
        if isSeminatural(dataSegmentYear.loc[index]):   
            accEdgeLength = accEdgeLength + dataSegmentYear.loc[index].Shape_Leng  # perimeter      
    if accArea > 0:
        return accEdgeLength / (accArea*1e-4)
    else:
        return 0  


"""
INPUT: 
    - a subset of ESYRCE data corresponding to a segment for a particular year 
    
OUTPUT: edge density of agricultural areas (meters per hectare), defined as 
(see manual FRAGSTATS) the sum of the lengths (m) of all edge segments, 
divided by the total area (hectares).

Notes: 
    - Edges of the segment are not considered. 
    - Water polygons are skipped.
"""
def calculateEdgeDensityFields(dataSegmentYear, log):   
    
    # Iterate through the polygons in dataSegmentYear
    accArea       = 0
    accEdgeLength = 0
    for index in dataSegmentYear.index:  
        # Ignore water codes
        try:
            if isWater(dataSegmentYear.loc[index]): continue    
        except Exception as e:
            log.write(str(e))
            continue  
        accArea  = accArea + dataSegmentYear.loc[index].Shape_Area  # area          
        if isCropfield(dataSegmentYear.loc[index]):   
            accEdgeLength = accEdgeLength + dataSegmentYear.loc[index].Shape_Leng  # perimeter           
    if accArea > 0:
        return accEdgeLength / (accArea*1e-4)
    else:
        return 0  


"""
INPUT: 
    - a subset of ESYRCE data corresponding to a segment for a particular year 
    
OUTPUT: edge density of non agricultural and non seminatural areas, defined as 
(see manual FRAGSTATS) the sum of the lengths (m) of all edge segments, 
divided by the total area (hectares).

Notes: 
    - Edges of the segment are not considered. 
    - Water polygons are skipped.
"""
def calculateEdgeDensityOther(dataSegmentYear, log):   
    
    # Iterate through the polygons in dataSegmentYear
    accArea       = 0
    accEdgeLength = 0
    for index in dataSegmentYear.index:         
        # Ignore water codes
        try:
            if isWater(dataSegmentYear.loc[index]): continue    
        except Exception as e:
            log.write(str(e))
            continue  
        accArea  = accArea + dataSegmentYear.loc[index].Shape_Area  # area            
        if isOther(dataSegmentYear.loc[index]):   
            accEdgeLength = accEdgeLength + dataSegmentYear.loc[index].Shape_Leng  # perimeter            
    if accArea > 0:
        return accEdgeLength / (accArea*1e-4)
    else:
        return 0
    

"""
INPUT: a slice of a dataframe from a groupBy operation, corresponding to one segment in ESYRCE data in different years
OUTPUT: the slope of the line derived from a linear regression using the values in each column
"""
def getEvolutionMetrics(segment):
    regr  = linear_model.LinearRegression()
    years = np.array(segment.YEA)
    out={}
    segmMetrics = segment.drop(columns=['D1_HUS','D2_NUM','YEA'])
    for column in segmMetrics:
        yaxis = np.array(segmMetrics[column])
        valid = ~np.isnan(years) & ~np.isnan(yaxis)
        xaxis = years[valid]
        yaxis = yaxis[valid]
        slope = np.nan
        if xaxis.size > 2:
            regr.fit(xaxis.reshape(-1,1), yaxis.reshape(-1,1))
            slope = regr.coef_[0][0]
        out[column+'_slope'] = slope
    return pd.Series(out)


"""
INPUT: a segment from ESYRCE data (squares of 25 or 49 ha) 
OUTPUT: the area of the segment
"""
def calculateSegmentArea(segment):
    totalArea = 0
    for index in segment.index:
        areaPolygon = segment.loc[index].Shape_Area
        totalArea = totalArea + areaPolygon
    return totalArea*1e-4; # m^2 to hectares


"""
INPUT: a segment from ESYRCE data
OUTPUT: the area of the segment ignoring the polygons with water codes
"""
def calculateSegmentAreaWithoutWater(segment, log):
    totalArea = 0
    for index in segment.index: 
        # Ignore water codes
        try:
            if isWater(segment.loc[index]): continue    
        except Exception as e:
            log.write(str(e))
            continue
        areaPolygon = segment.loc[index].Shape_Area
        totalArea = totalArea + areaPolygon
    return totalArea*1e-4; # m^2 to hectares

    
"""
INPUT: directory and extension
OUTPUT: list of files in the directory and subdirs (recursive) with that extension
"""
def allFiles(root, ext):
    files = []
    for x in glob.glob(root+'**\\*.'+ext, recursive=True):
        files.append(x)
    return files

"""
INPUT: array of ESYRCE codes, and dictionaries to relate codes with seminatural and crop classes
OUTPUT: aggregation class-> crop, seminatural, water or other (plus exception)
"""
def getAggregatedClass(dataRow, dictIsSeminatural, dictIsCrop):
    waterCodes = np.array(['AG','MO'])
    d4_grc     = dataRow.D4_GRC
    d5_cul     = dataRow.D5_CUL
    aggClass   = "Other" 
    classSet   = False
    try:
        if (d5_cul[0:2] in dictIsSeminatural):
            if (dictIsSeminatural[d5_cul[0:2]] == "YES"): 
                aggClass = "Seminatural"
                classSet = True
        if (not classSet) & (d5_cul[0:2] in dictIsCrop):
            if (dictIsCrop[d5_cul[0:2]] == "YES"):
                aggClass = "Crop"
                classSet = True
        if (not classSet) & np.isin(d4_grc, waterCodes):
            aggClass = "Water"     
    except:
        return "Exception"
    return aggClass


"""
INPUT: ESYRCE data row
OUTPUT: is water? yes or no
"""
def isWater(data):
    return data.aggClass == "Water"


"""
INPUT: ESYRCE data row
OUTPUT: is classified as seminatural? yes or no
"""
def isSeminatural(data):
    return data.aggClass == "Seminatural"


"""
INPUT: array of ESYRCE codes
OUTPUT: is classified as crop field? yes or no
"""
def isCropfield(data):
    return data.aggClass == "Crop"


"""
INPUT: array of ESYRCE codes
OUTPUT: is classified as other? yes or no
"""
def isOther(data):
    return data.aggClass == "Other"

