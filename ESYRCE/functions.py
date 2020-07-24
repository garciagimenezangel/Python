# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:56:21 2020

@author: angel.gimenez
"""
import numpy as np
import geopandas as gpd
from sklearn import linear_model
from os.path import expanduser
home = expanduser("~")
dictDemandValues = { ''           :     0,
                   'no increase':       0, 
                   'increase':          0.5,
                   'increase-breeding': 0.5,
                   'increase-seed production': 0.5,
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
            log.write("Problems dissolving segment "+str(dataSegmentNr.iloc[0]['D2_NUM']))
            return gpd.GeoDataFrame()
            
        if (cont == 0):  
            intersection = dissolved
            cont = cont+1
        else:
            try:
                intersection = gpd.overlay(intersection, dissolved, how='intersection')
            except:
                log.write("Problems performing intersection "+str(dataSegmentNr.iloc[0]['D2_NUM']))
                return gpd.GeoDataFrame()
    
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
    - dictionary with land cover types that can be associated with other combinations of ESYRCE codes 
    (because, unfortunately, some land cover types can be identified with more than one combination of codes) 
    
OUTPUT: dictionary with the porportion of each land cover type within the segment

Note: water is ignored in the calculations
"""
def calculateLandCoverProportion(dataSegmentYear, landCoverTypes, alternatCodes, log):
    
    # Read codes from dictionary landCoverTypes
    keys     = list(landCoverTypes.keys())
    codes    = list(landCoverTypes.values())
    lcGrc    = np.array([x[0] for x in codes])
    lcCul    = np.array([x[1] for x in codes])
    
    # Read alternative codes
    keysAlt  = list(alternatCodes.keys())
    codesAlt = list(alternatCodes.values())
    lcGrcAlt = np.array([x[0] for x in codesAlt])
    lcCulAlt = np.array([x[1] for x in codesAlt])
    
    # Initialize variables to store accumulated area values 
    lcAcc = np.zeros(len(lcGrc)) # accumulated area of each land cover type
    totalArea = 0
    
    # Ignore water codes
    ignoreGrc = np.array(['AG','MO'])
    ignoreCul = np.array(['AG','MO'])
    
    # Iterate through the polygons in dataSegmentYear
    for index in dataSegmentYear.index:
        # area of the polygon
        areaPolygon = dataSegmentYear.loc[index].Shape_Area
        
        # landcover codes (2 first characteres)
        try:
            polyGrc = dataSegmentYear.loc[index].D4_GRC[0:2]       
            polyCul = dataSegmentYear.loc[index].D5_CUL[0:2]
        except:
            log.write("Problem with land cover codes:"+str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"+str(dataSegmentYear.loc[index].YEA))
            continue            
        
        # Ignore water codes
        if np.isin(polyGrc, ignoreGrc): continue
        if np.isin(polyCul, ignoreCul): continue
    
        # identify landcover index
        ii = np.where(lcGrc == polyGrc)[0]
        if (len(ii)>1): 
            ii = np.where(lcCul == polyCul)[0]
        
        # in case we found more than one compatible index, try with the alternative codes
        if (len(ii) != 1):
            ii = np.where(lcGrcAlt == polyGrc)[0]
            if (len(ii)>1): 
                ii = np.where(lcCulAlt == polyCul)[0] 
            # if index found in the alternative dictionary, get the index in the main dictionary
            if (len(ii)==1):
                ind = ii[0]
                ii = np.where(np.array(keys) == keysAlt[ind])[0]
                ind = ii[0]
            
            else: 
                log.write("Alternative land cover codes not working:"+str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"+str(dataSegmentYear.loc[index].YEA)+
                  "...D4_GRC:"+str(polyGrc)+
                  "...D5_CUL:"+str(polyCul))
                continue
        
        # add area of the land cover type
        if (len(ii)==1): # it should find only one index
            ind = ii[0]
            lcAcc[ind] = lcAcc[ind] + areaPolygon
            totalArea = totalArea + areaPolygon 
            
        else: 
            log.write("Index not found in calculateLandCoverPercentages. Parcel IGNORED"+
                  "...Segment:" +str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"  +str(dataSegmentYear.loc[index].YEA)+
                  "...D4_GRC:"+str(polyGrc)+
                  "...D5_CUL:"+str(polyCul))
            
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
    codes    = list(soilCodes.values())
    codesCS  = np.array([x[0] for x in codes])
    
    # Initialize variables to store accumulated area values 
    soilAcc = np.zeros(len(codesCS)) # accumulated area of each soil management technique
    totalArea = 0
    
    # Iterate through the polygons in dataSegmentYear
    for index in dataSegmentYear.index:
        # area of the polygon
        areaPolygon = dataSegmentYear.loc[index].Shape_Area
        
        # soil management technique code
        polyCS = dataSegmentYear.loc[index].DE_CS       
        
        # Ignore water codes
        if np.isin(polyCS, np.array(list(ignoreCodes.values()))): continue
    
        # If code found (polyCS != None), add area to the corresponding soil management technique
        if polyCS:
    
            # identify soil technique index
            ii = np.where(codesCS == polyCS)[0]
        
            # add area of this soil technique
            if (len(ii)==1): # it should find only one index
                ind = ii[0]
                soilAcc[ind] = soilAcc[ind] + areaPolygon
                totalArea    = totalArea + areaPolygon
            else: 
                log.write("Index not found in calculateSoilTechniqueProportion. Parcel IGNORED"+
                  "...Segment:" +str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"  +str(dataSegmentYear.loc[index].YEA)+
                  "...DE_CS:"+str(polyCS))
                        
    if totalArea != 0:
        values = soilAcc/totalArea
    else:
        values = np.zeros(len(keys))
        
    # Output dictionary. Key: soil management technique; value: accumulated area in the segment
    return dict((keys[ind], values[ind]) for ind in range(0,len(keys)))              


"""
INPUT: 
    - a subset of ESYRCE data corresponding to a segment for a particular year 
    - dictionary to find out whether a given ESYRCE code correspond to a crop or not
    
OUTPUT: average size of the crop fields
"""
def calculateAvgFieldSize(dataSegmentYear, dictIsCrop, log):
    
    # Iterate through the polygons in dataSegmentYear
    accArea     = 0
    nCropfields = 0
    for index in dataSegmentYear.index:  
        # area of the polygon
        areaPolygon = dataSegmentYear.loc[index].Shape_Area

        # landcover codes (2 first characteres)
        try:
            polyGrc = dataSegmentYear.loc[index].D4_GRC[0:2]       
        except:
            log.write("Problem with land cover codes:"+str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"+str(dataSegmentYear.loc[index].YEA))
            continue            
        
        try:
            isCropfield = dictIsCrop[polyGrc] == 'YES'
        except:
            isCropfield = False       

        if isCropfield:   
            nCropfields = nCropfields + 1
            accArea  = accArea + areaPolygon
    
    if nCropfields > 0:
        return accArea / nCropfields  
    else:
        return 0
    

"""
INPUT: 
    - a subset of ESYRCE data corresponding to a segment for a particular year 
    - dictionary to find out whether a given ESYRCE code correspond to a crop or not
    
OUTPUT: number of crop types, per km^2
"""
def calculateHeterogeneity(dataSegmentYear, dictIsCrop, log):   
    
    # Ignore water codes for the total area
    ignoreGrc = np.array(['AG','MO'])
    
    # Iterate through the polygons in dataSegmentYear
    crops     = []
    totalArea = 0
    for index in dataSegmentYear.index:      
        # area of the polygon
        areaPolygon = dataSegmentYear.loc[index].Shape_Area
        
        # landcover codes (2 first characteres)
        try:
            polyGrc = dataSegmentYear.loc[index].D4_GRC[0:2]       
        except:
            log.write("Problem with land cover codes:"+str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"+str(dataSegmentYear.loc[index].YEA))
            continue            
        
        # Ignore water codes
        if np.isin(polyGrc, ignoreGrc): continue
        
        try:
            isCropfield = dictIsCrop[polyGrc] == 'YES'
        except:
            isCropfield = False    

        if isCropfield:   
            crops = np.append(crops, dataSegmentYear.loc[index].D5_CUL)
        
        # add up area of the polygon (convert m^2 into km^2)
        totalArea = totalArea + areaPolygon*1e-6

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
    
    # Ignore water codes for the total area
    ignoreCul = np.array(['AG','MO'])

    # Iterate through the polygons in dataSegmentYear
    totalArea = 0
    accDemand = 0
    for index in dataSegmentYear.index:  
        # area of the polygon
        areaPolygon = dataSegmentYear.loc[index].Shape_Area
                  
        # Ignore water codes
        polyCul = dataSegmentYear.loc[index].D5_CUL
        if np.isin(polyCul, ignoreCul): continue 
       
        # Calculate demand. If association of cultivars, calculate average
        try:
            assocElts = polyCul.split("-")
        except:
            log.write("Problem with land cover codes:"+str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"+str(dataSegmentYear.loc[index].YEA))
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
def calculateCropYield(dataSegmentYear, cropCodes, log):
    # Read codes from dictionary landCoverTypes
    keys     = list(cropCodes.keys())
    codes    = list(cropCodes.values())
    lcCul    = np.array([x[0] for x in codes])
    yieldVal = np.repeat(np.nan, len(keys)) 

    # Initialize variables to store accumulated yields and areas
    yieldAcc = np.zeros(len(lcCul)) # accumulated weighted yield (by area) of each crop type
    areaAcc  = np.zeros(len(lcCul)) # accumulated area of each crop type
    
    # Iterate through the polygons in dataSegmentYear
    for index in dataSegmentYear.index:
        # area of the polygon
        areaPolygon = dataSegmentYear.loc[index].Shape_Area
        
        # yield (if 0 or None, ignore segment)
        fieldYield = dataSegmentYear.loc[index].D9_RTO
        if(fieldYield is None): continue
        if(fieldYield == 0):    continue 
            
        # landcover codes (2 first characteres)
        try:
            polyCul = dataSegmentYear.loc[index].D5_CUL[0:2]
        except:
            log.write("Problem with land cover codes:"+str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"+str(dataSegmentYear.loc[index].YEA))
            continue            
    
        # identify landcover index
        ii = np.where(lcCul == polyCul)[0]
        
        # accumulate weighted yield and area
        if (len(ii)==1): # it should find only one index
            ind = ii[0]
            yieldAcc[ind] = yieldAcc[ind] + fieldYield*areaPolygon
            areaAcc[ind]  = areaAcc[ind]  + areaPolygon
            
        else: 
            log.write("Index not found in calculateCropYield. Parcel IGNORED"+
                  "...Segment:" +str(dataSegmentYear.loc[index].D2_NUM)+
                  "...Parcel:"+str(dataSegmentYear.loc[index].D3_PAR)+
                  "...Year:"  +str(dataSegmentYear.loc[index].YEA)+
                  "...D5_CUL:"+str(polyCul))
            
    for ind in range(0,len(keys)):
        if (areaAcc[ind] > 0): yieldVal[ind] = yieldAcc[ind]/areaAcc[ind]
        
    # Output dictionary. Key: crop type; value: weighted average of the yield
    return dict((keys[ind], yieldVal[ind]) for ind in range(0,len(keys)))          


"""
INPUT: a slice of a dataframe from a groupBy operation, corresponding to one segment number in ESYRCE data 
OUTPUT: the slope of the line derived from a linear regression using the values in each column
"""
def calculateSlopes(df):
    regr = linear_model.LinearRegression()
    slopes = np.array([])
    for index in df.index:
        X = np.array(range(2001,2017), dtype=int)
        Y = np.array(df.loc[index][1:17], dtype=float)
        valid = np.invert(np.isnan(Y))
        X = X[valid]
        Y = Y[valid]
        slope = np.nan
        if X.size > 2:
            regr.fit(X.reshape(-1,1),Y.reshape(-1,1))
            slope = regr.coef_[0][0]
        slopes = np.append(slopes, slope)   
    df['slope'] = slopes
    return df


"""
INPUT: a segment from ESYRCE data (squares of 700x700m or 500x500m) 
OUTPUT: the area of the segment
"""
def getSegmentArea(segment):
    totalArea = 0
    for index in segment.index:
        areaPolygon = segment.loc[index].Shape_Area
        totalArea = totalArea + areaPolygon
    return totalArea*1e-6; # m^2 to km^2
