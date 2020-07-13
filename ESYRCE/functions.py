# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 16:56:21 2020

@author: angel.gimenez
"""
import numpy as np
import geopandas as gpd
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
INPUT: subset of ESYRCE data corresponding to one block number
OUTPUT: polygon corresponding to the data of the year with the smallest spatial coverage
"""
def getPolygonToClip(dataBlockNr):
    years = np.unique(dataBlockNr.YEA)
    cont = 0
    for year in years:
#        selectedInd   = dataBlockNr.YEA == year
#        dataBlockYear = [dataBlockNr.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
#        dataBlockYear = gpd.GeoDataFrame(dataBlockYear)
        ii = np.where(dataBlockNr.YEA == year)
        i0 = ii[0][0]
        iM = ii[0][len(ii[0])-1]
        dataBlockYear = dataBlockNr[i0:(iM+1)]
        try:
            dissolved = dataBlockYear.dissolve(by='YEA')    
        except:
            print("Warning (getPolygonToClip): problems dissolving block ", dataBlockNr.iloc[0]['D2_NUM'])
            return gpd.GeoDataFrame()
            
        if (cont == 0):  
            intersection = dissolved
            cont = cont+1
        else:
            try:
                intersection = gpd.overlay(intersection, dissolved, how='intersection')
            except:
                print("Warning: problems performing intersection ", dataBlockNr.iloc[0]['D2_NUM'])
                return gpd.GeoDataFrame()
    
    return gpd.GeoDataFrame(intersection.geometry)


"""
INPUT: subset of ESYRCE data corresponding to one block number
OUTPUT: quality flag. 0:ok; 1:size changed, clip to smallest; 2: blocks not aligned; 3: geometry problems dissolving
"""
def getBlockQualityFlag(dataBlockNr, tol):
    flag = 0 
    years = np.unique(dataBlockNr.YEA)
    cont = 0
    for year in years:
#        selectedInd   = dataBlockNr.YEA == year
#        dataBlockYear = [dataBlockNr.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
#        dataBlockYear = gpd.GeoDataFrame(dataBlockYear)
        ii = np.where(dataBlockNr.YEA == year)
        i0 = ii[0][0]
        iM = ii[0][len(ii[0])-1]
        dataBlockYear = dataBlockNr[i0:(iM+1)]
        try:
            dissolved     = dataBlockYear.dissolve(by='YEA')    
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
            if (not(np.isclose(lastMinX, newMinX, rtol=tol)) or not(np.isclose(lastMinY, newMinY, rtol=tol))):
                return 2
            elif (not(np.isclose(lastMaxX, newMaxX, rtol=tol)) or not(np.isclose(lastMaxY, newMaxY, rtol=tol))):
                flag = 1
                minBBox = newBBox
            else:
                minBBox = newBBox
                
    return flag
        

"""
INPUT: 
    - a subset of ESYRCE data corresponding to a block for a particular year 
    - dictionary of land cover types with associated ESYRCE codes
    - dictionary with land cover types that can be associated with other combinations of ESYRCE codes 
    (because, unfortunately, some land cover types can be identified with more than one combination of codes) 
    
OUTPUT: dictionary with the porportion of each land cover type within the block

Note: water is ignored in the calculations
"""
def calculateLandCoverProportion(dataBlockYear, landCoverTypes, alternatCodes):
    
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
    
    # Iterate through the polygons in dataBlockYear
    for index in dataBlockYear.index:
        # area of the polygon
        areaPolygon = dataBlockYear.loc[index].Shape_Area
        
        # landcover codes (2 first characteres)
        polyGrc = dataBlockYear.loc[index].D4_GRC[0:2]       
        polyCul = dataBlockYear.loc[index].D5_CUL[0:2]
        
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
                print("Warning... ALTERNATIVE NOT WORKING:", dataBlockYear.loc[index].D2_NUM,
                  "...Parcel:", dataBlockYear.loc[index].D3_PAR,
                  "...Year:", dataBlockYear.loc[index].YEA,
                  "...D4_GRC:", polyGrc,
                  "...D5_CUL:", polyCul)
                continue
        
        # add area of the land cover type
        if (len(ii)==1): # it should find only one index
            ind = ii[0]
            lcAcc[ind] = lcAcc[ind] + areaPolygon
            totalArea = totalArea + areaPolygon 
            
        else: 
            print("Warning... Index not found in calculateLandCoverPercentages. Parcel IGNORED",
                  "...Block:", dataBlockYear.loc[index].D2_NUM,
                  "...Parcel:", dataBlockYear.loc[index].D3_PAR,
                  "...Year:", dataBlockYear.loc[index].YEA,
                  "...D4_GRC:", polyGrc,
                  "...D5_CUL:", polyCul)
            
    if totalArea != 0:
        values = lcAcc/totalArea
    else:
        values = np.zeros(len(keys))
        
    # Output dictionary. Key: land cover type; value: accumulated area in the block
    return dict((keys[ind], values[ind]) for ind in range(0,len(keys)))          

    
"""
INPUT: 
    - a subset of ESYRCE data corresponding to a block for a particular year 
    - dictionary of soil techniques with associated ESYRCE codes. 
    - dictionary of soil techniques to be ignored (in order to use the same function for soil maintenance and sowing techniques). 
    
OUTPUT: dictionary with the porportion of each soil management technique within the block

Note: proportion is computed only with regard to codes present in the input dictionary. If no code present, then returns 0.
"""
def calculateSoilTechniqueProportion(dataBlockYear, soilCodes, ignoreCodes):
    # Read codes from dictionary landCoverTypes
    keys     = list(soilCodes.keys())
    codes    = list(soilCodes.values())
    codesCS  = np.array([x[0] for x in codes])
    
    # Initialize variables to store accumulated area values 
    soilAcc = np.zeros(len(codesCS)) # accumulated area of each soil management technique
    totalArea = 0
    
    # Iterate through the polygons in dataBlockYear
    for index in dataBlockYear.index:
        # area of the polygon
        areaPolygon = dataBlockYear.loc[index].Shape_Area
        
        # soil management technique code
        polyCS = dataBlockYear.loc[index].DE_CS       
        
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
                print("Warning... Index problems in calculateSoilTechniqueProportion. Parcel IGNORED",
                  "...Block:", dataBlockYear.loc[index].D2_NUM,
                  "...Parcel:", dataBlockYear.loc[index].D3_PAR,
                  "...Year:", dataBlockYear.loc[index].YEA,
                  "...DE_CS:", polyCS)
                        
    if totalArea != 0:
        values = soilAcc/totalArea
    else:
        values = np.zeros(len(keys))
        
    # Output dictionary. Key: soil management technique; value: accumulated area in the block
    return dict((keys[ind], values[ind]) for ind in range(0,len(keys)))              


"""
INPUT: 
    - a subset of ESYRCE data corresponding to a block for a particular year 
    - dictionary to find out whether a given ESYRCE code correspond to a crop or not
    
OUTPUT: average size of the crop fields
"""
def calculateAvgFieldSize(dataBlockYear, dictIsCrop):
    
    # Iterate through the polygons in dataBlockYear
    accArea     = 0
    nCropfields = 0
    for index in dataBlockYear.index:  
        # area of the polygon
        areaPolygon = dataBlockYear.loc[index].Shape_Area

        # landcover codes (2 first characteres)
        polyGrc = dataBlockYear.loc[index].D4_GRC[0:2]       
        
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
    - a subset of ESYRCE data corresponding to a block for a particular year 
    - dictionary to find out whether a given ESYRCE code correspond to a crop or not
    
OUTPUT: number of crop types, per km^2
"""
def calculateHeterogeneity(dataBlockYear, dictIsCrop):   
    
    # Ignore water codes for the total area
    ignoreGrc = np.array(['AG','MO'])
    
    # Iterate through the polygons in dataBlockYear
    crops     = []
    totalArea = 0
    for index in dataBlockYear.index:      
        # area of the polygon
        areaPolygon = dataBlockYear.loc[index].Shape_Area
        
        # land cover code (2 first characteres)
        polyGrc = dataBlockYear.loc[index].D4_GRC[0:2] 
        
        # Ignore water codes
        if np.isin(polyGrc, ignoreGrc): continue
        
        try:
            isCropfield = dictIsCrop[polyGrc] == 'YES'
        except:
            isCropfield = False    

        if isCropfield:   
            crops = np.append(crops, dataBlockYear.loc[index].D5_CUL)
        
        # add up area of the polygon (convert m^2 into km^2)
        totalArea = totalArea + areaPolygon*1e-6

    if totalArea > 0:
        return len(np.unique(crops)) / totalArea
    else:
        return 0
 
    
"""
INPUT: 
    - a subset of ESYRCE data corresponding to a block for a particular year 
    - dictionary with indicators of demand for the crops 
    
OUTPUT: demand value for the block, using an average weighted by the area of the polygons 
"""
def calculateDemand(dataBlockYear, dictCultivarDemand):
    
    # Ignore water codes for the total area
    ignoreCul = np.array(['AG','MO'])

    # Iterate through the polygons in dataBlockYear
    totalArea = 0
    accDemand = 0
    for index in dataBlockYear.index:  
        # area of the polygon
        areaPolygon = dataBlockYear.loc[index].Shape_Area
                  
        # Ignore water codes
        polyCul = dataBlockYear.loc[index].D5_CUL
        if np.isin(polyCul, ignoreCul): continue 
       
        # Calculate demand. If association of cultivars, calculate average
        assocElts = polyCul.split("-")
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
INPUT: a block from ESYRCE data (squares of 700x700m or 500x500m) 
OUTPUT: the area of the block
"""
def getBlockArea(block):
    totalArea = 0
    for index in block.index:
        areaPolygon = block.loc[index].Shape_Area
        totalArea = totalArea + areaPolygon
    return totalArea*1e-6; # m^2 to km^2
