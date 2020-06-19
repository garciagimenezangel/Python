# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import dill
import geopandas as gpd
import numpy as np

from os.path import expanduser
home = expanduser("~")

# INPUT
dissolve = False # dissolve polygons in each block? 
layer = "z30"
#inputFile = home + '\\Documents\\DATA\\OBServ\\LandCover\\ESYRCE\\PROCESSED\\esyrceProcessed_' + layer + '.shp'
#inputFile = home + '\\Documents\\DATA\\Observ\\LandCover\\ESYRCE\\PROCESSED\\session_esyrceFiltered_z30_epsg23030_selectedCols_addIntenMetrics_addDemand.pkl'
inputFile = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\Esyrce2001_2016.gdb'
outDir = home + '\\Documents\\DATA\\OBServ\\ESYRCE\\PROCESSED\\z30\\yearly\\'

if layer == 'z28':
    crs = "EPSG:23028"

if layer == 'z30':
    crs = "EPSG:23030"
    
# Read file
#processedData = gpd.read_file(inputFile)
#dill.load_session(inputFile) # data in dataSel
#data = dataSel
data = gpd.read_file(inputFile, layer=layer)
   
# To files, by year
years = np.unique(data.YEA)
for year in years:
    selectedInd   = data.YEA == year
    validDataYear = [data.iloc[i] for i in range(0,len(selectedInd)) if selectedInd.iloc[i]]
    validDataYear = gpd.GeoDataFrame(validDataYear)
    if (dissolve):
        try:
            validDataYear = validDataYear.dissolve(by='D2_NUM')
            shapefile = outDir+'dissolved\\'+str(year)+".shp"
        except:
            print("Warning: dissolve in year "+str(year)+" failed...")
            shapefile = outDir+str(year)+".shp"
    else:
        shapefile = outDir+str(year)+".shp"
    validDataYear.crs = crs;
    validDataYear.to_file(filename = shapefile, driver="ESRI Shapefile")
    print("Processed year... Saved file: "+shapefile)
    
