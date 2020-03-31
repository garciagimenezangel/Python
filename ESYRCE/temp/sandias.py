# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 09:04:55 2020

@author: angel.gimenez
"""
import dill   
#import geopandas as gpd
import fiona

# files                 
session = '.\\Documents\\REPOSITORIES\\Python\\ESYRCE\\gdbLayerZ30_LoadedSession.pkl'
sandias_shp = '.\\Documents\\REPOSITORIES\\Python\\ESYRCE\\Shapefiles\\sandias.shp'

# load session
dill.load_session(session)

# rendimiento > 0
rtoGt0 = mallard["D9_RTO"] > 0

# sandias
sa = mallard["D5_CUL"] == "SA"

sa2 = mallard[rtoGt0]
sandias = sa2[sa]

# export
sandias.to_file(filename=sandias_shp, driver="ESRI Shapefile")

ind2001 = mallard["YEA"] == 2001
ind2002 = mallard["YEA"] == 2002
ind2003 = mallard["YEA"] == 2003
ind2004 = mallard["YEA"] == 2004
ind2005 = mallard["YEA"] == 2005
ind2006 = mallard["YEA"] == 2006
ind2007 = mallard["YEA"] == 2007
ind2008 = mallard["YEA"] == 2008
ind2009 = mallard["YEA"] == 2009
ind2010 = mallard["YEA"] == 2010
ind2011 = mallard["YEA"] == 2011
ind2012 = mallard["YEA"] == 2012
ind2013 = mallard["YEA"] == 2013
ind2014 = mallard["YEA"] == 2014
ind2015 = mallard["YEA"] == 2015
ind2016 = mallard["YEA"] == 2016

data2001 = mallard[ind2001]
data2002 = mallard[ind2002]
data2003 = mallard[ind2003]
data2004 = mallard[ind2004]
data2005 = mallard[ind2005]
data2006 = mallard[ind2006]
data2007 = mallard[ind2007]
data2008 = mallard[ind2008]
data2009 = mallard[ind2009]
data2010 = mallard[ind2010]
data2011 = mallard[ind2011]
data2012 = mallard[ind2012]
data2013 = mallard[ind2013]
data2014 = mallard[ind2014]
data2015 = mallard[ind2015]
data2016 = mallard[ind2016]

data2001.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2001.shp", driver="ESRI Shapefile")
data2002.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2002.shp", driver="ESRI Shapefile")
data2003.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2003.shp", driver="ESRI Shapefile")
data2004.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2004.shp", driver="ESRI Shapefile")
data2005.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2005.shp", driver="ESRI Shapefile")
data2006.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2006.shp", driver="ESRI Shapefile")
data2007.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2007.shp", driver="ESRI Shapefile")
data2008.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2008.shp", driver="ESRI Shapefile")
data2009.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2009.shp", driver="ESRI Shapefile")
data2010.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2010.shp", driver="ESRI Shapefile")
data2011.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2011.shp", driver="ESRI Shapefile")
data2012.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2012.shp", driver="ESRI Shapefile")
data2013.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2013.shp", driver="ESRI Shapefile")
data2014.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2014.shp", driver="ESRI Shapefile")
data2015.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2015.shp", driver="ESRI Shapefile")
data2016.to_file(filename=".\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Shapefiles\\Yearly\\esyrce2016.shp", driver="ESRI Shapefile")



