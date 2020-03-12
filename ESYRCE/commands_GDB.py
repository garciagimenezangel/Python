# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import geopandas as gpd

# file from local path
mallard = gpd.read_file('.\\Google Drive\\PROJECTS\\OBSERV\\Data\\Land cover\\ESYRCE\\Esyrce2001_2016.gdb', layer="z30")
rtogt0 = mallard["D9_RTO"] > 0
data_with_RTO = mallard[rtogt0]

world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
westhem = world[(world['continent'] == 'Europe')]

# making sure the coordinates line up:
data_with_RTO = data_with_RTO.to_crs(world.crs)

# establishing figure axes
base = westhem.plot(color='white', edgecolor='black', figsize=(11, 11))

# plot
data_with_RTO.plot(ax=base, color='red', alpha=.5)

data_with_RTO.to_file(filename="result.shp", driver="ESRI Shapefile")

import fiona
fiona.supported_drivers