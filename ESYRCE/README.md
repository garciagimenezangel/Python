Data processing pipeline:

1: Filter data by spatial extent: run filterDataByExtent.py, for layers z28 and z30

2: Go to QGIS and save the shapefile with the corresponding CRS: EPSG:32628 or EPSG:32630

3: Process the ESYRCE land cover codes, so that they can be ingested by k.LAB: run processESYRCEcodes.py

3: Split the shapefiles by year: run splitByYear.py

4: Go to k.LAB and annotate the data
