Data processing pipeline:

1: Filter data by spatial extent: run filterDataByExtent.py, for layers z28 and z30

2: Process the ESYRCE land cover codes, so that they can be ingested by k.LAB: run processESYRCEcodes.py

3: Split the shapefiles by year: run splitByYear.py

4: Go to k.LAB and annotate the data

CRS must be 'EPSG:32628' for layer z28 and 'EPSG:32630' for z30

