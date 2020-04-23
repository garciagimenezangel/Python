Data processing for k.LAB:

1: Process the ESYRCE land cover codes, so that they can be ingested by k.LAB: run processESYRCEcodes.py

2: Split the shapefiles by year: run splitByYear.py

3: Rasterize

4: Go to k.LAB and annotate the data

CRS must be 'EPSG:32628' for layer z28 and 'EPSG:32630' for z30

