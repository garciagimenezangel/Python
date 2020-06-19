Scripts for processing and analyzing ESYRCE data:

Main scripts:

- filterDataByExtent.py: filter data by spatial extent

- mergeSplitted.py: merge data files produced by 'filterDataByExtent.py'

- addMetrics.py: add columns with metrics associated to each block, such as area of land cover types, field size or demand.

- dissolveBlocks.py: dissolve geometries of the blocks to simplify the data (to use after splitting by year and adding metrics)

- calculateEvolutionMetrics.py: calculate time series of metrics

Other scripts:

- splitByYear.py: split data into shapefiles by year

- filterColumns.py: filter some columns, in case some "pruning" of the data is desired

- processESYRCEcodes.py: process the ESYRCE land cover codes to get main and complementary codes

- divideFieldByAreaBlock.py: divide a field in the data by the area of the block

Notes: 

- CRS must be 'EPSG:23028' for layer z28 and 'EPSG:23030' for z30

- Some blocks (noticed in the data of 2015) need some cleaning of geometries in order to be able to apply the dissolve operation. Therefore the dissolve operation and the cleaning for that year (2015) is performed in QGIS (cleaning using Processing>Delete holes)

