Scripts for processing and analyzing ESYRCE data:

Main files:

- functions.py: important functions used in the scripts (e.g. to calculate metrics) are stored in this file

- filterDataByExtent.py: filter data by spatial extent

- mergeSplitted.py: merge data files produced by 'filterDataByExtent.py'

- addMetrics.py: add columns with metrics associated to each block, such as area of land cover types, field size or demand.

- splitByYear.py: split data into shapefiles by year

- dissolveBlocks.py: dissolve geometries of the blocks to simplify the data (to use after adding metrics and splitting by year, but see notes below)

- calculateEvolutionMetrics.py: calculate time series of metrics


Other scripts (ancillary files, not needed for the main pipeline):

- plotTimeSeries: plot histograms

- clusterEvolutionMetrics.py: get trajectory types for the evolution of the metrics

- filterColumns.py: filter some columns, in case some "pruning" of the data is desired

- processESYRCEcodes.py: process the ESYRCE land cover codes to get main and complementary codes

- divideFieldByAreaBlock.py: divide a field in the data by the area of the block

- rasterize.py: rasterize shapefile


Notes: 

- CRS must be 'EPSG:23028' for layer z28 and 'EPSG:23030' for z30

- Some blocks (noticed in the data of 2015) need some cleaning of geometries in order to be able to apply the dissolve operation. Therefore the dissolve operation and the cleaning for that year (2015) is performed in QGIS (cleaning using Processing>Delete holes)

