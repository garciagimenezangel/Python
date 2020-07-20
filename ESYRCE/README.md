Scripts for processing and analyzing ESYRCE data:

Main files:

- functions.py: important functions used in the scripts (e.g. to calculate metrics) are stored in this file.

- flagDataSegments.py: flag data segments. 0:ok; 1:size changed, clip to smallest; 2: data not aligned.

- addMetrics.py: create csv with metrics associated to each segment, such as area of land cover types, field size or demand.

- addSlopeMetrics.py: create csv with the slope of each metric calculated using 'addMetrics.py'.


Other scripts (ancillary files, not needed for the main pipeline, and most likely outdated and useless, unless some updates are done):

- mergeSplitted.py: merge data files produced by 'flagDatasegments.py'

- splitByYear.py: split data into shapefiles by year

- dissolveSegments.py: dissolve geometries of the blocks to simplify the data (to use after adding metrics and splitting by year, but see notes below)

- calculateEvolutionMetrics.py: calculate time series of metrics

- plotTimeSeries: plot histograms

- clusterEvolutionMetrics.py: get trajectory types for the evolution of the metrics

- filterColumns.py: filter some columns, in case some "pruning" of the data is desired

- processESYRCEcodes.py: process the ESYRCE land cover codes to get main and complementary codes

- divideFieldByAreaSegment.py: divide a field in the data by the area of the segment

- rasterize.py: rasterize shapefile


Notes: 

- CRS must be 'EPSG:23028' for layer z28 and 'EPSG:23030' for z30

- The position of the segments can be derived from the zone and number of the segment (D1_HUS and D2_NUM). 
For example, the segment with D1_HUS=29 and D2_NUM=4824756 corresponds to the segment that has its left-bottom corner at coordinates (with 1km precision),
approximately equal to [482000, 4756000], using the coordinate system EPSG:23029 (D1_HUS=29, change the last two digits of the EPSG code accordingly)

