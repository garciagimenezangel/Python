Scripts for processing ESYRCE data with the goal of creating models in k.LAB:

- filterDataByExtent.py: filter data by spatial extent

- mergeSplitted.py: merge data files produced by 'filterDataByExtent.py'

- filterColumns.py: filter some columns, in case some "pruning" of the data is desired

- processESYRCEcodes.py: process the ESYRCE land cover codes 

- splitByYear.py: split data into shapefiles by year

- addIntensificationMetrics.py: add columns with metrics of agricultural intensification

- addDemand.py: add column with demand of pollinators according to the crop

- dissolveBlocks.py: dissolve geometries of the blocks to simplify the data (to use after splitting by year and after adding demand and intensification metrics)

- divideFieldByAreaBlock.py: divide a field in the data by the area of the block

Notes: 

- CRS must be 'EPSG:23028' for layer z28 and 'EPSG:23030' for z30

- Some blocks (noticed in the data of 2015) need some cleaning of geometries in order to be able to apply the dissolve operation. This cleaning is performed in QGIS using Processing>Delete holes

- After applying all or some of these scripts to the ESYRCE data, other additional operations might still be necessary to create the rasters for its ingestion in k.LAB. The following operations were quite straigthforward to do in QGIS, so I didn't implement them in Python: 
	- create centroids of the (dissolved) polygons
	- interpolate centroids using Grid(Linear)
	- mask the interpolated rasters using administrative boundaries and some additional clipping to avoid areas where no points are available

