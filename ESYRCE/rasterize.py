# -*- coding: utf-8 -*-
"""
Created on Fri Apr  3 16:02:48 2020

@author: angel.gimenez
"""
import gdal
import ogr
import osr

def rasterize(shapefile, epsg, rasterfile, field):

    #making the shapefile as an object.
    input_shp = ogr.Open(shapefile)

    #getting layer information of shapefile.
    shp_layer = input_shp.GetLayer()

    #pixel_size determines the size of the new raster.
    #pixel_size is proportional to size of shapefile.
    pixel_size = 10

    #get extent values to set size of output raster.
    x_min, x_max, y_min, y_max = shp_layer.GetExtent()

    #calculate size/resolution of the raster.
    x_res = int((x_max - x_min) / pixel_size)
    y_res = int((y_max - y_min) / pixel_size)

    #get GeoTiff driver by 
    image_type = 'GTiff'
    driver = gdal.GetDriverByName(image_type)

    #passing the filename, x and y direction resolution, no. of bands, new raster.
    new_raster = driver.Create(rasterfile, x_res, y_res, 1, gdal.GDT_Byte)

    #transforms between pixel raster space to projection coordinate space.
    new_raster.SetGeoTransform((x_min, pixel_size, 0, y_min, 0, pixel_size))

    #get required raster band.
    band = new_raster.GetRasterBand(1)

    #assign no data value to empty cells.
    no_data_value = 0
    band.SetNoDataValue(no_data_value)
    band.FlushCache()

    #adding a spatial reference
    new_rasterSRS = osr.SpatialReference()
    new_rasterSRS.ImportFromEPSG(epsg)
    new_raster.SetProjection(new_rasterSRS.ExportToWkt())
    
    #main conversion method
    OPTIONS = ['ATTRIBUTE=' + field]
    gdal.RasterizeLayer(new_raster, [1], shp_layer, None, options=OPTIONS)
    return 