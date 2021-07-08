import rasterio
import glob
from dask_rasterio import read_raster, write_raster
import dask.array as da

earthstat_dir  = "C:/Users/angel/DATA/Earthstat/HarvestedAreaYield175Crops_Geotiff/HarvestedAreaYield175Crops_Geotiff/"
layer = "Production"
ext = ".tif"
selected_files = [file for file in glob.iglob(earthstat_dir + '**/*' + layer + ext, recursive=True)]
map2array=[]
for raster in selected_files:
    map2array.append(read_raster(raster))

ds_stack = da.stack(map2array)
with rasterio.open(selected_files[0]) as src:
    profile = src.profile
    profile.update(compress='lzw')

write_raster(earthstat_dir + "Sum" + layer + ".tif", da.nansum(ds_stack,0), **profile)

