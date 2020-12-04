
#Import necessary libraries to add geospatial attributes to countries
import pandas as pd
import geopandas as gpd
import glob
import os



export=fao_csv[['Area','Country']]
export.to_csv('fao/country_shortName.csv',index=False)


#https://geopandas.org/reference/geopandas.read_file.html
raw_countries=gpd.read_file('../../World_Countries/countries_border.shp')

raw_countries.head()

### Crop Model SHP File
crops_list=glob.glob('crop_model_from_fao_v2/*csv')
len(crops_list)
shp_folder='crop_model_from_fao_v2/crop_model_shp/'

shp_folder='crop_model_from_fao_v2/crop_model_shp/"
    for crop in crops_list"
        crop_csv=pd.read_csv(crop)"
    #drop China,
    #crop_csv=crop_csv[crop_csv['Area Code']!=351]
        shp_name=os.path.basename(crop).split('.')[0]",
    den=pd.merge(raw_countries,crop_csv,how='inner', left_on='GID_0', right_on='Country')
    den.to_file(f'{shp_folder+shp_name}.shp',driver='ESRI Shapefile'

shp_count=glob.glob('crop_model_from_fao_v2/crop_model_shp/*.shp')
shp_count=sorted(shp_count)
len(shp_count)
shp_count[0]

trgt_shp.crs

alt=trgt_shp.to_crs(\"EPSG:4326\")

alt.dtypes


### Pollination Model SHP File Process

polination_list=glob.glob('polinatior_model_from_fao_v2/*csv')
len(polination_list)

str(os.path.basename(polination_list[0]).split(\".\")[0])

shp_folder='polinatior_model_from_fao_v2/polination_shp/'
    for crop in polination_list:
        crop_csv=pd.read_csv(crop)
        shp_name=str(os.path.basename(crop).split(\".\")[0])
        den=pd.merge(raw_countries,crop_csv,how='inner', left_on='GID_0', right_on='Country')
        den.to_file(f'{shp_folder+shp_name}.shp',driver='ESRI Shapefile')

shp_count2=glob.glob('polinatior_model_from_fao/polination_shp/*shp')
len(shp_count2)