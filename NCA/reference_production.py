import pandas as pd
import numpy as np

production_faostat = pd.read_csv("data/faostat.crop.pollination.csv")

dictionary_crops = {
    "apple"     :"agriculture:Apple",
    "melon"     :"agriculture:Melon",
    "peach"     :"agriculture:Peach",
    "pear"      :"agriculture:Pear",
    "plum"      :"agriculture:Plum",
    "watermelon":"agriculture:Watermelon",
    "cucumber"  :"agriculture:Cucumber",
    "pumpkin"   :"agriculture:Pumpkin",
    "cocoa_bean":"agriculture:Cocoa",
    "mango"     :"agriculture:Mango",
    "avocado"   :"agriculture:Avocado"
}

def mean_1997_2003(d):
    selected_years = d.loc[ (d.Year >= 1997) & (d.Year <= 2003),]
    selected_years = selected_years.drop(columns=['country_extended','GID'])
    selected_years = selected_years.replace(" ", "NaN")
    selected_years = np.mean(selected_years.astype(float))
    return selected_years

def clean_columns(d, dict):
    clean_d = d.drop(columns=['country_extended','Year'])
    clean_d.rename(columns=dict, inplace=True)
    sel_columns = list(dictionary_crops.values())
    sel_columns.append('GID')
    clean_d = clean_d.loc[:, clean_agg.columns.isin(sel_columns)]
    return clean_d.replace(np.nan, 0)

country_agg = production_faostat.groupby(['GID']).agg(mean_1997_2003).reset_index()
clean_agg = clean_columns(country_agg, dictionary_crops)

clean_agg.to_csv("data/reference.country.production.csv", index=False)
