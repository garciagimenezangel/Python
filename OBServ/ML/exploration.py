import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Repository
modelsRepo    = "C:/Users/angel/git/Observ_models/"
fieldRepo     = "C:/Users/angel/git/OBservData/"

# Folders
featuresDir   = modelsRepo + "data/GEE/GEE features/"
fieldDataDir  = fieldRepo + "Final_Data/"

if __name__ == '__main__':
    df_features = pd.read_csv(featuresDir+'Features.csv')
    df_field    = pd.read_csv(fieldDataDir+'CropPol_field_level_data.csv')

    #######################################
    # Add other useful columns
    #######################################
    # Comparable abundance wildbees
    df_field['comp_ab_wb']         = (df_field['ab_wildbees']+1) / (df_field['total_sampled_time'] * df_field['total_sampled_area'])
    df_field['comp_ab_syr']        = (df_field['ab_syrphids']+1) / (df_field['total_sampled_time'] * df_field['total_sampled_area'])
    df_field['comp_ab_bmb']        = (df_field['ab_bombus']+1)   / (df_field['total_sampled_time'] * df_field['total_sampled_area'])
    df_field['comp_ab_wb_syr']     = df_field['comp_ab_wb'] + df_field['comp_ab_syr']
    df_field['comp_ab_wb_bmb_syr'] = df_field['comp_ab_wb_syr'] + df_field['comp_ab_bmb']

    # Comp abundance syrphids
    # Comp abundance bombus
    # Comp abundance wb+syr
    # Relative importance guilds large vs small
    # Combined models using relative importance of guilds large and small
