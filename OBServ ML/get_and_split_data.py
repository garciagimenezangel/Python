import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.model_selection import train_test_split
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

def get_feature_data():
    models_repo = "C:/Users/angel/git/Observ_models/"
    featuresDir = models_repo + "data/GEE/GEE features/"
    df_features = pd.read_csv(featuresDir + 'Features.csv')
    df_features.drop(columns=['system:index', '.geo', 'refYear'], inplace=True)
    return apply_minimum_conditions_features(df_features)

def get_field_data():
    field_repo    = "C:/Users/angel/git/OBservData/"
    field_data_dir = field_repo + "Final_Data/"
    df_field     = pd.read_csv(field_data_dir+'CropPol_field_level_data.csv')
    return apply_minimum_conditions_field(df_field)

def apply_minimum_conditions_field(data):
    # Conditions:
    # 1. Abundances must be integer numbers (tolerance of 0.05)
    decimal_wb  = (data['ab_wildbees'] % 1)
    decimal_syr = (data['ab_syrphids'] % 1)
    decimal_bmb = (data['ab_bombus'] % 1)
    cond1 = ((decimal_wb < 0.05) & (decimal_syr < 0.05) & (decimal_bmb < 0.05)) | \
            ((decimal_wb > 0.95) & (decimal_syr > 0.95) & (decimal_bmb > 0.95))
    cond1[cond1.isna()] = False
    # 2. Latitude and longitude must be !na (otherwise there will not be model values anyway)
    cond2 = (~data['latitude'].isna()) & (~data['longitude'].isna())
    # 3. Strictly positive abundances
    cond3 = (~data['ab_wildbees'].isna() & data['ab_wildbees'] > 0) | \
            (~data['ab_syrphids'].isna() & data['ab_syrphids'] > 0) | \
            (~data['ab_bombus'].isna() & data['ab_bombus'] > 0)
    # 4. Set temporal threshold (sampling year >= 2000). This removes years 1990, 1991, that show not-very-healthy values of "comparable abundance"
    refYear = data['sampling_year'].str[:4].astype('int')
    cond4 = (refYear >= 2000)
    # Filter by conditions
    data = data[ (cond1 & cond2 & cond3 & cond4) ]
    # Compute comparable abundance
    # 5. Total sampled time NA replaced by median (120), abundance=NA replace by zero
    data.loc[data['total_sampled_time'].isna(), 'total_sampled_time'] = np.nanmedian(data['total_sampled_time'])
    data.loc[data['ab_wildbees'].isna(), 'ab_wildbees'] = 0
    data.loc[data['ab_syrphids'].isna(), 'ab_syrphids'] = 0
    data.loc[data['ab_bombus'].isna()  , 'ab_bombus']   = 0
    # 6. Compute comparable abundances
    data['comp_ab_wb_bmb_syr'] = (data['ab_wildbees']+ data['ab_syrphids']+ data['ab_bombus']) / data['total_sampled_time']
    data['log_abundance']      = np.log(data['comp_ab_wb_bmb_syr'])
    return data[['site_id', 'study_id', 'management', 'log_abundance']]

def apply_minimum_conditions_features(data):
    # 1. Average columns _small and _large
    cols_to_avg = [col.split('_small')[0] for col in data.columns if 'small' in col]
    for col in cols_to_avg:
        col_small = col+'_small'
        col_large = col+'_large'
        data[col] = (data[col_small] + data[col_large])/2
        data.drop(columns=[col_small, col_large], inplace=True)
    # 2. Remove rows with 5 or more NaN values
    data = data.iloc[data[(data.isnull().sum(axis=1) < 5)].index]
    data.reset_index(drop=True, inplace=True)
    # 3. Set biome as categorical
    data['biome_num'] = data.biome_num.astype('object')
    data['biome_num'] = data.biome_num.replace(np.nan,"unknown")
    return data

def check_normality(data, column):
    sns.distplot(data[column])
    # skewness and kurtosis
    print("Skewness: %f" % data[column].skew()) # Skewness: -0.220768
    print("Kurtosis: %f" % data[column].kurt()) # Kurtosis: -0.168611
    # Check normality log_abundance
    sns.distplot(data[column], fit=norm)
    # fig = plt.figure()
    # res = stats.probplot(data[column], plot=plt)

def boxplot(data, x, ymin=-5, ymax=2):
    fig = sns.boxplot(x=x, y="log_abundance", data=data)
    fig.axis(ymin=ymin, ymax=ymax)

if __name__ == 'main':

    #######################################
    # Get, explore
    #######################################
    df_features = get_feature_data()
    df_field    = get_field_data()
    data = df_features.merge(df_field)
    data.drop(columns=['study_id', 'site_id'], inplace=True)
    check_normality(data, 'log_abundance')
    boxplot(data, 'biome_num', 'log_abundance')
    # Check normality other variables
    sns.distplot(data['elevation'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(data['elevation'], plot=plt)

    #######################################
    # Stratified split training and test
    #######################################
    cond = (data['biome_num'] != "unknown")
    xx = data[cond]
    strata = xx.biome_num.astype('category')
    x_train, x_test, y_train, y_test = train_test_split(xx, strata, stratify=strata,  test_size=0.25, random_state=135)
    # x_train = pd.concat([x_train, data[~cond]], axis=0) # add row with 'unknown' biome to training set
    x_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)

    #######################################
    # Save
    #######################################
    data.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/train+test.csv', index=False)
    x_train.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/train_set.csv', index=False)
    x_test.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/test_set.csv', index=False)




