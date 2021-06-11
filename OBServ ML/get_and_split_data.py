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
    # Set biome as categorical and replace NA by unknown
    df_features['biome_num'] = df_features.biome_num.astype('object')
    df_features['biome_num'] = df_features.biome_num.replace(np.nan,"unknown")
    df_features.drop(columns=['system:index', '.geo', 'refYear'], inplace=True)
    cols_to_avg = [col.split('_small')[0] for col in df_features.columns if 'small' in col]
    for col in cols_to_avg:
        col_small = col+'_small'
        col_large = col+'_large'
        df_features[col] = (df_features[col_small] + df_features[col_large])/2
        df_features.drop(columns=[col_small, col_large], inplace=True)
    return df_features

def get_field_data():
    field_repo    = "C:/Users/angel/git/OBservData/"
    field_data_dir = field_repo + "Final_Data/"
    df_field     = pd.read_csv(field_data_dir+'CropPol_field_level_data.csv')
    return df_field[['site_id', 'study_id',
                     'ab_wildbees', 'ab_syrphids', 'ab_bombus',
                     'total_sampled_time', 'sampling_year', 'management']]

def apply_minimum_conditions(data):
    # Conditions:
    # 0. Latitude and longitude must be !na. Implicit because df_features only has data with defined lat and lon.
    # cond2 = (~data['latitude'].isna()) & (~data['longitude'].isna())
    # 1. Abundances must be integer numbers (tolerance of 0.05)
    decimal_wb  = (data['ab_wildbees'] % 1)
    decimal_syr = (data['ab_syrphids'] % 1)
    decimal_bmb = (data['ab_bombus'] % 1)
    nas   =  data['ab_wildbees'].isna() | data['ab_syrphids'].isna() | data['ab_bombus'].isna()
    cond1 = ((decimal_wb < 0.05) & (decimal_syr < 0.05) & (decimal_bmb < 0.05)) | \
            ((decimal_wb > 0.95) & (decimal_syr > 0.95) & (decimal_bmb > 0.95))
    cond1[cond1.isna()] = False
    print("Integer abundances:")
    print(cond1.describe())
    # 2. Strictly positive abundances
    cond2 = (~data['ab_wildbees'].isna() & data['ab_wildbees'] > 0) | \
            (~data['ab_syrphids'].isna() & data['ab_syrphids'] > 0) | \
            (~data['ab_bombus'].isna() & data['ab_bombus'] > 0)
    print("Strictly positive abundances:")
    print(cond2.describe())
    # 3. Set temporal threshold (sampling year >= 1992). This removes years 1990, 1991, that show not-very-healthy values of "comparable abundance"
    refYear = data['sampling_year'].str[:4].astype('int')
    cond3 = (refYear >= 1992)
    print("Ref year >=1992:")
    print(cond3.describe())
    # 4. Remove rows with 7 or more NaN values
    cond4 = (data.isnull().sum(axis=1) < 7)
    print("Less than 6 NAs per row:")
    print(cond4.describe())
    # 5. Biome defined
    cond5 = (data.biome_num != "unknown")
    print("Biome defined:")
    print(cond5.describe())

    # Filter by conditions
    all_cond = (cond1 & cond2 & cond3 & cond4 & cond5)
    print("ALL:")
    print(all_cond.describe())
    return data[ all_cond ]

def compute_comparable_abundance(data):
    # Compute comparable abundance
    # 5. Total sampled time NA replaced by median (120), abundance=NA replace by zero
    data.loc[data['total_sampled_time'].isna(), 'total_sampled_time'] = np.nanmedian(data['total_sampled_time'])
    data.loc[data['ab_wildbees'].isna(), 'ab_wildbees'] = 0
    data.loc[data['ab_syrphids'].isna(), 'ab_syrphids'] = 0
    data.loc[data['ab_bombus'].isna()  , 'ab_bombus']   = 0
    # 6. Compute comparable abundances
    data['comp_ab_wb_bmb_syr'] = (data['ab_wildbees']+ data['ab_syrphids']+ data['ab_bombus']) / data['total_sampled_time']
    data['log_abundance']      = np.log(data['comp_ab_wb_bmb_syr'])
    data.drop(columns=['ab_wildbees', 'ab_syrphids', 'ab_bombus', 'total_sampled_time', 'comp_ab_wb_bmb_syr'], inplace=True)
    return data

def check_normality(data, column):
    sns.distplot(data[column])
    # skewness and kurtosis
    print("Skewness: %f" % data[column].skew()) # Skewness: -0.220768
    print("Kurtosis: %f" % data[column].kurt()) # Kurtosis: -0.168611
    # Check normality log_abundance
    sns.distplot(data[column], fit=norm)
    fig = plt.figure()
    res = stats.probplot(data[column], plot=plt)

def boxplot(data, x, ymin=-5, ymax=2):
    fig = sns.boxplot(x=x, y="log_abundance", data=data)
    fig.axis(ymin=ymin, ymax=ymax)

if __name__ == 'main':

    #######################################
    # Get, explore
    #######################################
    df_features = get_feature_data()
    df_field    = get_field_data()
    data = df_features.merge(df_field, on=['study_id', 'site_id'])
    data = apply_minimum_conditions(data)
    data = compute_comparable_abundance(data)

    # data.drop(columns=['study_id', 'site_id'], inplace=True)
    check_normality(data, 'log_abundance')
    boxplot(data, 'biome_num', 'log_abundance')
    # Check normality other variables
    sns.distplot(data['elevation'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(data['elevation'], plot=plt)

    #######################################
    # Stratified split training and test
    #######################################
    strata = data.biome_num.astype('category')
    x_train, x_test, y_train, y_test = train_test_split(data, strata, stratify=strata,  test_size=0.25, random_state=135)
    x_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)

    #######################################
    # Save
    #######################################
    data.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/train+test_withIDs.csv', index=False)
    x_train.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/train/train_set_withIDs.csv', index=False)
    x_test.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/test/test_set_withIDs.csv', index=False)




