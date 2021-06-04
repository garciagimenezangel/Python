import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
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
    # Manipulate columns df_field
    #######################################
    # Conditions:
    # 1. Abundances must be integer numbers
    # 2. Latitude and longitude must be !na (there will not be model values anyways)
    # 3. Strictly positive abundances
    # 4. Discard sampling years 1990, 1991 ( much older than the rest (2004 onwards) )
    # Fill values
    # 5. Total sampled time NA replaced by median (120)
    # 6. Compute comparable abundances

    # Conditions:
    # 1. Abundances must be integer numbers
    decimal_wb  = (df_field['ab_wildbees'] % 1)
    decimal_syr = (df_field['ab_syrphids'] % 1)
    decimal_bmb = (df_field['ab_bombus'] % 1)
    cond1 = ((decimal_wb < 0.05) & (decimal_syr < 0.05) & (decimal_bmb < 0.05)) | \
            ((decimal_wb > 0.95) & (decimal_syr > 0.95) & (decimal_bmb > 0.95))
    cond1[cond1.isna()] = False
    # 2. Latitude and longitude must be !na (otherwise there will not be model values anyway)
    cond2 = (~df_field['latitude'].isna()) & (~df_field['longitude'].isna())
    # 3. Strictly positive abundances
    cond3 = (df_field['ab_wildbees'] > 0) | (df_field['ab_syrphids'] > 0) | (df_field['ab_bombus'] > 0)
    # 4. Discard sampling years 1990, 1991 ( much older than the rest (2004 onwards) )
    cond4 = (df_field['sampling_year'] != '1990') & (df_field['sampling_year'] != '1991')

    # Filter by conditions
    df_field = df_field[ (cond1 & cond2 & cond3 & cond4) ]

    # Fill values
    # 5. Total sampled time must be !na -> replace by median
    df_field.loc[ df_field['total_sampled_time'].isna(), 'total_sampled_time'] = np.nanmedian(df_field['total_sampled_time'])

    # 6. Compute comparable abundances
    df_field['comp_ab_wb']         = df_field['ab_wildbees']    / df_field['total_sampled_time']
    df_field['comp_ab_syr']        = df_field['ab_syrphids']    / df_field['total_sampled_time']
    df_field['comp_ab_bmb']        = df_field['ab_bombus']      / df_field['total_sampled_time']
    df_field['comp_ab_wb_syr']     = df_field['comp_ab_wb']     + df_field['comp_ab_syr']
    df_field['comp_ab_wb_bmb_syr'] = df_field['comp_ab_wb_syr'] + df_field['comp_ab_bmb']
    df_field['log_abundance']      = np.log(df_field['comp_ab_wb_bmb_syr'])

    #######################################
    # Explore df_field
    #######################################
    sns.distplot(df_field['log_abundance'])
    # skewness and kurtosis
    print("Skewness: %f" % np.log(df_field['comp_ab_wb_bmb_syr'].skew())) # Skewness: 1.556788
    print("Kurtosis: %f" % np.log(df_field['comp_ab_wb_bmb_syr'].kurt())) # Kurtosis: 3.654915

    # Check normality log_abundance
    sns.distplot(df_field['log_abundance'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(df_field['log_abundance'], plot=plt)

    #######################################
    # Manipulate columns df_features
    #######################################
    # 1. Average columns _small and _large
    # 2. Remove rows with 5 or more NaN values
    # 3. Fill numeric columns with mean
    # Drop system:index and .geo
    df_features.drop(columns=['system:index','.geo'], inplace=True)

    # 1. Average columns _small and _large
    cols_to_avg = [col.split('_small')[0] for col in df_features.columns if 'small' in col]
    for col in cols_to_avg:
        col_small = col+'_small'
        col_large = col+'_large'
        df_features[col] = (df_features[col_small] + df_features[col_large])/2
        df_features.drop(columns=[col_small, col_large], inplace=True)

    # 2. Remove rows with 5 or more NaN values
    df_features = df_features.iloc[df_features[(df_features.isnull().sum(axis=1) < 5)].index]
    df_features.reset_index(drop=True, inplace=True)
    rows_with_na_values = df_features.iloc[df_features[(df_features.isnull().any(axis=1))].index]

    # 3. Fill numeric columns with mean
    df_features['biome_num'] = df_features.biome_num.astype('object')
    df_features['biome_num'] = df_features.biome_num.replace(np.nan,"unknown")
    numeric = df_features.select_dtypes('number')
    df_features[numeric.columns] = numeric.fillna(numeric.mean())

    #######################################
    # Merge
    #######################################
    data = df_features.merge(df_field[['site_id', 'study_id', 'log_abundance']])

    #######################################
    # Explore data
    #######################################
    # box plot biome/abundance
    var = 'biome_num'
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="log_abundance", data=data)
    fig.axis(ymin=-5, ymax=2)

    # box plot biome/abundance
    var = 'refYear'
    f, ax = plt.subplots(figsize=(8, 6))
    fig = sns.boxplot(x=var, y="log_abundance", data=data)
    fig.axis(ymin=-5, ymax=2)

    # correlation matrices
    # All
    corrmat = data.corr()
    f, ax = plt.subplots(figsize=(15, 12))
    sns.heatmap(corrmat, vmax=.8, square=True)

    # log abundance correlations
    k = 10  # number of variables for heatmap
    cols = corrmat.nlargest(k, 'log_abundance')['log_abundance'].index
    cm = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values,
                     xticklabels=cols.values)
    plt.show()

    # Bioclim
    cols = [col for col in data.columns if ( ('bio' in col) & ~('biome' in col))]
    cm = np.corrcoef(data[cols].values.T)
    sns.set(font_scale=0.5)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, yticklabels=cols, xticklabels=cols)
    plt.show()
    sns.pairplot(data[cols], size=2.5)
    plt.show()

    # Check normality other variables
    sns.distplot(data['vs'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(data['vs'], plot=plt)

    #######################################
    # Manipulate columns data
    #######################################
    # Standardize numeric columns (except target log_abundance
    numeric = data.select_dtypes('number')
    numeric.drop(columns=['log_abundance'], inplace=True)
    data[numeric.columns] = StandardScaler().fit_transform(numeric)

    #######################################
    # Stratified split training and test
    #######################################
    cond = (data['biome_num'] != "unknown")
    xx = data[cond]
    strata = xx.biome_num.astype('category')
    x_train, x_test, y_train, y_test = train_test_split(xx, strata, stratify=strata,  test_size=0.25, random_state=135)
    x_train = pd.concat([x_train, data[~cond]], axis=0) # add row with 'unknown' biome to training set
    x_train.reset_index(drop=True, inplace=True)
    x_test.reset_index(drop=True, inplace=True)

    data.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/exploration_full.csv', index=False)
    x_train.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/exploration_train.csv', index=False)
    x_test.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/exploration_test.csv', index=False)
