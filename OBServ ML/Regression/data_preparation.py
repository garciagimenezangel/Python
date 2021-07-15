import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split, StratifiedKFold
from scipy import stats
import pollinators_dependency as poll_dep
import pickle
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
    df_features.rename(columns={'crop':'cropland'}, inplace=True)
    return df_features

def get_field_data():
    field_repo    = "C:/Users/angel/git/OBservData/"
    field_data_dir = field_repo + "Final_Data/"
    df_field     = pd.read_csv(field_data_dir+'CropPol_field_level_data.csv')
    return df_field[['site_id', 'study_id', 'crop', 'management',
                     'ab_wildbees', 'ab_syrphids', 'ab_bombus',
                     'total_sampled_time', 'sampling_year']]

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
    print("Less than 7 NAs per row:")
    print(cond4.describe())
    # 5. Total sampled time != NA
    cond5 = ~data['total_sampled_time'].isna()
    print("Defined sampled time:")
    print(cond5.describe())

    # Filter by conditions
    all_cond = (cond1 & cond2 & cond3 & cond4 & cond5)
    print("ALL:")
    print(all_cond.describe())
    return data[ all_cond ]

def compute_visit_rate(data):
    # Compute comparable abundance
    # 5. Total sampled time NA replaced by median (120), abundance=NA replace by zero
    data.loc[data['total_sampled_time'].isna(), 'total_sampled_time'] = np.nanmedian(data['total_sampled_time'])
    data.loc[data['ab_wildbees'].isna(), 'ab_wildbees'] = 0
    data.loc[data['ab_syrphids'].isna(), 'ab_syrphids'] = 0
    data.loc[data['ab_bombus'].isna()  , 'ab_bombus']   = 0
    # 6. Compute comparable abundances
    data['visit_rate_wb_bmb_syr'] = (data['ab_wildbees']+ data['ab_syrphids']+ data['ab_bombus']) / data['total_sampled_time']
    data['log_visit_rate']      = np.log(data['visit_rate_wb_bmb_syr'])
    data.drop(columns=['ab_wildbees', 'ab_syrphids', 'ab_bombus', 'total_sampled_time', 'visit_rate_wb_bmb_syr'], inplace=True)
    return data

def compute_visit_rate_small(data):
    # Compute comparable abundance
    # 5. Total sampled time NA replaced by median (120), abundance=NA replace by zero
    data.loc[data['total_sampled_time'].isna(), 'total_sampled_time'] = np.nanmedian(data['total_sampled_time'])
    data.loc[data['ab_wildbees'].isna(), 'ab_wildbees'] = 0
    data.loc[data['ab_syrphids'].isna(), 'ab_syrphids'] = 0
    # 6. Compute comparable abundances
    data['visit_rate_wb_syr'] = (data['ab_wildbees']+ data['ab_syrphids']) / data['total_sampled_time']
    data['log_vr_small']  = np.log(data['visit_rate_wb_syr'])
    data.drop(columns=['ab_wildbees', 'ab_syrphids', 'visit_rate_wb_syr'], inplace=True)
    return data

def compute_visit_rate_large(data):
    # Compute comparable abundance
    # 5. Total sampled time NA replaced by median (120), abundance=NA replace by zero
    data.loc[data['total_sampled_time'].isna(), 'total_sampled_time'] = np.nanmedian(data['total_sampled_time'])
    data.loc[data['ab_bombus'].isna(), 'ab_bombus'] = 0
    # 6. Compute comparable abundances
    data['visit_rate_bmb'] = data['ab_bombus'] / data['total_sampled_time']
    data['log_vr_large']  = np.log(data['visit_rate_bmb'])
    data.drop(columns=['ab_bombus', 'visit_rate_bmb'], inplace=True)
    return data

def fill_biome(x, data):
    data_study_id = data.loc[data.study_id == x, ]
    return data_study_id.biome_num.mode().iloc[0]

def fill_missing_biomes(data):
    missing_biome = data.loc[data.biome_num == 'unknown',]
    new_biome     = [fill_biome(x, data) for x in missing_biome.study_id]
    data.loc[data.biome_num == 'unknown', 'biome_num'] = new_biome
    return data

def remap_crops(data):
    data['crop'] = data['crop'].map(poll_dep.dep)
    return data

def check_normality(data, column):
    sns.distplot(data[column])
    # skewness and kurtosis
    print("Skewness: %f" % data[column].skew()) # Skewness: -0.220768
    print("Kurtosis: %f" % data[column].kurt()) # Kurtosis: -0.168611
    # Check normality log_visit_rate
    sns.distplot(data[column], fit=norm)
    fig = plt.figure()
    res = stats.probplot(data[column], plot=plt)

def boxplot(data, x, ymin=-5, ymax=2):
    fig = sns.boxplot(x=x, y="log_visit_rate", data=data)
    fig.axis(ymin=ymin, ymax=ymax)

def add_mechanistic_values(data, model_name='Lonsdorf.Delphi_lcCont1_open0_forEd0_crEd0_div0_ins0max_dist0_suitmult'):
    data_dir = "C:/Users/angel/git/Observ_models/data/"
    model_data  = pd.read_csv(data_dir + 'model_data_lite.csv')[['site_id','study_id',model_name]]
    return data.merge(model_data, on=['study_id', 'site_id'])

if __name__ == '__main__':

    #######################################
    # Get
    #######################################
    df_features = get_feature_data()
    df_field    = get_field_data()
    data = df_features.merge(df_field, on=['study_id', 'site_id'])
    data = apply_minimum_conditions(data)
    data = fill_missing_biomes(data)
    # data = remap_crops(data)
    data = compute_visit_rate(data)
    # data = compute_visit_rate_small(data)
    # data = compute_visit_rate_large(data)
    # data = add_mechanistic_values(data)

    # Separate predictors and labels
    predictors = data.drop("log_visit_rate", axis=1)
    labels     = data['log_visit_rate'].copy()

    # (Set biome as categorical)
    predictors['biome_num'] = predictors.biome_num.astype('object')

    #######################################
    # Pipeline
    #######################################
    pred_num = predictors.select_dtypes('number')
    numeric_col = list(pred_num)
    # ordinal_col = ["management"]
    onehot_col  = ["biome_num"]
    dummy_col   = ["study_id","site_id"] # keep this to use later (e.g. create custom cross validation iterator)
    num_pipeline = Pipeline([
        ('num_imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler())
    ])
    # ordinal_pipeline = Pipeline([
    #     ('manag_imputer', SimpleImputer(strategy="constant", fill_value="conventional")),
    #     ('ordinal_encoder', OrdinalEncoder(categories=[['conventional','IPM','unmanaged','organic']]))
    # ])
    onehot_pipeline = Pipeline([
        ('onehot_encoder', OneHotEncoder())
    ])
    dummy_pipeline = Pipeline([('dummy_imputer', SimpleImputer(strategy="constant", fill_value=""))])
    X = onehot_pipeline.fit(predictors[onehot_col])
    onehot_encoder_names = X.named_steps['onehot_encoder'].get_feature_names()
    full_pipeline = ColumnTransformer([
        ("numeric", num_pipeline, numeric_col),
        # ("ordinal", ordinal_pipeline, ordinal_col),
        ("dummy", dummy_pipeline, dummy_col),
        ("onehot",  onehot_pipeline, onehot_col )
    ])

    #######################################
    # Transform
    #######################################
    x_transformed = full_pipeline.fit_transform(predictors)

    # Convert into data frame
    numeric_col = np.array(pred_num.columns)
    # ordinal_col = np.array(["management"])
    dummy_col = np.array(["study_id","site_id"])
    onehot_col  = np.array(onehot_encoder_names)
    # feature_names = np.concatenate( (numeric_col, ordinal_col, onehot_col), axis=0)
    feature_names = np.concatenate( (numeric_col, dummy_col, onehot_col), axis=0)
    predictors_prepared = pd.DataFrame(x_transformed, columns=feature_names, index=predictors.index)
    dataset_prepared = predictors_prepared.copy()
    dataset_prepared['log_visit_rate'] = labels

    # Reset indices
    data.reset_index(inplace=True, drop=True)
    dataset_prepared.reset_index(inplace=True, drop=True)

    #############################################################
    # Stratified split training and test (split by study_id)
    #############################################################
    df_studies = data.groupby('study_id', as_index=False).first()[['study_id','biome_num']]
    # For the training set, take biomes with more than one count (otherwise I get an error in train_test_split.
    # They are added in the test set later, to keep all data
    has_more_one     = df_studies.groupby('biome_num').count().study_id > 1
    df_studies_split = df_studies.loc[has_more_one[df_studies.biome_num].reset_index().study_id,]
    strata           = df_studies_split.biome_num.astype('category')

    x_train, x_test, y_train, y_test = train_test_split(df_studies_split, strata, stratify=strata, test_size=0.25, random_state=135)
    studies_train   = x_train.study_id
    train_selection = [ (x_train.study_id == x).any() for x in data.study_id ]
    df_train = dataset_prepared[train_selection].reset_index(drop=True)
    df_test  = dataset_prepared[[~x for x in train_selection]].reset_index(drop=True)

    # Save predictors and labels (train and set), removing study_id
    df_train.drop(columns=['study_id', 'site_id']).to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML/Regression/train/data_prepared.csv', index=False)
    df_test.drop(columns=['study_id', 'site_id']).to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML/Regression/test/data_prepared.csv', index=False)

    # Save predictors and labels including model data (train and set)
    # df_train.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML/Regression/train/data_prepared_with_mech.csv', index=False)
    # df_test.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML/Regression/test/data_prepared_with_mech.csv', index=False)

    # Save data (not processed by pipeline) including study_id and site_id
    train_withIDs = data[train_selection].copy().reset_index(drop=True)
    test_withIDs  = data[[~x for x in train_selection]].copy().reset_index(drop=True)
    train_withIDs.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML/Regression/train/data_prepared_withIDs.csv', index=False)
    test_withIDs.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML/Regression/test/data_prepared_withIDs.csv', index=False)

    # Save custom cross validation iterator
    df_studies = data[train_selection].reset_index(drop=True).groupby('study_id', as_index=False).first()[['study_id', 'biome_num']]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=135)
    target = df_studies.loc[:, 'biome_num'].astype(int)
    df_studies['fold'] = -1
    n_fold = 0
    for train_index, test_index in skf.split(df_studies, target):
        df_studies.loc[test_index,'fold'] = n_fold
        n_fold = n_fold+1
    df_studies.drop(columns=['biome_num'], inplace=True)
    dict_folds = df_studies.set_index('study_id').T.to_dict('records')[0]
    df_train.replace(to_replace=dict_folds, inplace=True)
    myCViterator = []
    for i in range(0,5):
        trainIndices = df_train[df_train['study_id'] != i].index.values.astype(int)
        testIndices = df_train[df_train['study_id'] == i].index.values.astype(int)
        myCViterator.append((trainIndices, testIndices))
    with open('C:/Users/angel/git/Observ_models/data/ML/Regression/train/myCViterator.pkl', 'wb') as f:
        pickle.dump(myCViterator, f)

    #######################################
    # Explore
    #######################################
    check_normality(data, 'log_visit_rate')
    data['year'] = data['study_id'].str[-4:]
    a = data.loc[data.year != "16_2",]
    boxplot(a, 'year', 'log_visit_rate')
    boxplot(data, 'biome_num', 'log_visit_rate')
    # Check normality other variables
    sns.distplot(data['elevation'], fit=norm)
    fig = plt.figure()
    res = stats.probplot(data['elevation'], plot=plt)

    # count NA's
    n_na = data.isnull().sum().sort_values(ascending = False).sum()

    # guilds
    small = data.copy()
    small = small.loc[ np.isfinite(small['log_vr_small']), ]
    check_normality(small, 'log_vr_small')
    large = data.copy().sort_values(by=["log_vr_large"])
    large = large.loc[ np.isfinite(large['log_vr_large']), ]
    check_normality(large, 'log_vr_large')