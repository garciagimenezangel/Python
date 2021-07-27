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
                     'total_sampled_time', 'sampling_year', 'sampling_abundance']]

def compute_visit_rate(data):
    data['visit_rate_wb_bmb_syr'] = (data['ab_wildbees'] + data['ab_syrphids'] + data['ab_bombus'] + 1) / data['total_sampled_time']
    data['log_visit_rate']        = np.log(data['visit_rate_wb_bmb_syr'])
    data.drop(columns=['ab_wildbees', 'ab_syrphids', 'ab_bombus', 'total_sampled_time', 'visit_rate_wb_bmb_syr'], inplace=True)
    return data

def compute_visit_rate_small(data):
    data['visit_rate_wb_syr'] = (data['ab_wildbees']+ data['ab_syrphids'] + 1) / data['total_sampled_time']
    data['log_vr_small']      = np.log(data['visit_rate_wb_syr'])
    data.drop(columns=['ab_wildbees', 'ab_syrphids', 'visit_rate_wb_syr'], inplace=True)
    return data

def compute_visit_rate_large(data):
    # Compute comparable abundance
    data['visit_rate_bmb'] = (data['ab_bombus']+1) / data['total_sampled_time']
    data['log_vr_large']   = np.log(data['visit_rate_bmb'])
    data.drop(columns=['ab_bombus', 'visit_rate_bmb'], inplace=True)
    return data

def fill_missing_abundances(data):
    data.loc[data['ab_bombus'].isna(), 'ab_bombus']     = 0
    data.loc[data['ab_wildbees'].isna(), 'ab_wildbees'] = 0
    data.loc[data['ab_syrphids'].isna(), 'ab_syrphids'] = 0
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

def boxplot(data, x, y, ymin=-5, ymax=2):
    fig = sns.boxplot(x=x, y=y, data=data)
    fig.axis(ymin=ymin, ymax=ymax)

def add_mechanistic_values(data, model_name='Lonsdorf.Delphi_lcCont1_open0_forEd0_crEd0_div0_ins0max_dist0_suitmult'):
    data_dir = "C:/Users/angel/git/Observ_models/data/"
    model_data  = pd.read_csv(data_dir + 'model_data_lite.csv')[['site_id','study_id',model_name]]
    return data.merge(model_data, on=['study_id', 'site_id'])

def is_sampling_method_accepted(x):
    cond1 = 'pan trap' not in x
    cond2 = x != "nan"
    return (cond1 & cond2)

def is_one_guild_measured(x,y,z, thresh):
    return ( (~np.isnan(x) & (x>thresh)) | (~np.isnan(y) & (y>thresh)) | (~np.isnan(z) & (z>thresh)) )

def are_abundances_integer(study_data): # do not exclude NAs (filtered or transformed in other steps)
    tol = 0.05
    cond_wb  = ((study_data['ab_wildbees'] % 1) < tol) | ((study_data['ab_wildbees'] % 1) > (1-tol)) | study_data['ab_wildbees'].isna()
    cond_syr = ((study_data['ab_syrphids'] % 1) < tol) | ((study_data['ab_syrphids'] % 1) > (1-tol)) | study_data['ab_syrphids'].isna()
    cond_bmb = ((study_data['ab_bombus'] % 1) < tol)   | ((study_data['ab_bombus'] % 1) > (1-tol))   | study_data['ab_bombus'].isna()
    cond = cond_wb & cond_syr & cond_bmb
    return all(cond)

def are_abundances_integer(study_data): # do not exclude NAs (filtered or transformed in other steps)
    tol = 0.05
    cond_wb  = ((study_data['ab_wildbees'] % 1) < tol) | ((study_data['ab_wildbees'] % 1) > (1-tol)) | study_data['ab_wildbees'].isna()
    cond_syr = ((study_data['ab_syrphids'] % 1) < tol) | ((study_data['ab_syrphids'] % 1) > (1-tol)) | study_data['ab_syrphids'].isna()
    cond_bmb = ((study_data['ab_bombus'] % 1) < tol)   | ((study_data['ab_bombus'] % 1) > (1-tol))   | study_data['ab_bombus'].isna()
    cond = cond_wb & cond_syr & cond_bmb
    return all(cond)

def apply_conditions(data, thresh_ab=0):
    # 1. Abundances of all sites in the study must be integer numbers (tolerance of 0.05)
    abs_integer = data.groupby('study_id').apply(are_abundances_integer)
    sel_studies = abs_integer.index[abs_integer]
    cond1       = data['study_id'].isin(sel_studies)
    print("1. Studies with all abundances integer:")
    print(abs_integer.describe())
    print("1b: Sites")
    print(cond1.describe())

    # 2. At least one guild measured with abundance > thresh_ab
    cond2 = pd.Series([is_one_guild_measured(x,y,z,thresh_ab) for (x,y,z) in zip(data['ab_wildbees'], data['ab_syrphids'], data['ab_bombus'])])
    print("2. At least one guild measured with abundance > "+str(thresh_ab)+":")
    print(cond2.describe())

    # 3. Set temporal threshold (sampling year >= 1992). This removes years 1990, 1991, that show not-very-healthy values of "comparable abundance"
    refYear = data['sampling_year'].str[:4].astype('int')
    cond3 = (refYear >= 1992)
    print("3. Ref year >=1992:")
    print(cond3.describe())

    # 4. Sampling method != pan trap
    cond4 = pd.Series([ is_sampling_method_accepted(x) for x in data['sampling_abundance'].astype('str') ])
    print("4. Sampling method not pan trap:")
    print(cond4.describe())
    data.drop(columns=['sampling_abundance'], inplace=True)

    # 5. Total sampled time != NA
    cond5 = ~data['total_sampled_time'].isna()
    print("5. Defined sampled time:")
    print(cond5.describe())

    # Filter by conditions
    all_cond = (cond1 & cond2 & cond3 & cond4 & cond5)
    print("ALL:")
    print(all_cond.describe())
    return data[ all_cond ]


if __name__ == '__main__':

    #######################################
    # Get
    #######################################
    df_features = get_feature_data()
    df_field    = get_field_data()
    data = df_features.merge(df_field, on=['study_id', 'site_id'])
    data = apply_conditions(data)
    data = fill_missing_biomes(data)
    data = remap_crops(data)
    data = fill_missing_abundances(data)
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
    # Apply transformations (fill values, standardize, one-hot encoding)
    # First, replace numeric by mean, grouped by study_id (if all sites have NAs, then replace by dataset mean later in the imputer)
    pred_num     = predictors.select_dtypes('number')
    n_nas        = pred_num.isna().sum().sum()
    pred_num['study_id'] = data.study_id
    pred_num = pred_num.groupby('study_id').transform(lambda x: x.fillna(x.mean()))
    print("NA'S before transformation: " + str(n_nas))
    print("Total numeric values: " + str(pred_num.size))
    print("Percentage: " + str(n_nas*100/pred_num.size))

    # Define pipleline
    numeric_col = list(pred_num)
    onehot_col  = ["biome_num"]
    ordinal_col = ["management"]
    dummy_col   = ["study_id","site_id"] # keep this to use later (e.g. create custom cross validation iterator)
    num_pipeline = Pipeline([
        ('num_imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler())
    ])
    ordinal_pipeline = Pipeline([
         ('manag_imputer', SimpleImputer(strategy="constant", fill_value="conventional")),
         ('ordinal_encoder', OrdinalEncoder(categories=[['conventional','IPM','unmanaged','organic']]))
    ])
    onehot_pipeline = Pipeline([
        ('onehot_encoder', OneHotEncoder())
    ])
    dummy_pipeline = Pipeline([('dummy_imputer', SimpleImputer(strategy="constant", fill_value=""))])
    X = onehot_pipeline.fit(predictors[onehot_col])
    onehot_encoder_names = X.named_steps['onehot_encoder'].get_feature_names()
    full_pipeline = ColumnTransformer([
        ("numeric", num_pipeline, numeric_col),
        ("ordinal", ordinal_pipeline, ordinal_col),
        ("dummy", dummy_pipeline, dummy_col),
        ("onehot",  onehot_pipeline, onehot_col )
    ])

    #######################################
    # Transform
    #######################################
    x_transformed = full_pipeline.fit_transform(predictors)

    # Convert into data frame
    numeric_col = np.array(pred_num.columns)
    dummy_col = np.array(["study_id","site_id"])
    onehot_col  = np.array(onehot_encoder_names)
    feature_names = np.concatenate( (numeric_col, ordinal_col, dummy_col, onehot_col), axis=0)
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

    x_train, x_test, y_train, y_test = train_test_split(df_studies_split, strata, stratify=strata, test_size=0.2, random_state=135)
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