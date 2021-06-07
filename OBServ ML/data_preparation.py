import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def get_train_test_sets():
    modelsRepo    = "C:/Users/angel/git/Observ_models/"
    dataDir   = modelsRepo + "data/ML_preprocessing/"
    return ( pd.read_csv(dataDir+'train_set.csv'), pd.read_csv(dataDir+'test_set.csv') )

if __name__ == '__main__':
    df_train, df_test = get_train_test_sets()

    # Separate predictors and labels
    predictors = df_train.drop("log_abundance", axis=1)
    labels     = df_train["log_abundance"].copy()

    # (Set biome as categorical)
    predictors['biome_num'] = predictors.biome_num.astype('object')

    #######################################
    # Transformations
    #######################################
    # 1. Fill NA's in numeric columns with mean
    # 2. Fill NA's in management column with "conventional"
    # 3. One-hot encoding of biome_num
    # 4. Ordinal encoding management
    # 5. Standardize numeric columns

    # # Imputers (fill NA's strategy)
    # numeric_imputer    = SimpleImputer(strategy="mean")
    # management_imputer = SimpleImputer(strategy="constant", fill_value="conventional")
    #
    # # One-hot encoding
    # biome_encoder = OneHotEncoder()
    # biome_one_hot = biome_encoder.fit_transform(predictors[['biome_num']])
    #
    # # Ordinal encoding
    # pred_management = predictors[['management']]
    # X = management_imputer.fit_transform(pred_management)
    # pred_management = pd.DataFrame(X, columns=pred_management.columns, index=pred_management.index)
    # management_encoder = OrdinalEncoder(categories=[['unmanaged','conventional','IPM','organic']])
    # a = management_encoder.fit_transform(pred_management)
    #
    # # Standardize numeric columns (except target log_abundance
    # pred_num = predictors.select_dtypes('number')
    # pred_num = StandardScaler().fit_transform(pred_num)

    #######################################
    # Pipeline
    #######################################
    pred_num = predictors.select_dtypes('number')
    numeric_col    = list(pred_num)
    management_col = ["management"]
    biome_col      = ["biome_num"]
    num_pipeline = Pipeline([
        ('num_imputer', SimpleImputer(strategy="mean")),
        ('std_scaler', StandardScaler())
    ])
    management_pipeline = Pipeline([
        ('manag_imputer', SimpleImputer(strategy="constant", fill_value="conventional")),
        ('ordinal_encoder', OrdinalEncoder(categories=[['unmanaged','conventional','IPM','organic']]))
    ])
    biome_pipeline = Pipeline([
        ('onehot_encoder', OneHotEncoder())
    ])
    X = biome_pipeline.fit(predictors[biome_col])
    biome_encoder_names = X.named_steps['onehot_encoder'].get_feature_names()
    full_pipeline = ColumnTransformer([
        ("numeric", num_pipeline, numeric_col),
        ("management", management_pipeline, management_col),
        ("biome", biome_pipeline, biome_col )
    ])
    X = full_pipeline.fit_transform(predictors)

    # Convert into data frame
    numeric_col    = np.array(pred_num.columns)
    management_col = np.array(["management"])
    biome_col      = np.array(biome_encoder_names)
    feature_names  = np.concatenate( (numeric_col, management_col, biome_col), axis=0)
    predictors_prepared = pd.DataFrame(X, columns=feature_names, index=predictors.index)

    predictors_prepared.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/predictors_prepared.csv', index=False)
    labels.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/labels.csv', index=False)

    pred_and_labels_prepared = pd.concat([predictors_prepared.reset_index(drop=True), labels], axis=1)
    pred_and_labels_prepared.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/labels_and_predictors_prepared.csv', index=False)