import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import Ridge
from matplotlib import pyplot
import warnings
warnings.filterwarnings('ignore')

# Repository
modelsRepo    = "C:/Users/angel/git/Observ_models/"

# Folders
dataDir   = modelsRepo + "data/ML_preprocessing/"

if __name__ == '__main__':
    df_train = pd.read_csv(dataDir+'exploration_train.csv')
    df_test  = pd.read_csv(dataDir+'exploration_test.csv')

    # Remove categorical variables
    df_train_num = df_train.drop(columns=['refYear','study_id','site_id','biome_num'])
    df_test_num  = df_test.drop(columns=['refYear','study_id','site_id','biome_num'])
    df_train_num.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/data_num_train.csv', index=False)
    df_test_num.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/data_num_test.csv', index=False)

    # # Remove variables after superficial inspection of scatter plots (in Weka)
    # df_train.drop(columns=['moss'], inplace=True) # Moss~0
    # df_train.drop(columns=['activity'], inplace=True) # Collinearity with bio05

    #######################################
    # Recursive Feature Elimination (RFE)
    #######################################
    # Explore number of features
    y = df_train_num.log_abundance
    X = df_train_num.drop(columns=['log_abundance'])
    rfe = RFE(estimator=Ridge(), n_features_to_select=20)
    rfe.fit(X, y)
    X_reduced = X[ X.columns[rfe.support_] ]
    df_train_red = pd.concat([X_reduced, y], axis=1)
    y_test = df_test.log_abundance
    df_test_red = df_test[ X.columns[rfe.support_] ]
    df_test_red = pd.concat([df_test_red, y_test], axis=1)

    df_train_red.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/data_num_train_red.csv', index=False)
    df_test_red.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/data_num_test_red.csv', index=False)
