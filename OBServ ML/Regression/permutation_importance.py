# Extract feature importance using the best models from model_selection, to see
# how much they depend on the model, and whether there are some that stand out
# as important most of the times

import pickle
import pandas as pd
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, ExtraTreesRegressor, \
    AdaBoostRegressor
from sklearn.feature_selection import RFE, RFECV
import warnings
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, cross_validate
from matplotlib import pyplot
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVR
from sklearn.inspection import permutation_importance
import datetime
warnings.filterwarnings('ignore')
models_repo = "C:/Users/angel/git/Observ_models/"

def get_train_data_prepared():
    data_dir   = models_repo + "data/ML/Regression/train/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def get_test_data_prepared():
    data_dir   = models_repo + "data/ML/Regression/test/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def get_train_data_prepared_with_management():
    data_dir   = models_repo + "data/ML/Regression/train/"
    return pd.read_csv(data_dir+'data_prepared_withManag.csv')

def get_test_data_prepared_with_management():
    data_dir   = models_repo + "data/ML/Regression/test/"
    return pd.read_csv(data_dir+'data_prepared_withManag.csv')

def get_feature_importance(model, predictors_train, labels_train):
    model.fit(predictors_train, labels_train)
    perm_importance = permutation_importance(model, predictors_train, labels_train, random_state=135, n_jobs=6)
    feature_names = predictors_train.columns
    return pd.DataFrame(sorted(zip(perm_importance.importances_mean, feature_names), reverse=True))

train_prepared = get_train_data_prepared()
test_prepared = get_test_data_prepared()
# Including mechanistic model value
train_prepared = get_train_data_prepared_with_management()
test_prepared = get_test_data_prepared_with_management()

# Get predictors and labels
predictors_train = train_prepared.iloc[:,:-1]
predictors_test  = test_prepared.iloc[:,:-1]
labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()

# Load custom cross validation
with open('C:/Users/angel/git/Observ_models/data/ML/Regression/train/myCViterator.pkl', 'rb') as file:
    myCViterator = pickle.load(file)

model = HistGradientBoostingRegressor(l2_regularization=0.02021888460670551, learning_rate=0.04277282248041758,
                                      loss='least_squares', max_depth=4, max_leaf_nodes=32,
                                      min_samples_leaf=16, warm_start=True)
f_imp_1 = get_feature_importance(model, predictors_train, labels_train);
f_imp_1.columns = ['imp1','feature']
f_imp_1['imp1'] = f_imp_1.imp1 / np.sum(f_imp_1.imp1)

model = SVR(C=2.9468542209755357, coef0=-0.6868465520687694, degree=4, epsilon=0.18702907953343395, gamma=0.1632449384464454, kernel='rbf', shrinking=True)
f_imp_2 = get_feature_importance(model, predictors_train, labels_train);
f_imp_2.columns = ['imp2','feature']
f_imp_2['imp2'] = f_imp_2.imp2 / np.sum(f_imp_2.imp2)
df_feat_imp = pd.merge(f_imp_1, f_imp_2, on=['feature'])

model = ExtraTreesRegressor(bootstrap=False, ccp_alpha=0.019454451882791046, criterion='mae', max_depth=8, max_leaf_nodes=32, min_impurity_decrease=0.49865469993092115, min_samples_leaf=0.38736893138328954, min_samples_split=0.5708381504562543, min_weight_fraction_leaf=0.11618121903130718, n_estimators=200, warm_start=False)
f_imp_3 = get_feature_importance(model, predictors_train, labels_train);
f_imp_3.columns = ['imp3','feature']
f_imp_3['imp3'] = f_imp_3.imp3 / np.sum(f_imp_3.imp3)
df_feat_imp = pd.merge(df_feat_imp, f_imp_3, on=['feature'])

model = AdaBoostRegressor(base_estimator=HistGradientBoostingRegressor(l2_regularization=0.02021888460670551,
                            learning_rate=0.04277282248041758, max_depth=4,
                            max_leaf_nodes=32, min_samples_leaf=16,
                            warm_start=True), learning_rate=0.21875144982480377, loss='linear', n_estimators=50)
f_imp_4 = get_feature_importance(model, predictors_train, labels_train);
f_imp_4.columns = ['imp4','feature']
f_imp_4['imp4'] = f_imp_4.imp4 / np.sum(f_imp_4.imp4)
df_feat_imp = pd.merge(df_feat_imp, f_imp_4, on=['feature'])

model = RandomForestRegressor(bootstrap=True, ccp_alpha=0.040560610473318826, criterion='mse', max_depth=4, max_leaf_nodes=64, min_impurity_decrease=0.003427302639933183, min_samples_split=0.22116336729743802, min_weight_fraction_leaf=0.086566739859892, n_estimators=400, warm_start=False)
f_imp_5 = get_feature_importance(model, predictors_train, labels_train);
f_imp_5.columns = ['imp5','feature']
f_imp_5['imp5'] = f_imp_5.imp5 / np.sum(f_imp_5.imp5)
df_feat_imp = pd.merge(df_feat_imp, f_imp_5, on=['feature'])

df_feat_imp["imp_sum"] = df_feat_imp.sum(axis=1)
df_feat_imp.sort_values(by='imp_sum', ascending=False, inplace=True)

threshold = 0.02
features_selected = f_imp_1.loc[f_imp_1.imp1 > threshold, 'feature']
n_features = str(len(features_selected))
data_reduced_train = train_prepared[np.append(features_selected, ['log_visit_rate'])]
data_reduced_test = test_prepared[np.append(features_selected, ['log_visit_rate'])]
data_reduced_train.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/train/data_reduced_'+n_features+'.csv', index=False)
data_reduced_test.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/test/data_reduced_'+n_features+'.csv', index=False)


