import numpy as np
import pandas as pd
import warnings

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, TweedieRegressor
from sklearn.svm import SVR
from sklearn.utils import all_estimators
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from scipy.stats import uniform
warnings.filterwarnings('ignore')

def get_data_prepared():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/train/"
    return pd.read_csv(data_dir+'data_prepared.csv')

if __name__ == '__main__':
    data_prepared = get_data_prepared()
    predictors    = data_prepared.iloc[:,:-1]
    labels        = np.array(data_prepared.iloc[:,-1:]).flatten()

    # LIST OF ESTIMATORS OF TYPE "REGRESSOR" (TRY ALL)
    estimators = all_estimators(type_filter='regressor')
    results = []
    for name, RegressorClass in estimators:
        try:
            print('Regressor: ', name)
            reg = RegressorClass()
            reg.fit(predictors, labels)
            abundance_predictions = reg.predict(predictors)
            mse = mean_squared_error(labels, abundance_predictions)
            rmse = np.sqrt(mse)
            print('rmse all: ', rmse)
            scores = cross_val_score(reg, predictors, labels, scoring="neg_mean_squared_error", cv=5)
            rmse_scores = np.sqrt(-scores)
            print("Mean:", rmse_scores.mean())
            print("Std:", rmse_scores.std())
            results.append({'reg':reg, 'rmse all':rmse, 'mean':rmse_scores.mean(), 'std':rmse_scores.std()})

        except Exception as e:
            print(e)
    df_results = pd.DataFrame(results)
    df_results_sorted = df_results.sort_values(by=['mean'], ascending=True)
    df_results_sorted.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML/Regression/model_selection.csv', index=False)
    ########################
    # Shortlist: check df_results and see which show low 'mean' and not-too-low 'rmse_all' (sign of possible overfitting)
    #######################
    # Selected estimators (no particular order):
    # 1 SVR
    # 2 ElasticNetCV
    # 3 LassoCV
    # 4 TweedieRegressor
    # 5 RandomForestRegressor

    ########################
    # Hyperparameter tuning
    # I use randomized search over a (small) set of parameters, to get the best score. I repeat the process several
    # times, using a parameter space "surrounding" the best parameters in the previous step
    ########################
    # Note: BayesSearchCV in currently latest version of scikit-optimize not compatible with scikit-learn 0.24.1
    # When scikit-optimize version 0.9.0 is available (currenlty in development), use: BayesSearchCV(model,params,cv=5)
    # SVR
    model = SVR()
    # define search space
    params = dict()
    params['kernel']  = ['linear', 'poly', 'rbf', 'sigmoid']
    params['C']       = uniform(loc=0, scale=4)
    params['gamma']   = uniform(loc=0, scale=0.5)
    params['coef0']   = uniform(loc=-1, scale=1)
    params['epsilon'] = uniform(loc=0, scale=0.2)
    # define the search
    search = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=-1)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ #{'C': 1.7342889543571887, 'coef0': -0.3345352614917637, 'epsilon': 0.09256863132721108, 'gamma': 0.14372942931130184, 'kernel': 'rbf'}
    search.best_score_  #-1.0015267911375498

    # ElasticNetCV
    model = ElasticNetCV()
    # define search space
    params = dict()
    params['l1_ratio']      = uniform(loc=0, scale=1)
    params['eps']           = uniform(loc=0.01, scale=0.1)
    params['n_alphas']      = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
    # define the search
    search = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=-1)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # {'eps': 0.02931633062568853, 'l1_ratio': 0.08421237467099396, 'n_alphas': 2}
    search.best_score_ # -1.0528635717776293

    # LassoCV
    model = LassoCV()
    # define search space
    params = dict()
    params['eps']           = uniform(loc=0.001, scale=0.02)
    params['n_alphas']      = [1,2,3,4,5]
    # define the search
    search = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=-1)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # {'eps': 0.0020065877446290396, 'n_alphas': 3}
    search.best_score_ # -1.0616612391220008

    # TweedieRegressor
    model = TweedieRegressor()
    # define search space
    params = dict()
    params['power'] = [0,1,2,3]
    # define the search
    search = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=-1)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_
    search.best_score_

    # RandomForestRegressor
    model = RandomForestRegressor()
    # define search space
    params = dict()
    params['n_estimators'] = [250,500,550,600]
    params['max_depth']    = [1,2,4,8,16]
    params['min_samples_split'] = [4,16,32]
    params['min_samples_leaf']  = [16,64,128]
    params['min_weight_fraction_leaf']  = uniform(loc=0, scale=1)
    params['max_leaf_nodes'] = [4,16,32,64]
    params['min_impurity_decrease'] = uniform(loc=0, scale=1)
    params['warm_start'] = [True, False]
    # define the search
    search = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=-1)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # {'max_depth': 8, 'max_leaf_nodes': 32, 'min_impurity_decrease': 0.03918141818683696, 'min_samples_leaf': 64, 'min_samples_split': 32, 'min_weight_fraction_leaf': 0.12148045457124956, 'n_estimators': 550, 'warm_start': True}
    search.best_score_ # -1.0822663779610597



