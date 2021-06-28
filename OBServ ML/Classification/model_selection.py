import numpy as np
import pandas as pd
import warnings

from sklearn.ensemble import ExtraTreesRegressor
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
    # 5 ExtraTreesRegressor

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
    search.best_params_
    search.best_score_

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
    search.best_params_
    search.best_score_

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

    # ExtraTreesRegressor
    model = ExtraTreesRegressor()
    # define search space
    params = dict()
    params['n_estimators'] = [2,8,32,128,512]
    params['max_depth']    = [1,4,16,64]
    params['min_samples_split'] = [1,4,16,64]
    params['min_samples_leaf']  = [1,4,16,64]
    params['min_weight_fraction_leaf']  = uniform(loc=0, scale=1)
    params['max_leaf_nodes'] = [1,4,16,64]
    params['min_impurity_split'] = [1,4,16,64]
    params['warm_start'] = [True, False]
    # define the search
    search = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=-1)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # {'max_depth': 4, 'max_leaf_nodes': 64, 'min_impurity_split': 1, 'min_samples_leaf': 4, 'min_samples_split': 16, 'min_weight_fraction_leaf': 0.07133051462209383, 'n_estimators': 8, 'warm_start': False}
    search.best_score_ # -1.077662406413546

    #model = ExtraTreesRegressor(max_depth=4, max_leaf_nodes=64, min_impurity_split=1, min_samples_leaf=4, min_samples_split=16, min_weight_fraction_leaf=0.07133051462209383, n_estimators=8, warm_start=False)
