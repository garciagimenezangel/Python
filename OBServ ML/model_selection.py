import numpy as np
import pandas as pd
import warnings

from sklearn.linear_model import ElasticNetCV, LassoCV, TweedieRegressor, BayesianRidge
from sklearn.svm import SVR
from sklearn.utils import all_estimators
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from scipy.stats import uniform
warnings.filterwarnings('ignore')

def get_data_prepared():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML_preprocessing/train/"
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
    df_results_sorted.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/model_selection.csv', index=False)
    ########################
    # Shortlist: check df_results and see which show low 'mean' and not-too-low 'rmse_all' (sign of possible overfitting)
    #######################
    # Selected estimators (no particular order):
    # 1 SVR
    # 2 ElasticNetCV
    # 3 LassoCV
    # 4 TweedieRegressor
    # 5 BayesianRidge

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
    search.best_params_ #{'C': 1.6979657260988366, 'coef0': -0.2855846601605023, 'epsilon': 0.061117460899596444, 'gamma': 0.10514781667802747, 'kernel': 'rbf'}
    search.best_score_  #-0.9869701127408879

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
    search.best_params_  #{'eps': 0.059404230559449316, 'l1_ratio': 0.9513270631254528, 'n_alphas': 3}
    search.best_score_  #-1.0280005710704525

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
    search.best_params_ # {'eps': 0.011183032750034745, 'n_alphas': 1}
    search.best_score_ #-1.0272733578190911

    # TweedieRegressor
    model = TweedieRegressor(max_iter=1000)
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
    search.best_params_  #{'power': 0}
    search.best_score_  #-1.049854731429676

    # BayesianRidge
    model = BayesianRidge(n_iter=3000)
    # define search space
    params = dict()
    params['alpha_1']     = uniform(loc=0.00000001, scale=0.1)
    params['alpha_2']     = uniform(loc=0.00000001, scale=0.1)
    params['lambda_1']    = uniform(loc=0.00000001, scale=0.1)
    params['lambda_2']    = uniform(loc=0.00000001, scale=0.1)
    params['alpha_init']  = uniform(loc=0.00000001, scale=0.1)
    params['lambda_init'] = uniform(loc=0.00000001, scale=0.1)
    # define the search
    search = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=-1)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_  # {'alpha_1': 0.0033435924818114803, 'alpha_2': 0.040853967588064345, 'alpha_init': 0.02715955787651284, 'lambda_1': 0.08930101099447024, 'lambda_2': 8.553613592314184e-05, 'lambda_init': 0.06775499126811027}
    search.best_score_  # -1.0580556094882547

