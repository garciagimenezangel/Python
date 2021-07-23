import numpy as np
import pandas as pd
import warnings
import pickle

from sklearn.cross_decomposition import PLSRegression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, \
    HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, TweedieRegressor, BayesianRidge, OrthogonalMatchingPursuitCV
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.utils import all_estimators
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from scipy.stats import uniform
warnings.filterwarnings('ignore')
models_repo = "C:/Users/angel/git/Observ_models/"
root_dir    = models_repo + "data/ML/Regression/train/"

def get_data_prepared():
    return pd.read_csv(root_dir+'data_prepared.csv')

def get_data_reduced(n_features):
    return pd.read_csv(root_dir+'data_reduced_'+str(n_features)+'.csv')

if __name__ == '__main__':
    data_prepared = get_data_prepared()
    # data_prepared = get_data_reduced(15)
    predictors    = data_prepared.iloc[:,:-1]
    labels        = np.array(data_prepared.iloc[:,-1:]).flatten()

    # Load custom cross validation
    with open('C:/Users/angel/git/Observ_models/data/ML/Regression/train/myCViterator.pkl', 'rb') as file:
        myCViterator = pickle.load(file)

    # LIST OF ESTIMATORS OF TYPE "REGRESSOR" (TRY ALL)
    estimators = all_estimators(type_filter='regressor')
    results = []
    for name, RegressorClass in estimators:
        try:
            print('Regressor: ', name)
            reg = RegressorClass()
            reg.fit(predictors, labels)
            abundance_predictions = reg.predict(predictors)
            mae = mean_absolute_error(labels, abundance_predictions)
            print('MAE all: ', mae)
            scores = cross_val_score(reg, predictors, labels, scoring="neg_mean_absolute_error", cv=myCViterator)
            mae_scores = -scores
            print("Mean:", mae_scores.mean())
            print("Std:", mae_scores.std())
            results.append({'reg':reg, 'MAE all':mae, 'mean':mae_scores.mean(), 'std':mae_scores.std()})
        except Exception as e:
            print(e)
    df_results = pd.DataFrame(results)
    df_results_sorted = df_results.sort_values(by=['mean'], ascending=True)
    df_results_sorted.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML/Regression/hyperparameters/model_selection.csv', index=False)
    ########################
    # Shortlist: check df_results and see which show low 'mean' and not-too-low 'rmse_all' (sign of possible overfitting)
    #######################
    # Selected estimators (no particular order):
    # 1 NuSVR
    # 2 MLPRegressor
    # 3 TweedieRegressor
    # 4 GradientBoostingRegressor
    # 5 ElasticNetCV
    # 6 PLSRegression

    ########################
    # Hyperparameter tuning
    # I use randomized search over a (small) set of parameters, to get the best score. I repeat the process several
    # times, using a parameter space "surrounding" the best parameters in the previous step
    ########################
    results=[]
    # Note: BayesSearchCV in currently latest version of scikit-optimize not compatible with scikit-learn 0.24.1
    # When scikit-optimize version 0.9.0 is available (currently in development), use: BayesSearchCV(model,params,cv=5)
    # SVR
    model = NuSVR()
    # define search space
    params = dict()
    params['kernel']  = ['linear', 'poly', 'rbf', 'sigmoid']
    params['nu']      = uniform(loc=0, scale=1)
    params['C']       = uniform(loc=0, scale=4)
    params['gamma']   = uniform(loc=0, scale=0.5)
    params['coef0']   = uniform(loc=-1, scale=1)
    params['degree']  = [3,4,5]
    params['shrinking'] = [False, True]
    # define the search
    search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=200,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=6)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(-mean_score, params)
    search.best_params_ # all features:
    search.best_score_  # all features:
    results.append({'model': model, 'best_params': search.best_params_, 'best_score': -search.best_score_})

    # MLPRegressor
    model = MLPRegressor(max_iter=10000, solver='sgd')
    # define search space
    params = dict()
    params['hidden_layer_sizes'] = [(20,),(50,),(100,),(200,)]
    params['activation'] = ['identity', 'logistic', 'tanh', 'relu']
    params['alpha'] = uniform(loc=0, scale=0.1)
    params['learning_rate'] = ['constant', 'invscaling', 'adaptive']
    params['learning_rate_init'] = uniform(loc=0, scale=0.1)
    params['power_t'] = uniform(loc=0, scale=1)
    params['momentum'] = uniform(loc=0, scale=1)
    # define the search
    search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=200,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=6)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # all features:
    search.best_score_ # all features:
    results.append({'model': model, 'best_params': search.best_params_, 'best_score': -search.best_score_})

    # TweedieRegressor
    model = TweedieRegressor(max_iter=10000)
    # define search space
    params = dict()
    params['power'] = [0,2,3]
    params['alpha'] = uniform(loc=0, scale=3)
    # define the search
    search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=6)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # all features:
    search.best_score_ # all features:
    results.append({'model': model, 'best_params': search.best_params_, 'best_score': -search.best_score_})

    # GradientBoostingRegressor
    model = GradientBoostingRegressor()
    # define search space
    params = dict()
    params['loss'] = ['ls', 'lad', 'huber', 'quantile']
    params['learning_rate'] = uniform(loc=0, scale=1)
    params['n_estimators'] = [50, 100, 200, 400, 600]
    params['subsample'] = uniform(loc=0, scale=1)
    params['min_samples_split'] = uniform(loc=0, scale=1)
    params['min_samples_leaf'] = uniform(loc=0, scale=0.5)
    params['min_weight_fraction_leaf'] = uniform(loc=0, scale=0.5)
    params['max_depth'] = [2, 4, 8, 16, 32]
    params['min_impurity_decrease'] = uniform(loc=0, scale=1)
    params['max_features'] = uniform(loc=0, scale=1)
    params['alpha'] = uniform(loc=0, scale=1)
    params['max_leaf_nodes'] = [8, 16, 32, 64]
    params['ccp_alpha'] = uniform(loc=0, scale=1)
    # define the search
    search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=5000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=6)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # all features:
    search.best_score_ # all features:
    results.append({'model': model, 'best_params': search.best_params_, 'best_score': -search.best_score_})

    # ElasticNetCV
    model = ElasticNetCV(max_iter=10000)
    # define search space
    params = dict()
    params['l1_ratio'] = uniform(loc=0, scale=1)
    params['eps'] = uniform(loc=0, scale=0.2)
    params['n_alphas'] = [50, 100, 200, 400]
    params['fit_intercept'] = [True, False]
    params['normalize'] = [True, False]
    # define the search
    search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=6)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # all features:
    search.best_score_ # all features:
    results.append({'model': model, 'best_params': search.best_params_, 'best_score': -search.best_score_})

    # PLSRegression
    model = PLSRegression(max_iter=10000)
    # define search space
    params = dict()
    params['n_components'] = [1,2,4,8,16,32]
    # define the search
    search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=6)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # all features:
    search.best_score_ # all features:
    results.append({'model': model, 'best_params': search.best_params_, 'best_score': -search.best_score_})

    df_results = pd.DataFrame(results)
    df_results_sorted = df_results.sort_values(by=['best_score'], ascending=True)
    df_results_sorted.to_csv(path_or_buf='C:/Users/angel/git/Observ_models/data/ML/Regression/hyperparameters/results_with_all_features.csv', index=False)

