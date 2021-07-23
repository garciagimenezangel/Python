import numpy as np
import pandas as pd
import warnings
import pickle
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor, \
    HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, TweedieRegressor, BayesianRidge, OrthogonalMatchingPursuitCV
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
    # data_prepared = get_data_prepared()
    data_prepared = get_data_reduced(15)
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
    # 1 HistGradientBoostingRegressor
    # 2 ExtraTreesRegressor
    # 3 AdaBoostRegressor
    # 4 RandomForestRegressor
    # 5 SVR

    ########################
    # Hyperparameter tuning
    # I use randomized search over a (small) set of parameters, to get the best score. I repeat the process several
    # times, using a parameter space "surrounding" the best parameters in the previous step
    ########################
    # Note: BayesSearchCV in currently latest version of scikit-optimize not compatible with scikit-learn 0.24.1
    # When scikit-optimize version 0.9.0 is available (currently in development), use: BayesSearchCV(model,params,cv=5)
    # SVR
    model = SVR()
    # define search space
    params = dict()
    params['kernel']  = ['linear', 'poly', 'rbf', 'sigmoid']
    params['C']       = uniform(loc=0, scale=4)
    params['gamma']   = uniform(loc=0, scale=0.5)
    params['coef0']   = uniform(loc=-1, scale=1)
    params['epsilon'] = uniform(loc=0, scale=0.2)
    params['degree']  = [3,4,5]
    params['shrinking'] = [False, True]
    # define the search
    search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=6)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(-mean_score, params)
    search.best_params_ # all features: {'C': 2.9468542209755357, 'coef0': -0.6868465520687694, 'degree': 4, 'epsilon': 0.18702907953343395, 'gamma': 0.1632449384464454, 'kernel': 'rbf', 'shrinking': True}
                        # 15 features: {'C': 2.9468542209755357, 'coef0': -0.6868465520687694, 'degree': 4, 'epsilon': 0.18702907953343395, 'gamma': 0.1632449384464454, 'kernel': 'rbf', 'shrinking': True}
                        # 22 features: {'C': 1.5393618387949028, 'coef0': -0.46947890084948296, 'degree': 3, 'epsilon': 0.19800137347940394, 'gamma': 0.12523398796877383, 'kernel': 'rbf', 'shrinking': False}
    search.best_score_  # all features: -0.9462104354789641
                        # 15 features: -0.87222462616609
                        # 22 features: -0.9023594174493642

    # HistGradientBoostingRegressor
    model = HistGradientBoostingRegressor()
    # define search space
    params = dict()
    params['loss'] = ['least_squares', 'least_absolute_deviation']
    params['learning_rate'] = uniform(loc=0, scale=1)
    params['max_leaf_nodes'] = [8,16,32,64]
    params['max_depth'] = [2,4,8,16,32]
    params['min_samples_leaf']  = [2,4,8,16,32]
    params['l2_regularization']  = uniform(loc=0, scale=1)
    params['warm_start']  = [False, True]
    # define the search
    search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=6)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(-mean_score, params)
    search.best_params_ # all features: {'l2_regularization': 0.02021888460670551, 'learning_rate': 0.04277282248041758, 'loss': 'least_squares', 'max_depth': 4, 'max_leaf_nodes': 32, 'min_samples_leaf': 16, 'warm_start': True}
                        # 5 features: {'l2_regularization': 0.1923237939031256, 'learning_rate': 0.10551346041298326, 'loss': 'least_absolute_deviation', 'max_depth': 4, 'max_leaf_nodes': 32, 'min_samples_leaf': 4, 'warm_start': False}
                        # 26 features: {'l2_regularization': 0.02021888460670551, 'learning_rate': 0.04277282248041758, 'loss': 'least_squares', 'max_depth': 4, 'max_leaf_nodes': 32, 'min_samples_leaf': 16, 'warm_start': True}
                        # 22 features: {'l2_regularization': 0.34991994773700774, 'learning_rate': 0.048052417381370005, 'loss': 'least_absolute_deviation', 'max_depth': 16, 'max_leaf_nodes': 32, 'min_samples_leaf': 16, 'warm_start': False}
                        # 4 features: {'l2_regularization': 0.7045103062427752, 'learning_rate': 0.04615711070677131, 'loss': 'least_absolute_deviation', 'max_depth': 4, 'max_leaf_nodes': 32, 'min_samples_leaf': 8, 'warm_start': True}
    search.best_score_ # all features: -0.9383553540061313
                        # 5 features: -0.8809402223744808
                        # 26 features: -0.863443455817459
                        # 22 features: -0.9447383447363904
                        # 4 features: -0.9888690062754308
                        # 6 features: -1.0022852600534802

    # ExtraTreesRegressor
    model = ExtraTreesRegressor()
    # define search space
    params = dict()
    params['n_estimators'] = [50,100,200,400,600]
    params['criterion'] = ['mse', 'mae']
    params['max_depth'] = [2,4,8,16,32]
    params['min_samples_split']  = uniform(loc=0, scale=10)
    params['min_samples_leaf']  = uniform(loc=0, scale=10)
    params['min_weight_fraction_leaf']  = uniform(loc=0, scale=1)
    params['max_leaf_nodes']  = [8,16,32,64]
    params['min_impurity_decrease']  = uniform(loc=0, scale=1)
    params['bootstrap']  = [False, True]
    params['warm_start']  = [False, True]
    params['ccp_alpha'] = uniform(loc=0, scale=1)
    # define the search
    search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=6)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # {'bootstrap': False, 'ccp_alpha': 0.019454451882791046, 'criterion': 'mae', 'max_depth': 8, 'max_leaf_nodes': 32, 'min_impurity_decrease': 0.49865469993092115, 'min_samples_leaf': 0.38736893138328954, 'min_samples_split': 0.5708381504562543, 'min_weight_fraction_leaf': 0.11618121903130718, 'n_estimators': 200, 'warm_start': False}
    search.best_score_ # -1.1432050688125917

    # AdaBoostRegressor
    # 2 ExtraTreesRegressor
    # 3 AdaBoostRegressor
    # 4 RandomForestRegressor
    model = AdaBoostRegressor()
    # define search space
    params = dict()
    estimator1 = HistGradientBoostingRegressor(l2_regularization=0.02021888460670551,learning_rate=0.04277282248041758,loss='least_squares',max_depth=4,max_leaf_nodes=32,min_samples_leaf=16,warm_start=True)
    estimator2 = ExtraTreesRegressor(bootstrap=False,ccp_alpha=0.019454451882791046,criterion='mae',max_depth=8,max_leaf_nodes=32,min_impurity_decrease=0.49865469993092115,min_samples_leaf=0.38736893138328954,min_samples_split=0.5708381504562543,min_weight_fraction_leaf=0.11618121903130718,n_estimators=200,warm_start=False)
    estimator3 = RandomForestRegressor(bootstrap=True,ccp_alpha=0.040560610473318826,criterion='mse',max_depth=4,max_leaf_nodes=64,min_impurity_decrease=0.003427302639933183,min_samples_split=0.22116336729743802,min_weight_fraction_leaf=0.086566739859892,n_estimators=400,warm_start=False)
    params['base_estimator'] = [estimator1, estimator2, estimator3]
    params['n_estimators']  = [50,100,200,400,600]
    params['learning_rate'] = uniform(loc=0, scale=1)
    params['loss'] = ['linear', 'square', 'exponential']
    # define the search
    search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=100,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=6)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ #{'base_estimator': HistGradientBoostingRegressor(l2_regularization=0.02021888460670551,
                              # learning_rate=0.04277282248041758, max_depth=4,
                              # max_leaf_nodes=32, min_samples_leaf=16,
                              # warm_start=True), 'learning_rate': 0.21875144982480377, 'loss': 'linear', 'n_estimators': 50}
    search.best_score_ #-0.9623461251009882

    # RandomForestRegressor
    model = RandomForestRegressor()
    # define search space
    params = dict()
    params['n_estimators'] = [50, 100, 200, 400, 600]
    params['criterion'] = ['mse', 'mae']
    params['max_depth'] = [2, 4, 8, 16, 32]
    params['min_samples_split'] = uniform(loc=0, scale=1)
    params['min_weight_fraction_leaf'] = uniform(loc=0, scale=0.5)
    params['max_leaf_nodes'] = [8, 16, 32, 64]
    params['min_impurity_decrease'] = uniform(loc=0, scale=1)
    params['bootstrap'] = [False, True]
    params['warm_start'] = [False, True]
    params['ccp_alpha'] = uniform(loc=0, scale=1)
    # define the search
    search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=6)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # {'bootstrap': True, 'ccp_alpha': 0.040560610473318826, 'criterion': 'mse', 'max_depth': 4, 'max_leaf_nodes': 64, 'min_impurity_decrease': 0.003427302639933183, 'min_samples_split': 0.22116336729743802, 'min_weight_fraction_leaf': 0.086566739859892, 'n_estimators': 400, 'warm_start': False}
    search.best_score_ # -1.041137728017587

