import numpy as np
import pandas as pd
import warnings
import pickle
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV, LassoCV, TweedieRegressor, BayesianRidge, OrthogonalMatchingPursuitCV
from sklearn.svm import SVR, NuSVR
from sklearn.utils import all_estimators
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from scipy.stats import uniform
warnings.filterwarnings('ignore')

def get_data_prepared():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/train/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def get_data_reduced(n_features):
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/train/"
    return pd.read_csv(data_dir+'data_reduced_'+str(n_features)+'.csv')

if __name__ == '__main__':
    data_prepared = get_data_prepared()
    # data_prepared = get_data_reduced(16)
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
    # 1 SVR
    # 2 OrthogonalMatchingPursuitCV
    # 3 BayesianRidge
    # 4 TweedieRegressor
    # 5 LassoLarsCV

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
    search.best_params_ # cv=5: {'C': 1.7342889543571887, 'coef0': -0.3345352614917637, 'epsilon': 0.09256863132721108, 'gamma': 0.14372942931130184, 'kernel': 'rbf'}; myCViterator: {'C': 0.21797997747706432, 'coef0': -0.4971113982700808, 'degree': 4, 'epsilon': 0.16537034144285234, 'gamma': 0.008623955809822392, 'kernel': 'rbf', 'shrinking': False}
    search.best_score_  # cv=5: -1.0277530513299957; myCViterator: -1.1000191225836446

    # BayesianRidge
    model = BayesianRidge()
    # define search space
    params = dict()
    params['alpha_1']      = uniform(loc=0, scale=20)
    params['alpha_2']      = uniform(loc=0, scale=20)
    params['lambda_1'] = uniform(loc=0, scale=20)
    params['lambda_2'] = uniform(loc=0, scale=20)
    # define the search
    search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=-1)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(-mean_score, params)
    search.best_params_ # {'alpha_1': 5.661182937742398, 'alpha_2': 8.158544161338462, 'lambda_1': 7.509288525874375, 'lambda_2': 0.08383802954777253}
    search.best_score_ # -1.0962985687867208
    # With 16 predictors: # {'alpha_1': 5.661182937742398, 'alpha_2': 8.158544161338462, 'lambda_1': 7.509288525874375, 'lambda_2': 0.08383802954777253}, score: -1.0962985687867208

    # LassoCV
    model = LassoCV()
    # define search space
    params = dict()
    params['eps']      = uniform(loc=0.001, scale=0.02)
    params['n_alphas'] = [1,2,3,4,5]
    # define the search
    search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=-1)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # {'eps': 0.02082979677480987, 'n_alphas': 1}
    search.best_score_ # -1.0906752658680903

    # TweedieRegressor
    model = TweedieRegressor()
    # define search space
    params = dict()
    params['power'] = [0,1,2,3]
    params['alpha'] = uniform(loc=0, scale=2)
    params['fit_intercept'] = [False, True]
    # define the search
    search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=-1)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ #{'alpha': 0.4785355536901701, 'fit_intercept': True, 'power': 0}
    search.best_score_ #-1.096445843001565

    # OrthogonalMatchingPursuitCV
    model = OrthogonalMatchingPursuitCV(max_iter=25)
    # define search space
    # define the search
    scores = cross_val_score(model, predictors, labels, scoring="neg_mean_absolute_error", cv=myCViterator)
    scores.mean()# -1.1441626667132963

    # GradientBoostingRegressor
    model = GradientBoostingRegressor()
    # define search space
    params = dict()
    params['loss'] = ['ls','lad','huber','quantile']
    params['learning_rate'] = uniform(loc=0, scale=1)
    params['n_estimators'] = [2,8,32,128,512,1024]
    params['min_samples_leaf']  = uniform(loc=0, scale=10)
    params['min_samples_split']  = uniform(loc=0, scale=10)
    params['min_weight_fraction_leaf']  = uniform(loc=0, scale=1)
    params['max_depth'] = [2,4,8,16,32]
    params['min_impurity_decrease'] = uniform(loc=0, scale=1)
    params['ccp_alpha'] = uniform(loc=0, scale=1)
    # define the search
    search = RandomizedSearchCV(model, params, cv=myCViterator, scoring='neg_mean_absolute_error', n_iter=1000,
                                return_train_score=True, verbose=2, random_state=135, n_jobs=-1)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # {'ccp_alpha': 0.8321352603958725, 'learning_rate': 0.23342126974413813, 'loss': 'lad', 'max_depth': 16, 'min_impurity_decrease': 0.8033722209318646, 'min_samples_leaf': 0.3534652115561199, 'min_samples_split': 0.015604580711630067, 'min_weight_fraction_leaf': 0.12205305102244157, 'n_estimators': 512}
    search.best_score_ # -1.1813459748984003

