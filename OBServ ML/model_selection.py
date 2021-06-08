import numpy as np
import pandas as pd
import warnings
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.utils import all_estimators
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
warnings.filterwarnings('ignore')

def get_predictors_and_labels():
    modelsRepo    = "C:/Users/angel/git/Observ_models/"
    dataDir   = modelsRepo + "data/ML_preprocessing/"
    return ( pd.read_csv(dataDir+'predictors_prepared.csv'), np.array(pd.read_csv(dataDir+'labels.csv')) )

if __name__ == '__main__':
    predictors, labels = get_predictors_and_labels()

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

    ########################
    # Shortlist: check df_results and see which show low 'mean' and not-too-low 'rmse_all' (sign of possible overfitting)
    #######################
    # Selected estimators (no particular order):
    # 1 RandomForestRegressor
    # 2 GradientBoostingRegressor
    # 3 KNeighborsRegressor
    # 4 SVR

    ########################
    # Hyperparameter tuning
    # I use randomized search over a (small) set of parameters, to get the best score. I repeat the process several
    # times, using a parameter space "surrounding" the best parameters in the previous step
    ########################
    # RandomForestRegressor
    model = RandomForestRegressor()
    # define search space
    params = dict()
    params['n_estimators'] = [110, 120, 130]
    params['min_samples_split'] = [1,2,3]
    params['min_samples_leaf'] = [4]
    params['bootstrap'] = [True]
    # define the search
    search = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', return_train_score=True, verbose=2)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # {'n_estimators': 120, 'min_samples_split': 3, 'min_samples_leaf': 4, 'bootstrap': True}
    search.best_score_  # -0.9836819886013484

    # GradientBoostingRegressor
    model = GradientBoostingRegressor()
    # define search space
    params = dict()
    params['n_estimators'] = [250,500]
    params['min_samples_split'] = [10,15]
    params['min_samples_leaf'] = [2,3,4]
    params['learning_rate'] = [0.2, 0.1, 0.05]
    params['loss'] = ['huber']
    # define the search
    search = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', return_train_score=True, verbose=2)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # {'n_estimators': 500, 'min_samples_split': 15, 'min_samples_leaf': 4, 'loss': 'huber', 'learning_rate': 0.05}
    search.best_score_  # -1.0469668259331786

    # KNeighborsRegressor
    model = KNeighborsRegressor()
    # define search space
    params = dict()
    params['weights'] = ['distance']
    params['leaf_size'] = [40,50,60,70,80,90]
    # define the search
    search = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', return_train_score=True, verbose=2)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_ # {'weights': 'distance', 'leaf_size': 40}
    search.best_score_  # -1.0990870514118136

    # SVR
    model = SVR()
    # define search space
    params = dict()
    #params['kernel'] = ['rbf']
    params['C'] = [0.1, 1.0, 2.0, 4.0]
    params['gamma'] = [0.01, 0.1, 1.0]
    #params['degree'] = [2,3,4]
    # define the search
    search = RandomizedSearchCV(model, params, cv=5, scoring='neg_mean_squared_error', return_train_score=True, verbose=2)
    search.fit(predictors, labels)
    cvres = search.cv_results_
    for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
        print(np.sqrt(-mean_score), params)
    search.best_params_  #{'gamma': 0.1, 'C': 2.0}
    search.best_score_  #-1.1259887075439634
