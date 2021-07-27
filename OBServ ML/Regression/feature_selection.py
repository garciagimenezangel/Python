import pickle
import pandas as pd
import numpy as np
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.feature_selection import RFE, RFECV
import warnings
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import cross_val_score, cross_validate
from matplotlib import pyplot
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR
from sklearn.inspection import permutation_importance
import datetime
import ast
warnings.filterwarnings('ignore')
models_repo = "C:/Users/angel/git/Observ_models/"

def get_train_data_prepared():
    data_dir   = models_repo + "data/ML/Regression/train/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def get_test_data_prepared():
    data_dir   = models_repo + "data/ML/Regression/test/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def get_best_models(n_features=0):
    data_dir = models_repo + "data/ML/Regression/hyperparameters/"
    if n_features>0:
        return pd.read_csv(data_dir + 'best_scores_'+str(n_features)+'.csv')
    else:
        return pd.read_csv(data_dir + 'best_scores.csv')

def evaluate_model_rfe(model, predictors, labels, n_features=50, cv=5, n_jobs=-1):
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(predictors, labels)
    predictors_reduced = predictors[ predictors.columns[rfe.support_] ]
    return cross_validate(model, predictors_reduced, labels, scoring="neg_mean_absolute_error", cv=cv, n_jobs=n_jobs, return_train_score=True)

def evaluate_model_sfs(model, predictors, labels, direction='backward', n_features=50, n_jobs=-1, cv=5):
    sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=n_features, cv=cv, direction=direction, n_jobs=n_jobs)
    sfs.fit(predictors, labels)
    predictors_reduced = predictors[ predictors.columns[sfs.support_] ]
    return cross_validate(model, predictors_reduced, labels, scoring="neg_mean_absolute_error", cv=cv, n_jobs=n_jobs, return_train_score=True)

if __name__ == '__main__':
    train_prepared = get_train_data_prepared()
    test_prepared = get_test_data_prepared()

    # Get predictors and labels
    predictors_train = train_prepared.iloc[:,:-1]
    predictors_test  = test_prepared.iloc[:,:-1]
    labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()

    # Get best models
    df_best_models = get_best_models()

    # Load custom cross validation
    with open('C:/Users/angel/git/Observ_models/data/ML/Regression/train/myCViterator.pkl', 'rb') as file:
        myCViterator = pickle.load(file)

    #######################################
    # Feature importance with RF
    #######################################
    # TODO: use other model among the ones tried in model_selection (SVR with rbf doesn't allow the extraction of feature importance). See: https://stats.stackexchange.com/questions/265656/is-there-a-way-to-determine-the-important-features-weight-for-an-svm-that-uses)
    model = RandomForestRegressor(max_depth=16, max_leaf_nodes=16, min_impurity_decrease=0.034256187052082665, min_samples_leaf=16, min_samples_split=16, min_weight_fraction_leaf=0.31790461400795267, n_estimators=250, warm_start=False) # parameters found in 'model_selection'
    model.fit(predictors_train, labels_train)
    feature_names = predictors_train.columns
    feature_importance = pd.DataFrame(sorted(zip(model.feature_importances_, feature_names), reverse=True))

    #######################################
    # SequentialFeatureSelector (SFS)
    #######################################
    d = ast.literal_eval(df_best_models.iloc[2].best_params)
    # model = KNeighborsRegressor(weights=d['weights'], p=d['p'], n_neighbors=d['n_neighbors'], leaf_size=10)
    model = NuSVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], nu=d['nu'], kernel=d['kernel'], shrinking=d['shrinking'])
    # model = SVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], epsilon=d['epsilon'], kernel=d['kernel'], shrinking=d['shrinking'])
    # model = MLPRegressor(activation=d['activation'], alpha=d['alpha'], hidden_layer_sizes=d['hidden_layer_sizes'], learning_rate=d['learning_rate'],
    #                      learning_rate_init=d['learning_rate_init'], momentum=d['momentum'], power_t=d['power_t'], max_iter=10000, solver='sgd')
    # Explore number of features
    min_n = 1
    max_n = 30
    results_train, results_test, n_features = list(), list(), list()
    for i in range(min_n,max_n+1):
        print(datetime.datetime.now())
        scores = evaluate_model_sfs(model, predictors_train, labels_train, cv=myCViterator, n_features=i, direction='forward', n_jobs=6)
        results_train.append(-scores['train_score'])
        results_test.append(-scores['test_score'])
        n_features.append(i)
        print('N> %s ; Train> %.3f ; Test> %.3f' % (i, np.mean(scores['train_score']), np.mean(scores['test_score'])))
    df_results_train               = pd.DataFrame(list(map(np.ravel, results_train)))
    df_results_test                = pd.DataFrame(list(map(np.ravel, results_test)))
    df_results_train['mean']       = df_results_train.mean(axis=1)
    df_results_train['n_features'] = range(min_n, max_n+1)
    df_results_test['mean']        = df_results_test.mean(axis=1)
    df_results_test['n_features']  = range(min_n, max_n+1)
    df_results_train.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/hyperparameters/feature_selection_train_SVR_1-30.csv', index=False)
    df_results_test.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/hyperparameters/feature_selection_train_SVR_1-30.csv', index=False)
    # Plot
    scores_train = df_results_train.iloc[:, 0:5].values.tolist()
    scores_test  = df_results_test.iloc[:, 0:5].values.tolist()
    mean_train   = np.mean(scores_train, axis=1)
    sd_train     = np.std(scores_train, axis=1)
    mean_test    = np.mean(scores_test, axis=1)
    sd_test      = np.std(scores_test, axis=1)
    pyplot.plot(df_results_train.n_features, mean_train, 'o-', color='r', label="Training")
    pyplot.plot(df_results_test.n_features,  mean_test,  'o-', color='g', label="Cross-validation")
    pyplot.fill_between(df_results_train.n_features, mean_train - sd_train, mean_train + sd_train, alpha=0.1)
    pyplot.fill_between(df_results_train.n_features, mean_test  - sd_test,  mean_test  + sd_test,  alpha=0.1)
    pyplot.ylabel('MAE', fontsize=16)
    pyplot.xlabel('N features', fontsize=16)
    pyplot.legend(loc="best")
    # Select n_features:
    sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=6, cv=myCViterator, direction='forward', n_jobs=6)
    sfs.fit(predictors_train, labels_train)
    data_reduced_train = train_prepared[ np.append(np.array(predictors_train.columns[sfs.support_]),['log_visit_rate']) ]
    data_reduced_test  = test_prepared[ np.append(np.array(predictors_test.columns[sfs.support_]),['log_visit_rate']) ]
    data_reduced_train.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/train/data_reduced_6.csv', index=False)
    data_reduced_test.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/test/data_reduced_6.csv', index=False)

    # #######################################
    # # Permutation importance
    # #######################################
    model = NuSVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], degree=d['degree'], kernel=d['kernel'], nu=d['nu'], shrinking=d['shrinking'])
    model.fit(predictors_train, labels_train)
    perm_importance = permutation_importance(model, predictors_train, labels_train, random_state=135, n_jobs=6)
    feature_names = predictors_train.columns
    feature_importance = pd.DataFrame(sorted(zip(perm_importance.importances_mean, feature_names), reverse=True))
    pyplot.barh(feature_importance.loc[:,1], feature_importance.loc[:,0])
    # Take XXX features (testing, threshold around XXX)
    features_selected = feature_importance.loc[0:10,1]
    data_reduced_train = train_prepared[np.append(features_selected, ['log_visit_rate'])]
    data_reduced_test  = test_prepared[np.append(features_selected, ['log_visit_rate'])]
    data_reduced_train.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/train/data_reduced_11.csv', index=False)
    data_reduced_test.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/test/data_reduced_11.csv', index=False)

    # #######################################
    # # Recursive Feature Elimination (RFE)
    # #######################################
    model = BayesianRidge(alpha_1 = 5.661182937742398, alpha_2 = 8.158544161338462, lambda_1 = 7.509288525874375, lambda_2 = 0.08383802954777253)
    # model = RandomForestRegressor(n_estimators=120, min_samples_split=3, min_samples_leaf=4, bootstrap=True) # parameters found in 'model_selection'
    # Explore number of features with rfe
    min_n = 3
    max_n = 40
    results, n_features = list(), list()
    for i in range(min_n,max_n+1):
        print(datetime.datetime.now())
        scores = evaluate_model_rfe(model, predictors_train, labels_train, cv=myCViterator, n_features=i, n_jobs=6)
        results.append(scores)
        n_features.append(i)
        print('>%s %.3f (%.3f)' % (i, np.mean(scores[:-1]), np.std(scores[:-1])))
    df_results = pd.DataFrame(list(map(np.ravel, results)))
    df_results['mean'] = df_results.loc[:,0:4].mean(axis=1)
    df_results['n_features'] = range(min_n, max_n+1)
    df_results.rename(columns={5: 'All'}, inplace=True)
    df_results.to_csv(
        path_or_buf='C:/Users/angel/git/Observ_models/data/ML/Regression/hyperparameters/feature_selection_BayesianRidge_3-40.csv',
        index=False)
    # Plot
    scores = df_results.iloc[:, :-3].values.tolist()
    pyplot.boxplot(scores, labels=df_results.n_features, showmeans=True)
    pyplot.ylabel('MAE', fontsize=16)
    pyplot.xlabel('N features', fontsize=16)
    pyplot.plot(df_results.n_features-2, df_results.All, label='Training', color='red')
    pyplot.plot(df_results.n_features-2, df_results[['mean']], label='Validation', color='green')
    pyplot.legend()
    # Select n_features:
    rfe = RFE(estimator=model, n_features_to_select=16)
    rfe.fit(predictors_train, labels_train)
    data_reduced_train = train_prepared[ np.append(np.array(predictors_train.columns[rfe.support_]),['log_visit_rate']) ]
    data_reduced_test  = test_prepared[ np.append(np.array(predictors_test.columns[rfe.support_]),['log_visit_rate']) ]
    data_reduced_train.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/train/data_reduced_16.csv', index=False)
    data_reduced_test.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/test/data_reduced_16.csv', index=False)

    # ##############################################################################
    # # Recursive Feature Elimination and Cross-Validated selection (RFECV)
    # ##############################################################################
    model = BayesianRidge(alpha_1 = 5.661182937742398, alpha_2 = 8.158544161338462, lambda_1 = 7.509288525874375, lambda_2 = 0.08383802954777253)
    # model = RandomForestRegressor(n_estimators=120, min_samples_split=3, min_samples_leaf=4, bootstrap=True)  # parameters found in 'model_selection'
    rfecv = RFECV(estimator=model, n_jobs=-1, cv=myCViterator, scoring="neg_mean_absolute_error")
    rfecv.fit(predictors_train, labels_train)
    data_reduced_train = train_prepared[
        np.append(np.array(predictors_train.columns[rfecv.support_]), ['log_visit_rate'])]
    data_reduced_test = test_prepared[np.append(np.array(predictors_test.columns[rfecv.support_]), ['log_visit_rate'])]
    data_reduced_train.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/train/data_reduced_RFECV.csv',
                              index=False)
    data_reduced_test.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/test/data_reduced_RFECV.csv',
                             index=False)


    # ##############################################################################
    # EVALUATE THE MODEL WITH REDUCED PREDICTORS:
    # ##############################################################################
    predictors_train = data_reduced_train.iloc[:,:-1]
    labels_train     = np.array(data_reduced_train.iloc[:,-1:]).flatten()
    model.fit(predictors_train, labels_train)
    ab_predictions = model.predict(predictors_train)
    mae = mean_absolute_error(labels_train, ab_predictions)
    scores = cross_val_score(model, predictors_train, labels_train, scoring="neg_mean_absolute_error", cv=myCViterator)
    mae_scores = -scores
    print('MAE all: ', mae)
    print('Mean: ', mae_scores.mean())
    print('Std: ', mae_scores.std())

    # Add mechanistic model values
    # predictors_train = pd.concat([predictors_train,
    #                                  pd.DataFrame(predictors_reduced_train['Lonsdorf.Delphi_lcCont1_open1_forEd1_crEd1_div1_ins1max_dist1_suitmult'])],
    #                               axis=1)
    model.fit(predictors_train, labels_train)
    ab_predictions = model.predict(predictors_train)
    mae = mean_absolute_error(labels_train, ab_predictions)
    scores = cross_val_score(model, predictors_train, labels_train, scoring="neg_mean_absolute_error", cv=myCViterator)
    mae_scores = np.sqrt(-scores)
    print('MAE all: ', mae)
    print('Mean: ', mae_scores.mean())
    print('Std: ', mae_scores.std())