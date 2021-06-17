import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV
import warnings
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVR

warnings.filterwarnings('ignore')

def get_train_predictors_and_labels():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML_preprocessing/train/"
    return ( pd.read_csv(data_dir+'predictors_prepared.csv'), np.array(pd.read_csv(data_dir+'labels.csv')).flatten() )

def get_test_predictors_and_labels():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML_preprocessing/test/"
    return ( pd.read_csv(data_dir+'predictors_prepared.csv'), np.array(pd.read_csv(data_dir+'labels.csv')).flatten() )

def get_train_predictors_and_labels_with_Lonsdorf():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/Lonsdorf evaluation/Model predictions/"
    return ( pd.read_csv(data_dir+'train_predictors_prepared.csv'), np.array(pd.read_csv(data_dir+'train_labels.csv')).flatten() )

def get_test_predictors_and_labels_with_Lonsdorf():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/Lonsdorf evaluation/Model predictions/"
    return ( pd.read_csv(data_dir+'test_predictors_prepared.csv'), np.array(pd.read_csv(data_dir+'test_labels.csv')).flatten() )

def evaluate_model_rfe(model, predictors, labels, n_features=50):
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(predictors, labels)
    predictors_reduced = predictors[ predictors.columns[rfe.support_] ]
    scores = cross_val_score(model, predictors_reduced, labels, scoring="neg_mean_absolute_error", cv=5, n_jobs=-1)
    return np.sqrt(-scores)

def evaluate_model_sfs(model, predictors, labels, direction='backward', n_features=50):
    sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=n_features, cv=5, direction=direction, n_jobs=-1)
    sfs.fit(predictors, labels)
    predictors_reduced = predictors[ predictors.columns[sfs.support_] ]
    scores = cross_val_score(model, predictors_reduced, labels, scoring="neg_mean_absolute_error", cv=5, n_jobs=-1)
    return np.sqrt(-scores)

if __name__ == '__main__':
    predictors_train, labels_train = get_train_predictors_and_labels()
    predictors_test, labels_test = get_test_predictors_and_labels()
    # Including mechanistic model value
    predictors_train, labels_train = get_train_predictors_and_labels_with_Lonsdorf()
    predictors_test, labels_test = get_test_predictors_and_labels_with_Lonsdorf()

    #######################################
    # Feature importance
    #######################################
    model = RandomForestRegressor(n_estimators=120, min_samples_split=3, min_samples_leaf=4, bootstrap=True) # parameters found in 'model_selection'
    model.fit(predictors_train, labels_train)
    feature_names = np.array(['bio01', 'bio02', 'bio03', 'bio04', 'bio05', 'bio06', 'bio07',
           'bio08', 'bio09', 'bio10', 'bio11', 'bio12', 'bio13', 'bio14',
           'bio15', 'bio16', 'bio17', 'bio18', 'bio19', 'chili', 'def',
           'dist_seminat', 'ec', 'ei', 'elevation', 'es', 'et', 'gHM', 'gpp',
           'le', 'pdsi', 'pet', 'ple', 'ro', 'soil', 'soil_carbon_b10',
           'soil_carbon_b200', 'soil_clay_b10', 'soil_clay_b200',
           'soil_den_b10', 'soil_den_b200', 'soil_pH_b10', 'soil_pH_b200', 'soil_sand_b10', 'soil_sand_b200',
           'soil_water_b10', 'soil_water_b200', 'srad', 'swe', 'topo_div',
           'vap', 'vpd', 'vs', 'activity', 'bare', 'crop', 'grass', 'moss',
           'shrub', 'tree', 'urban',
           'Lonsdorf.Delphi_lcCont1_open1_forEd1_crEd1_div1_ins1max_dist1_suitmult',
           'management', 'x0_1.0', 'x0_2.0', 'x0_4.0', 'x0_5.0', 'x0_6.0', 'x0_7.0', 'x0_8.0', 'x0_10.0', 'x0_12.0'])
    feature_importance = pd.DataFrame(sorted(zip(model.feature_importances_, feature_names), reverse=True))

    #######################################
    # Recursive Feature Elimination (RFE)
    #######################################
    model = RandomForestRegressor(n_estimators=120, min_samples_split=3, min_samples_leaf=4, bootstrap=True) # parameters found in 'model_selection'
    # Explore number of features with rfe
    min_n = 10
    max_n = 30
    results, n_features = list(), list()
    for i in range(min_n,max_n+1):
        scores = evaluate_model_rfe(model, predictors_train, labels_train, n_features=i)
        results.append(scores)
        n_features.append(i)
        print('>%s %.3f (%.3f)' % (i, np.mean(scores), np.std(scores)))
    pyplot.boxplot(results, labels=n_features, showmeans=True)
    df_results = pd.DataFrame(list(map(np.ravel, results)))
    df_results['mean'] = df_results.mean(axis=1)
    df_results['n_features'] = range(min_n, max_n+1)
    df_results.to_csv(
        path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/feature_selection_RF.csv',
        index=False)
    # n_features ~15 yields good results:
    rfe = RFE(estimator=model, n_features_to_select=15)
    rfe.fit(predictors_train, labels_train)
    predictors_reduced_train = predictors_train[ predictors_train.columns[rfe.support_]]
    predictors_reduced_test  = predictors_test[ predictors_test.columns[rfe.support_]]
    predictors_reduced_train.to_csv('C:/Users/angel/git/Observ_models/data/ML_preprocessing/train/predictors_red15RF.csv', index=False)
    predictors_reduced_test.to_csv('C:/Users/angel/git/Observ_models/data/ML_preprocessing/test/predictors_red15RF.csv', index=False)

    ##############################################################################
    # Recursive Feature Elimination and Cross-Validated selection (RFECV)
    ##############################################################################
    model = RandomForestRegressor(n_estimators=120, min_samples_split=3, min_samples_leaf=4, bootstrap=True)  # parameters found in 'model_selection'
    rfecv = RFECV(estimator=model, n_jobs=-1, cv=5, scoring="neg_mean_absolute_error")
    rfecv.fit(predictors_train, labels_train)
    predictors_reduced_train = predictors_train[ predictors_train.columns[rfecv.support_]]
    predictors_reduced_test  = predictors_test[ predictors_test.columns[rfecv.support_]]
    predictors_reduced_train.to_csv('C:/Users/angel/git/Observ_models/data/ML_preprocessing/train/predictors_redRFECV.csv', index=False)
    predictors_reduced_test.to_csv('C:/Users/angel/git/Observ_models/data/ML_preprocessing/test/predictors_redRFECV.csv', index=False)

    #######################################
    # SequentialFeatureSelector (SFS)
    #######################################
    model = SVR(gamma=0.1, C=2.0)
    # Explore number of features
    min_n = 15
    max_n = 30
    results, n_features = list(), list()
    for i in range(min_n,max_n+1):
        scores = evaluate_model_sfs(model, predictors_train, labels_train, n_features=i, direction='forward')
        results.append(scores)
        n_features.append(i)
        print('>%s %.3f (%.3f)' % (i, np.mean(scores), np.std(scores)))
    pyplot.boxplot(results, labels=n_features, showmeans=True)
    df_results = pd.DataFrame(list(map(np.ravel, results)))
    df_results['mean'] = df_results.mean(axis=1)
    df_results['n_features'] = range(15, 31)
    df_results.to_csv(
        path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/feature_selection_SVR.csv',
        index=False)
    # n_features ~30 yields good results:
    sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=30, cv=5, direction='forward', n_jobs=-1)
    sfs.fit(predictors_train, labels_train)
    predictors_reduced_train = predictors_train[ predictors_train.columns[sfs.support_] ]
    predictors_reduced_test  = predictors_test[ predictors_test.columns[sfs.support_]]
    predictors_reduced_train.to_csv('C:/Users/angel/git/Observ_models/data/ML_preprocessing/train/predictors_red30SVR.csv', index=False)
    predictors_reduced_test.to_csv('C:/Users/angel/git/Observ_models/data/ML_preprocessing/test/predictors_red30SVR.csv', index=False)


    # EVALUATE THE MODEL WITH REDUCED PREDICTORS:
    # rfe = RFE(estimator=model, n_features_to_select=n_features)
    # rfe.fit(predictors, labels)
    # predictors_reduced = predictors[ predictors.columns[rfe.support_] ]
    # model.fit(predictors_reduced, labels)
    # ab_predictions = model.predict(predictors_reduced)
    # mse = mean_squared_error(labels, ab_predictions)
    # rmse = np.sqrt(mse)
    # scores = cross_val_score(model, predictors_reduced, labels, scoring="neg_mean_absolute_error", cv=5)
    # rmse_scores = np.sqrt(-scores)
    # print('rmse all '+ str(n_features)+': ', rmse)
    # print('Mean '+ str(n_features)+': ', rmse_scores.mean())
    # print('Std  '+ str(n_features)+': ', rmse_scores.std())