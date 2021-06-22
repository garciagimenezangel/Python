import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import RFE, RFECV
import warnings
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.svm import SVR
import datetime
warnings.filterwarnings('ignore')

def get_train_data_prepared():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML_preprocessing/train/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def get_test_data_prepared():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML_preprocessing/test/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def get_train_data_prepared_with_mechanistic():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML_preprocessing/train/"
    return pd.read_csv(data_dir+'data_prepared_with_mech.csv')

def get_test_data_prepared_with_mechanistic():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML_preprocessing/test/"
    return pd.read_csv(data_dir+'data_prepared_with_mech.csv')

def evaluate_model_rfe(model, predictors, labels, n_features=50):
    rfe = RFE(estimator=model, n_features_to_select=n_features)
    rfe.fit(predictors, labels)
    predictors_reduced = predictors[ predictors.columns[rfe.support_] ]
    scores = cross_val_score(model, predictors_reduced, labels, scoring="neg_mean_absolute_error", cv=5, n_jobs=-1)
    return np.sqrt(-scores)

def evaluate_model_sfs(model, predictors, labels, direction='backward', n_features=50, n_jobs=-1):
    sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=n_features, cv=5, direction=direction, n_jobs=n_jobs)
    sfs.fit(predictors, labels)
    predictors_reduced = predictors[ predictors.columns[sfs.support_] ]
    scores = cross_val_score(model, predictors_reduced, labels, scoring="neg_mean_absolute_error", cv=5, n_jobs=n_jobs)
    return np.sqrt(-scores)

if __name__ == '__main__':
    # train_prepared = get_train_data_prepared()
    # test_prepared = get_test_data_prepared()
    # Including mechanistic model value
    train_prepared = get_train_data_prepared_with_mechanistic()
    test_prepared = get_test_data_prepared_with_mechanistic()

    # Get predictors and labels
    predictors_train = train_prepared.iloc[:,:-1]
    predictors_test  = test_prepared.iloc[:,:-1]
    labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()

    #######################################
    # Feature importance
    #######################################
    # TODO: use other model among the ones tried in model_selection (SVR with rbf doesn't allow the extraction of feature importance). See: https://stats.stackexchange.com/questions/265656/is-there-a-way-to-determine-the-important-features-weight-for-an-svm-that-uses)
    # model = RandomForestRegressor(n_estimators=120, min_samples_split=3, min_samples_leaf=4, bootstrap=True) # parameters found in 'model_selection'
    # model.fit(predictors_train, labels_train)
    # feature_names = np.array(['bio01', 'bio02', 'bio03', 'bio04', 'bio05', 'bio06', 'bio07', 'bio08',
    #     'bio09', 'bio10', 'bio11', 'bio12', 'bio13', 'bio14', 'bio15', 'bio16',
    #     'bio17', 'bio18', 'bio19', 'chili', 'def', 'dist_seminat', 'ec', 'ei',
    #     'elevation', 'es', 'et', 'gHM', 'gpp', 'le', 'pdsi', 'pet', 'ple', 'ro',
    #     'soil', 'soil_carbon_b10', 'soil_carbon_b200', 'soil_clay_b10',
    #     'soil_clay_b200', 'soil_den_b10', 'soil_den_b200', 'soil_pH_b10',
    #     'soil_pH_b200', 'soil_sand_b10', 'soil_sand_b200', 'soil_water_b10',
    #     'soil_water_b200', 'srad', 'swe', 'topo_div',
    #     'vpd', 'vs', 'activity', 'bare', 'cropland', 'grass', 'moss', 'shrub',
    #     'tree', 'urban', 'management', 'x0_1.0', 'x0_2.0', 'x0_4.0', 'x0_5.0',
    #     'x0_6.0', 'x0_7.0', 'x0_8.0', 'x0_10.0', 'x0_12.0',
    #     'x1_Abelmoschus esculentus', 'x1_Allium porrum',
    #     'x1_Anacardium occidentale', 'x1_Bixa orellana', 'x1_Brassica napus',
    #     'x1_Brassica rapa', 'x1_Citrullus lanatus', 'x1_Citrus limon',
    #     'x1_Coffea arabica', 'x1_Coffea canephora', 'x1_Cucumis melo',
    #     'x1_Cucumis sativus', 'x1_Cucurbita pepo', 'x1_Fagopyrum esculentum',
    #     'x1_Fragaria x ananassa', 'x1_Gossypium hirsutum',
    #     'x1_Helianthus annuus', 'x1_Lagenaria siceraria', 'x1_Luffa acutangula',
    #     'x1_Macadamia integrifolia', 'x1_Malpighia emarginata',
    #     'x1_Malus domestica', 'x1_Mangifera indica', 'x1_Momordica charantia',
    #     'x1_Passiflora edulis', 'x1_Persea americana',
    #     'x1_Phaseolus vulgaris L.', 'x1_Prunus avium', 'x1_Prunus dulcis',
    #     'x1_Psidium guajava', 'x1_Pyrus communis', 'x1_Rubus idaeus',
    #     'x1_Solanum lycopersicum', 'x1_Trifolium pratense',
    #     'x1_Vaccinium corymbosum', 'x1_Vaccinium macrocarpon',
    #     'x1_Vaccinium meridionale', 'x1_Vicia faba'])
    #     # 'Lonsdorf.Delphi_lcCont1_open1_forEd1_crEd1_div1_ins1max_dist1_suitmult',
    # feature_importance = pd.DataFrame(sorted(zip(model.feature_importances_, feature_names), reverse=True))
    #
    # #######################################
    # # Recursive Feature Elimination (RFE)
    # #######################################
    # model = RandomForestRegressor(n_estimators=120, min_samples_split=3, min_samples_leaf=4, bootstrap=True) # parameters found in 'model_selection'
    # # Explore number of features with rfe
    # min_n = 10
    # max_n = 30
    # results, n_features = list(), list()
    # for i in range(min_n,max_n+1):
    #     scores = evaluate_model_rfe(model, predictors_train, labels_train, n_features=i)
    #     results.append(scores)
    #     n_features.append(i)
    #     print('>%s %.3f (%.3f)' % (i, np.mean(scores), np.std(scores)))
    # pyplot.boxplot(results, labels=n_features, showmeans=True)
    # df_results = pd.DataFrame(list(map(np.ravel, results)))
    # df_results['mean'] = df_results.mean(axis=1)
    # df_results['n_features'] = range(min_n, max_n+1)
    # df_results.to_csv(
    #     path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/feature_selection_RF.csv',
    #     index=False)
    # # n_features ~15 yields good results:
    # rfe = RFE(estimator=model, n_features_to_select=15)
    # rfe.fit(predictors_train, labels_train)
    # predictors_reduced_train = predictors_train[ predictors_train.columns[rfe.support_]]
    # predictors_reduced_test  = predictors_test[ predictors_test.columns[rfe.support_]]
    # predictors_reduced_train.to_csv('C:/Users/angel/git/Observ_models/data/ML_preprocessing/train/predictors_red15RF.csv', index=False)
    # predictors_reduced_test.to_csv('C:/Users/angel/git/Observ_models/data/ML_preprocessing/test/predictors_red15RF.csv', index=False)
    #
    # ##############################################################################
    # # Recursive Feature Elimination and Cross-Validated selection (RFECV)
    # ##############################################################################
    # model = RandomForestRegressor(n_estimators=120, min_samples_split=3, min_samples_leaf=4, bootstrap=True)  # parameters found in 'model_selection'
    # rfecv = RFECV(estimator=model, n_jobs=-1, cv=5, scoring="neg_mean_absolute_error")
    # rfecv.fit(predictors_train, labels_train)
    # predictors_reduced_train = predictors_train[ predictors_train.columns[rfecv.support_]]
    # predictors_reduced_test  = predictors_test[ predictors_test.columns[rfecv.support_]]
    # predictors_reduced_train.to_csv('C:/Users/angel/git/Observ_models/data/ML_preprocessing/train/predictors_redRFECV.csv', index=False)
    # predictors_reduced_test.to_csv('C:/Users/angel/git/Observ_models/data/ML_preprocessing/test/predictors_redRFECV.csv', index=False)

    #######################################
    # SequentialFeatureSelector (SFS)
    #######################################
    model = SVR(C=1.7, coef0=-0.29, epsilon=0.06, gamma=0.11, kernel='rbf')
    # Explore number of features
    min_n = 5
    max_n = 40
    results, n_features = list(), list()
    for i in range(min_n,max_n+1):
        print(datetime.datetime.now())
        scores = evaluate_model_sfs(model, predictors_train, labels_train, n_features=i, direction='forward', n_jobs=6)
        results.append(scores)
        n_features.append(i)
        print('>%s %.3f (%.3f)' % (i, np.mean(scores), np.std(scores)))
    pyplot.boxplot(results, labels=n_features, showmeans=True)
    df_results = pd.DataFrame(list(map(np.ravel, results)))
    df_results['mean'] = df_results.mean(axis=1)
    df_results['n_features'] = range(min_n, max_n+1)
    df_results.to_csv(
        path_or_buf='C:/Users/angel/git/Observ_models/data/ML_preprocessing/feature_selection_SVR_5-40.csv',
        index=False)
    # n_features ~58 yields good score:
    # sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=58, cv=5, direction='forward', n_jobs=6)
    # sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=10, cv=5, direction='forward', n_jobs=6) # test with only 10, that gives not-that-bad score
    sfs = SequentialFeatureSelector(estimator=model, n_features_to_select=20, cv=5, direction='forward', n_jobs=6)
    sfs.fit(predictors_train, labels_train)
    data_reduced_train = train_prepared[ np.append(np.array(predictors_train.columns[sfs.support_]),['log_abundance']) ]
    data_reduced_test  = test_prepared[ np.append(np.array(predictors_test.columns[sfs.support_]),['log_abundance']) ]
    data_reduced_train.to_csv('C:/Users/angel/git/Observ_models/data/ML_preprocessing/train/data_reduced_6.csv', index=False)
    data_reduced_test.to_csv('C:/Users/angel/git/Observ_models/data/ML_preprocessing/test/data_reduced_6.csv', index=False)

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