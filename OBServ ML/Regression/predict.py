import ast
import pandas as pd
import numpy as np
import warnings
import os
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
from matplotlib.pyplot import plot
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, NuSVR
import seaborn as sns
from scipy.stats import norm
import plotly.express as px
warnings.filterwarnings('ignore')
models_repo = "C:/Users/angel/git/Observ_models/"
# root_folder = models_repo + "data/ML/_initial test - Fill total sampled time/"
root_folder = models_repo + "data/ML/Regression/"

def check_normality(array):
    sns.distplot(array)
    # skewness and kurtosis
    print("Skewness: %f" % array.skew())
    print("Kurtosis: %f" % array.kurt())
    # Check normality log_visit_rate
    sns.distplot(array, fit=norm)
    fig = plt.figure()
    res = stats.probplot(array, plot=plt)

def get_lonsdorf_prediction_files():
    path = models_repo + 'data/Lonsdorf evaluation/Model predictions/'
    return [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and 'lm ' in i]

def get_lonsdorf_predictions(file='lm pred Open forest.csv'):
    return pd.read_csv(models_repo+'data/Lonsdorf evaluation/Model predictions/'+file)

def get_train_data_reduced(n_features):
    return pd.read_csv(root_folder+'train/data_reduced_'+str(n_features)+'.csv')

def get_test_data_reduced(n_features):
    return pd.read_csv(root_folder+'test/data_reduced_'+str(n_features)+'.csv')

def get_train_data_full():
    return pd.read_csv(root_folder+'train/data_prepared.csv')

def get_test_data_full():
    return pd.read_csv(root_folder+'test/data_prepared.csv')

def get_train_data_withIDs():
    return pd.read_csv(root_folder+'train/data_prepared_withIDs.csv')

def get_test_data_withIDs():
    return pd.read_csv(root_folder+'test/data_prepared_withIDs.csv')

def get_best_models(n_features=0):
    data_dir = models_repo + "data/ML/Regression/hyperparameters/"
    if n_features>0:
        return pd.read_csv(data_dir + 'best_scores_'+str(n_features)+'.csv')
    else:
        return pd.read_csv(data_dir + 'best_scores.csv')

def check_normality(data, column):
    sns.distplot(data[column])
    # skewness and kurtosis
    print("Skewness: %f" % data[column].skew()) # Skewness: -0.220768
    print("Kurtosis: %f" % data[column].kurt()) # Kurtosis: -0.168611
    # Check normality log_visit_rate
    sns.distplot(data[column], fit=norm)
    fig = plt.figure()
    res = stats.probplot(data[column], plot=plt)

def compute_svr_predictions(n_features):
    train_prepared   = get_train_data_reduced(n_features)
    test_prepared    = get_test_data_reduced(n_features)
    predictors_train = train_prepared.iloc[:,:-1]
    labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
    predictors_test  = test_prepared.iloc[:,:-1]
    labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()
    df_best_models   = get_best_models(n_features)
    best_model       = df_best_models.loc[df_best_models.model.astype(str) == "SVR()"].iloc[0]
    d     = ast.literal_eval(best_model.best_params)
    model = SVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], epsilon=d['epsilon'], kernel=d['kernel'], shrinking=d['shrinking'])
    model.fit(predictors_train, labels_train)
    yhat  = model.predict(predictors_test)
    return yhat, labels_test

def compute_nusvr_predictions(n_features):
    train_prepared   = get_train_data_reduced(n_features)
    test_prepared    = get_test_data_reduced(n_features)
    predictors_train = train_prepared.iloc[:,:-1]
    labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
    predictors_test  = test_prepared.iloc[:,:-1]
    labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()
    df_best_models   = get_best_models(n_features)
    best_model       = df_best_models.loc[df_best_models.model.astype(str) == "NuSVR()"].iloc[0]
    d     = ast.literal_eval(best_model.best_params)
    model = NuSVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], nu=d['nu'], kernel=d['kernel'], shrinking=d['shrinking'])
    model.fit(predictors_train, labels_train)
    yhat  = model.predict(predictors_test)
    return yhat, labels_test

def compute_mlp_predictions(n_features):
    train_prepared   = get_train_data_reduced(n_features)
    test_prepared    = get_test_data_reduced(n_features)
    predictors_train = train_prepared.iloc[:,:-1]
    labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
    predictors_test  = test_prepared.iloc[:,:-1]
    labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()
    df_best_models   = get_best_models(n_features)
    best_model       = df_best_models.loc[df_best_models.model.astype(str) == "MLPRegressor(max_iter=10000, solver='sgd')"].iloc[0]
    d     = ast.literal_eval(best_model.best_params)
    model = MLPRegressor(activation=d['activation'], alpha=d['alpha'], hidden_layer_sizes=d['hidden_layer_sizes'],
                         learning_rate=d['learning_rate'], learning_rate_init=d['learning_rate_init'], momentum=d['momentum'],
                         power_t=d['power_t'], max_iter=10000, solver='sgd', random_state=135)
    model.fit(predictors_train, labels_train)
    yhat  = model.predict(predictors_test)
    return yhat, labels_test

def compute_svr_stats(n_features):
    yhat, labels_test = compute_svr_predictions(n_features)
    X_reg, y_reg = yhat.reshape(-1, 1), labels_test.reshape(-1, 1)
    mae   = mean_absolute_error(X_reg, y_reg)
    reg   = LinearRegression().fit(X_reg, y_reg)
    r2    = reg.score(X_reg, y_reg)
    slope = reg.coef_[0][0]
    return pd.DataFrame({
        'model': "SVR",
        'n_features': n_features,
        'mae': mae,
        'r2': r2,
        'slope': slope
    }, index=[0])

def compute_nusvr_stats(n_features):
    train_prepared   = get_train_data_reduced(n_features)
    test_prepared    = get_test_data_reduced(n_features)
    predictors_train = train_prepared.iloc[:,:-1]
    labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
    predictors_test  = test_prepared.iloc[:,:-1]
    labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()
    df_best_models   = get_best_models(n_features)
    best_model       = df_best_models.loc[df_best_models.model.astype(str) == "NuSVR()"].iloc[0]
    d     = ast.literal_eval(best_model.best_params)
    model = NuSVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], nu=d['nu'], kernel=d['kernel'], shrinking=d['shrinking'])
    model.fit(predictors_train, labels_train)
    yhat  = model.predict(predictors_test)
    X_reg, y_reg = yhat.reshape(-1, 1), labels_test.reshape(-1, 1)
    mae   = mean_absolute_error(X_reg, y_reg)
    reg   = LinearRegression().fit(X_reg, y_reg)
    r2    = reg.score(X_reg, y_reg)
    slope = reg.coef_[0][0]
    return pd.DataFrame({
        'model': "NuSVR",
        'n_features': n_features,
        'mae': mae,
        'r2': r2,
        'slope': slope
    }, index=[0])

def compute_mlp_stats(n_features):
    train_prepared   = get_train_data_reduced(n_features)
    test_prepared    = get_test_data_reduced(n_features)
    predictors_train = train_prepared.iloc[:,:-1]
    labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
    predictors_test  = test_prepared.iloc[:,:-1]
    labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()
    df_best_models   = get_best_models(n_features)
    best_model       = df_best_models.loc[df_best_models.model.astype(str) == "MLPRegressor(max_iter=10000, solver='sgd')"].iloc[0]
    d     = ast.literal_eval(best_model.best_params)
    model = MLPRegressor(activation=d['activation'], alpha=d['alpha'], hidden_layer_sizes=d['hidden_layer_sizes'],
                         learning_rate=d['learning_rate'], learning_rate_init=d['learning_rate_init'], momentum=d['momentum'],
                         power_t=d['power_t'], max_iter=10000, solver='sgd', random_state=135)
    model.fit(predictors_train, labels_train)
    yhat  = model.predict(predictors_test)
    X_reg, y_reg = yhat.reshape(-1, 1), labels_test.reshape(-1, 1)
    mae   = mean_absolute_error(X_reg, y_reg)
    reg   = LinearRegression().fit(X_reg, y_reg)
    r2    = reg.score(X_reg, y_reg)
    slope = reg.coef_[0][0]
    return pd.DataFrame({
        'model': "MLP",
        'n_features': n_features,
        'mae': mae,
        'r2': r2,
        'slope': slope
    }, index=[0])

def compute_lons_stats():
    results = pd.DataFrame(columns=['model', 'mae', 'r2', 'slope'])
    files = get_lonsdorf_prediction_files()
    for file in files:
        df_lons = get_lonsdorf_predictions(file)
        X_reg, y_reg = np.array(df_lons.lm_predicted), np.array(df_lons.log_visit_rate)
        mae = mean_absolute_error(X_reg, y_reg)
        reg = LinearRegression().fit(X_reg.reshape(-1, 1), y_reg.reshape(-1, 1))
        r2 = reg.score(X_reg.reshape(-1, 1), y_reg.reshape(-1, 1))
        slope = reg.coef_[0][0]
        model = df_lons.iloc[0].model
        model_res = pd.DataFrame({'model': model, 'mae': mae, 'r2':r2, 'slope':slope}, index=[0])
        results = pd.concat([results, model_res], axis=0, ignore_index=True)
    return results

def compute_combined_stats(n_features):
    df_lons = get_lonsdorf_predictions()
    yhat, labels_test = compute_nusvr_predictions(n_features)
    test_withIDs = get_test_data_withIDs()
    df_ml = pd.DataFrame({'obs':labels_test, 'pred':yhat, 'study_id':test_withIDs.study_id, 'site_id':test_withIDs.site_id})
    df_combined = pd.merge(df_lons, df_ml, on=['study_id', 'site_id'])
    df_combined['yhat'] = df_combined[['pred', 'lm_predicted']].mean(axis=1)
    X_reg, y_reg = np.array(df_combined.yhat), np.array(df_combined.log_visit_rate)
    mae = mean_absolute_error(X_reg, y_reg)
    reg = LinearRegression().fit(X_reg.reshape(-1,1), y_reg.reshape(-1,1))
    r2 = reg.score(X_reg.reshape(-1,1), y_reg.reshape(-1,1))
    slope = reg.coef_[0][0]
    model = "Average("+df_lons.iloc[0].model + ", NuSVR)"
    return pd.DataFrame({
        'model': model,
        'n_features': n_features,
        'mae': mae,
        'r2': r2,
        'slope': slope
    }, index=[0])

def get_mechanistic_values(model_name):
    data_dir = "C:/Users/angel/git/Observ_models/data/"
    return pd.read_csv(data_dir + 'model_data_lite.csv')[['site_id','study_id',model_name]]

# def compute_ml_with_lons(n_features, model_name='Lonsdorf.Delphi_lcCont1_open0_forEd0_crEd0_div0_ins0max_dist0_suitmult'):
#     train_prepared   = get_train_data_reduced(n_features)
#     test_prepared    = get_test_data_reduced(n_features)
#     predictors_train = train_prepared.iloc[:,:-1]
#     labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
#     predictors_test  = test_prepared.iloc[:,:-1]
#     labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()
#     train_with_IDs   = get_train_data_withIDs()
#     train_prepared['study_id'] = train_with_IDs.study_id
#     train_prepared['site_id']  = train_with_IDs.site_id
#     train_mech_values= get_mechanistic_values(model_name)
#     train_prepared   = pd.merge(train_prepared, train_mech_values, on=['study_id','site_id'])
#     train_prepared.drop(columns=['study_id','site_id'], inplace=True)
#
#
#     return data.merge(model_data, on=['study_id', 'site_id'])
#
#
#     df_best_models   = get_best_models(n_features)
#     best_model       = df_best_models.loc[df_best_models.model.astype(str) == "SVR()"].iloc[0]
#     d     = ast.literal_eval(best_model.best_params)
#     model = SVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], epsilon=d['epsilon'], kernel=d['kernel'], shrinking=d['shrinking'])
#     model.fit(predictors_train, labels_train)
#     yhat  = model.predict(predictors_test)

if __name__ == '__main__':

    train_prepared   = get_train_data_reduced(14)
    test_prepared    = get_test_data_reduced(14)
    # train_prepared   = get_train_data_full()
    # test_prepared    = get_test_data_full()
    predictors_train = train_prepared.iloc[:,:-1]
    labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
    predictors_test  = test_prepared.iloc[:,:-1]
    labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()

    # Stats ( MAE, R2, Slope: for a few ml and all mechanistic configurations )
    # TODO> crear las siguientes funciones:
    #     compute_ml_stats(n_features) -> svr, nusvr, mlp (n_features) DONE
    #     compute_lons_stats() DONE
    #     compute_combined_stats(file, n_features) -> combine lons y ml, using average for example -> DONE
    #     compute_ml_with_lons(file, n_features) -> compute ml but adding the value of a mech. model as a predictor
    svr_stats   = compute_svr_stats(14)
    svr_stats['type'] = "ML"
    nusvr_stats = compute_nusvr_stats(14)
    nusvr_stats['type'] = "ML"
    mlp_stats   = compute_mlp_stats(14)
    mlp_stats['type'] = "ML"
    comb_stats  = compute_combined_stats(14)
    comb_stats['type'] = "Combination"
    lons_stats  = compute_lons_stats()
    lons_stats['type']  = "Mechanistic"
    all_stats   = pd.concat([svr_stats, nusvr_stats, mlp_stats, lons_stats, comb_stats], axis=0, ignore_index=True).drop(columns=['n_features'])
    cols = all_stats.columns.tolist()
    cols = cols[-1:] + cols[:-1]
    all_stats = all_stats[cols]
    print(all_stats.to_latex(index=False, float_format='%.2f'))

    # Plots
    df_best_models = get_best_models(14)
    best_model       = df_best_models.loc[df_best_models.model.astype(str) == "NuSVR()"].iloc[0]
    d     = ast.literal_eval(best_model.best_params)
    model = NuSVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], nu=d['nu'], kernel=d['kernel'], shrinking=d['shrinking'])
    model.fit(predictors_train, labels_train)
    yhat = model.predict(predictors_test)

    # Observed versus predicted
    fig, ax           = plt.subplots()
    df_ml             = pd.DataFrame({'obs':labels_test, 'pred':yhat})
    df_ml['source']   = 'ML'
    df_lons           = get_lonsdorf_predictions()[['log_visit_rate','lm_predicted']]
    df_lons['source'] = 'Mechanistic'
    df_lons.columns   = df_ml.columns
    limits_obs        = np.array([np.min(df_lons['obs']) - 0.5, np.max(df_lons['obs']) + 0.5])
    limits_pred       = np.array([np.min(df_lons['pred']) - 0.4, np.max(df_lons['pred']) + 0.4])
    m_ml, b_ml        = np.polyfit(df_ml.pred  , df_ml.obs  , 1)
    m_lons, b_lons    = np.polyfit(df_lons.pred, df_lons.obs, 1)
    ax.scatter(df_lons['pred'], df_lons['obs'],  color='green', alpha=0.5, label="Mechanistic")       # predictions mechanistic
    ax.scatter(df_ml['pred'],   df_ml['obs'],    color='red',   alpha=0.5, label="Machine Learning")  # predictions ml
    ax.plot(limits_obs, limits_obs, alpha=0.5, color='orange',label='observed=prediction')            # obs=pred
    plt.plot(limits_pred, m_lons * limits_pred + b_lons, color='green')   # linear reg mechanistic
    plt.plot(limits_pred, m_ml   * limits_pred + b_ml, color='red')     # linear reg ml
    ax.set_xlim(limits_pred[0], limits_pred[1])
    ax.set_ylim(limits_obs[0], limits_obs[1])
    ax.set_xlabel("Prediction", fontsize=16)
    ax.set_ylabel("log(Visitation Rate)", fontsize=16)
    ax.legend(loc='best', fontsize=14)
    plt.show()
    plt.savefig('C:/Users/angel/git/Observ_models/report/figures/temp/predictions.tiff', format='tiff')

    # Stats ( MAE, R2, Slope: for a few ml and all mechanistic configurations )
    # TODO> crear las siguientes funciones:
    #     compute_ml_stats(n_features) -> svr, nusvr, mlp (n_features) DONE
    #     compute_lons_stats() DONE
    #     compute_combined_stats(file, n_features) -> combine lons y ml, using average for example -> DONE
    #     compute_ml_with_lons(file, n_features) -> compute ml but adding the value of a mech. model as a predictor

    mae_ml   = mean_absolute_error(df_ml.pred, df_ml.obs)
    mae_lons = mean_absolute_error(df_lons.pred, df_lons.obs)
    m_ml, b_ml   = np.polyfit(df_ml.pred, df_ml.obs, 1)
    X_reg, y_reg = np.array(df_ml.pred).reshape(-1, 1), np.array(df_ml.obs).reshape(-1, 1)
    reg   = LinearRegression().fit(X_reg, y_reg)
    r2_ml = reg.score(X_reg, y_reg)
    slope_ml = reg.coef_
    m_lons, b_lons   = np.polyfit(df_lons.pred, df_lons.obs, 1)
    X_reg, y_reg = np.array(df_lons.pred).reshape(-1, 1), np.array(df_lons.obs).reshape(-1, 1)
    reg   = LinearRegression().fit(X_reg, y_reg)
    r2_lons = reg.score(X_reg, y_reg)
    slope_lons = reg.coef_
    stats_names = ['MAE','R2','Slope']
    stats_ml    = [mae_ml, r2_ml, slope_ml]
    stats_lons  = [mae_lons, r2_lons, slope_lons]
    df_stats = pd.DataFrame({'':stats_names, 'Machine Learning':stats_ml, 'Mechanistic':stats_lons})
    df_stats.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/tables/prediction_stats.csv', index=False)
    print(df_stats.to_latex(index=False))

    # Density difference (observed-predicted), organic vs not-organic
    test_management = get_test_data_full()
    kwargs = dict(hist_kws={'alpha': .4}, kde_kws={'linewidth': 1})
    plt.figure()
    df = pd.DataFrame({'obs':labels_test, 'pred':yhat, 'is_organic':[ x == 3 for x in test_management.management ]})
    df_org     = df[ df.is_organic ]
    df_noorg   = df[ [(x==False) for x in df.is_organic] ]
    diff_org   = df_org.obs   - df_org.pred
    diff_noorg = df_noorg.obs - df_noorg.pred
    sns.distplot(diff_org, color="green", label="Organic farming", **kwargs)
    sns.distplot(diff_noorg, color="red", label="Not organic", **kwargs)
    plt.xlabel("(Observed - Predicted)", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.legend()

    # Density difference (observed-predicted), ML vs mechanistic
    kwargs = dict(hist_kws={'alpha': .4}, kde_kws={'linewidth': 1})
    plt.figure()
    df_ml       = pd.DataFrame({'obs':labels_test, 'pred':yhat})
    df_ml['source'] = 'ML'
    df_lons = get_lonsdorf_predictions()
    df_lons['source'] = 'Mechanistic'
    df_lons.columns = df_ml.columns
    diff_ml   = df_ml.obs   - df_ml.pred
    diff_lons = df_lons.obs - df_lons.pred
    sns.distplot(diff_lons, color="green", label="Mechanistic", **kwargs)
    sns.distplot(diff_ml,   color="red",   label="ML", **kwargs)
    plt.xlabel("(Observed - Predicted)", fontsize=16)
    plt.ylabel("Density", fontsize=16)
    plt.legend()

    # Linear regression
    X_reg, y_reg = np.array(df_lons.pred).reshape(-1, 1), np.array(df_lons.obs).reshape(-1, 1)
    reg = LinearRegression().fit(X_reg, y_reg)
    reg.score(X_reg, y_reg)
    X_reg, y_reg = np.array(df_ml.pred).reshape(-1, 1), np.array(df_ml.obs).reshape(-1, 1)
    reg = LinearRegression().fit(X_reg, y_reg)
    reg.score(X_reg, y_reg)

    # Scatter plot organic vs not-organic
    test_management = get_test_data_full()
    fig, ax = plt.subplots()
    df = pd.DataFrame({'obs':labels_test, 'pred':yhat, 'is_organic':[ x == 3 for x in test_management.management ]})
    df_org     = df[ df.is_organic ]
    df_noorg   = df[ [(x==False) for x in df.is_organic] ]
    ax.scatter(df_org['pred'],   df_org['obs'],   color='green', alpha=0.5, label='Organic farming')
    ax.scatter(df_noorg['pred'], df_noorg['obs'], color='red',   alpha=0.5, label='Not organic')
    ax.plot(yhat,yhat, alpha=0.5, color='orange',label='y=prediction ML')
    ax.set_xlim(-5.5,0)
    ax.set_xlabel("Prediction ML", fontsize=16)
    ax.set_ylabel("log(Visitation Rate)", fontsize=16)
    ax.legend()
    plt.show()

    # Interactive plot - organic
    check_data = get_test_data_withIDs()
    test_management = get_test_data_full()
    is_organic = (test_management.management == 3)
    check_data['is_organic'] = is_organic
    df = pd.concat([ check_data, pd.DataFrame(yhat, columns=['predicted']) ], axis=1)
    # fig = px.scatter(df, x="vr_pred", y="vr_obs", hover_data=df.columns, color="is_organic", trendline="ols")
    # fig = px.scatter(df, x="predicted", y="log_visit_rate", hover_data=df.columns, color="is_organic", trendline="ols")
    fig = px.scatter(df, x="predicted", y="log_visit_rate", hover_data=df.columns, trendline="ols")
    fig.show()

    # Interactive plot - lonsdorf
    check_data  = get_test_data_withIDs()
    df_ml       = pd.DataFrame({'obs':labels_test, 'pred':yhat})
    df_ml['source'] = 'ML'
    df_ml = pd.concat([df_ml, check_data], axis=1)
    df_lons = get_lonsdorf_predictions()
    df_lons['source'] = 'Mechanistic'
    df_lons = pd.concat([df_lons, check_data], axis=1)
    df_lons.columns = df_ml.columns
    df = pd.concat([ df_ml, df_lons ], axis=0)
    fig = px.scatter(df, x="pred", y="obs", hover_data=df.columns, color="source", trendline="ols")
    fig.show()





