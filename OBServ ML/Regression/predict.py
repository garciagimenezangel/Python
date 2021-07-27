import ast
import pandas as pd
import numpy as np
import warnings
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
from matplotlib.pyplot import plot
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from sklearn.neighbors import KNeighborsRegressor
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
    print("Skewness: %f" % array.skew()) # Skewness: -0.220768
    print("Kurtosis: %f" % array.kurt()) # Kurtosis: -0.168611
    # Check normality log_visit_rate
    sns.distplot(array, fit=norm)
    fig = plt.figure()
    res = stats.probplot(array, plot=plt)

def get_lonsdorf_predictions():
    return pd.read_csv(models_repo+'data/Lonsdorf evaluation/Model predictions/lm_pred_all.csv')

def get_train_data_reduced(n_features):
    return pd.read_csv(root_folder+'train/data_reduced_'+str(n_features)+'.csv')

def get_test_data_reduced(n_features):
    return pd.read_csv(root_folder+'test/data_reduced_'+str(n_features)+'.csv')

def get_train_data_full():
    return pd.read_csv(root_folder+'train/data_prepared.csv')

def get_test_data_full():
    return pd.read_csv(root_folder+'test/data_prepared.csv')

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

if __name__ == '__main__':
    train_prepared   = get_train_data_reduced(6)
    test_prepared    = get_test_data_reduced(6)
    # train_prepared   = get_train_data_full()
    # test_prepared    = get_test_data_full()
    predictors_train = train_prepared.iloc[:,:-1]
    labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
    predictors_test  = test_prepared.iloc[:,:-1]
    labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()

    # Model
    df_best_models = get_best_models(6)
    d = ast.literal_eval(df_best_models.iloc[0].best_params)
    model = NuSVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], nu=d['nu'], kernel=d['kernel'], shrinking=d['shrinking'])
    model.fit(predictors_train, labels_train)
    yhat = model.predict(predictors_test)

    # Linear regression
    # scatter(yhat, labels_test)
    # plt.xlabel("Prediction ML", fontsize=16)
    # plt.ylabel("log(Visitation Rate)", fontsize=16)
    m, b = np.polyfit(yhat, labels_test, 1)
    # plot(yhat, yhat)
    X_reg, y_reg = yhat.reshape(-1, 1), labels_test.reshape(-1, 1)
    reg = LinearRegression().fit(X_reg, y_reg)
    reg.score(X_reg, y_reg)

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

    # Observed versus predicted
    fig, ax = plt.subplots()
    df_ml       = pd.DataFrame({'obs':labels_test, 'pred':yhat})
    df_ml['source'] = 'ML'
    df_lons = get_lonsdorf_predictions()
    df_lons['source'] = 'Mechanistic'
    df_lons.columns = df_ml.columns
    # ax.scatter(df_lons['pred'], df_lons['obs'],  color='green', alpha=0.5, label="Mechanistic")
    ax.scatter(df_ml['pred'],   df_ml['obs'],    color='red',   alpha=0.5, label="Machine Learning")
    ax.plot(yhat,yhat, alpha=0.5, color='orange',label='y=prediction')
    # ax.set_xlim(-5.5,0)
    ax.set_xlabel("Prediction", fontsize=16)
    ax.set_ylabel("log(Visitation Rate)", fontsize=16)
    ax.legend(loc='best', fontsize=14)
    plt.show()

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

    # Root mean square error
    mse  = mean_squared_error(labels_test, yhat)
    rmse = np.sqrt(mse)
    confidence = 0.95
    squared_errors = (yhat - labels_test)**2
    np.sqrt(stats.t.interval(confidence, len(squared_errors)-1, loc=squared_errors.mean(), scale=stats.sem(squared_errors)))

    # Mean absolute error
    vr_pred = np.exp(yhat)
    vr_obs  = np.exp(labels_test)
    scatter(vr_pred, vr_obs)
    rel_err = np.abs( (vr_pred - vr_obs)*100/vr_obs )
    mae = mean_absolute_error(labels_test, yhat)
    rel_err_mae = np.abs( (labels_test - yhat)*100/labels_test )
    sns.distplot(rel_err_mae)

    # Interactive plot
    check_data = get_test_data_withIDs()
    test_management = get_test_data_full()
    is_organic = (test_management.management == 3)
    check_data['is_organic'] = is_organic
    df = pd.concat([ check_data, pd.DataFrame(yhat, columns=['predicted']) ], axis=1)
    # fig = px.scatter(df, x="vr_pred", y="vr_obs", hover_data=df.columns, color="is_organic", trendline="ols")
    # fig = px.scatter(df, x="predicted", y="log_visit_rate", hover_data=df.columns, color="is_organic", trendline="ols")
    fig = px.scatter(df, x="predicted", y="log_visit_rate", hover_data=df.columns, trendline="ols")
    fig.show()

    # Stats ( MAE, R2, Mu(obs-pred), Sigma(obs-pred) )
    mae_ml   = mean_absolute_error(df_ml.pred, df_ml.obs)
    mae_lons = mean_absolute_error(df_lons.pred, df_lons.obs)
    m_ml, b_ml   = np.polyfit(df_ml.pred, df_ml.obs, 1)
    X_reg, y_reg = np.array(df_ml.pred).reshape(-1, 1), np.array(df_ml.obs).reshape(-1, 1)
    reg   = LinearRegression().fit(X_reg, y_reg)
    r2_ml = reg.score(X_reg, y_reg)
    m_lons, b_lons   = np.polyfit(df_lons.pred, df_lons.obs, 1)
    X_reg, y_reg = np.array(df_lons.pred).reshape(-1, 1), np.array(df_lons.obs).reshape(-1, 1)
    reg   = LinearRegression().fit(X_reg, y_reg)
    r2_lons = reg.score(X_reg, y_reg)
    (mu_ml, sigma_ml) = stats.norm.fit(df_ml.obs - df_ml.pred)
    (mu_lons, sigma_lons) = stats.norm.fit(df_lons.obs - df_lons.pred)
    stats_names = ['MAE','R2','Mean (Observed-Predicted)', 'SD (Observed-Predicted)']
    stats_ml    = [mae_ml, r2_ml, mu_ml, sigma_ml]
    stats_lons  = [mae_lons, r2_lons, mu_lons, sigma_lons]
    df_stats = pd.DataFrame({'':stats_names, 'Machine Learning':stats_ml, 'Mechanistic':stats_lons})
    df_stats.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/tables/prediction_stats.csv', index=False)




