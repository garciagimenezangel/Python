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
from sklearn.svm import SVR
import seaborn as sns
from scipy.stats import norm
import plotly.express as px
warnings.filterwarnings('ignore')
models_repo = "C:/Users/angel/git/Observ_models/"
# root_folder = models_repo + "data/ML/_initial test - Fill total sampled time/"
root_folder = models_repo + "data/ML/Regression/"

def get_train_data_reduced(n_features):
    return pd.read_csv(root_folder+'train/data_reduced_'+str(n_features)+'.csv')

def get_test_data_reduced(n_features):
    return pd.read_csv(root_folder+'test/data_reduced_'+str(n_features)+'.csv')

def get_train_data_withManagement():
    return pd.read_csv(root_folder+'train/data_prepared_withManag.csv')

def get_test_data_withManagement():
    return pd.read_csv(root_folder+'test/data_prepared_withManag.csv')

def get_train_data_full():
    return pd.read_csv(root_folder+'train/data_prepared.csv')

def get_test_data_full():
    return pd.read_csv(root_folder+'test/data_prepared.csv')

def get_test_data_withIDs():
    return pd.read_csv(root_folder+'test/data_prepared_withIDs.csv')

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
    train_prepared   = get_train_data_reduced(5)
    test_prepared    = get_test_data_reduced(5)
    train_management = get_train_data_withManagement()
    test_management  = get_test_data_withManagement()
    predictors_train = train_prepared.iloc[:,:-1]
    labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
    predictors_test  = test_prepared.iloc[:,:-1]
    labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()

    # Model
    # model = BayesianRidge(alpha_1 = 5.661182937742398, alpha_2 = 8.158544161338462, lambda_1 = 7.509288525874375, lambda_2 = 0.08383802954777253)
    model = HistGradientBoostingRegressor(l2_regularization=0.1923237939031256, learning_rate=0.10551346041298326,
                                              loss='least_absolute_deviation', max_depth=4, max_leaf_nodes=32,
                                              min_samples_leaf=4, warm_start=False)
    # model = HistGradientBoostingRegressor(l2_regularization=0.02021888460670551, learning_rate=0.04277282248041758,
    #                                           loss='least_squares', max_depth=4, max_leaf_nodes=32, min_samples_leaf=16,
    #                                           warm_start=True)
    # model = SVR(C=2.9468542209755357, coef0=-0.6868465520687694, degree=4, epsilon=0.18702907953343395, gamma=0.1632449384464454, kernel='rbf', shrinking=True)
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
    df = pd.DataFrame({'obs':labels_test, 'pred':yhat, 'is_organic':[ x == 3 for x in test_management.management ]})
    ax.scatter(df['pred'],   df['obs'],   color='blue', alpha=0.5)
    ax.plot(yhat,yhat, alpha=0.5, color='orange',label='y=prediction ML')
    ax.set_xlim(-5.5,0)
    ax.set_xlabel("Prediction ML", fontsize=16)
    ax.set_ylabel("log(Visitation Rate)", fontsize=16)
    ax.legend(loc='best', fontsize=14)
    plt.show()

    # Scatter plot organic vs not-organic
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
    organic_studies = np.array(["Ana_Montero_Castano_Vaccinium_corymbosum_Canada_2018",
                                "Charlie_Nicholson_Vaccinium_corymbosum_USA_2013" ,
                                "Charlie_Nicholson_Vaccinium_corymbosum_USA_2014" ,
                                "Georg_Andersson_Brassica_rapa_Sweden_2010"  ,
                                "Smitha_Krishnan_Coffea_canephora_India_2014" ,
                                "Virginie_Boreux_Malus_domestica_Germany_2015" ])
    rule_out_studies = np.array(["Blande_Viana_Passiflora_edulis_Brazil_2005"])
    check_data = get_test_data_withIDs()
    ruled_out = [ (rule_out_studies == x).any() for x in check_data.study_id ]
    is_organic = pd.DataFrame(check_data.study_id.tolist()).isin(organic_studies).any(1).values
    sel_columns = [col for col in test_prepared.columns if not ('x0_' in col)]
    check_data = check_data[ np.append( sel_columns, ['study_id', 'site_id', 'biome_num'] ) ]
    check_data['is_organic'] = is_organic
    df = pd.concat([ check_data, pd.DataFrame(yhat, columns=['predicted']) ], axis=1)
    df = df[ [~x for x in ruled_out] ]
    df['vr_obs'] = np.exp(df.log_visit_rate)
    df['vr_pred'] = np.exp(df.predicted)
    # fig = px.scatter(df, x="vr_pred", y="vr_obs", hover_data=df.columns, color="is_organic", trendline="ols")
    fig = px.scatter(df, x="predicted", y="log_visit_rate", hover_data=df.columns, color="is_organic", trendline="ols")
    # fig = px.scatter(df, x="predicted", y="log_visit_rate", hover_data=df.columns, trendline="ols")
    fig.show()

    # Linear regression
    df = df[~(df.is_organic)]
    obs = np.array(df.log_visit_rate)
    pred = np.array(df.predicted)
    scatter(pred, obs)
    plt.xlabel("Prediction ML", fontsize=16)
    plt.ylabel("log(Visitation Rate)", fontsize=16)
    m, b = np.polyfit(pred, obs, 1)
    plot(pred, m * pred + b)
    X_reg, y_reg = pred.reshape(-1, 1), obs.reshape(-1, 1)
    reg = LinearRegression().fit(X_reg, y_reg)
    reg.score(X_reg, y_reg)

    vr_pred = np.exp(pred)
    vr_obs  = np.exp(obs)
    scatter(vr_pred, vr_obs)
    rel_err = np.abs( (vr_pred - vr_obs)*100/vr_obs )
    mae = mean_absolute_error(vr_obs, vr_pred)
    plt.hist(rel_err, bins='auto')