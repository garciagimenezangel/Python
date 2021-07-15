import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, BayesianRidge
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
from matplotlib.pyplot import plot
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from sklearn.svm import SVR
import seaborn as sns
from scipy.stats import norm
warnings.filterwarnings('ignore')

def get_train_data_reduced(n_features):
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/train/"
    return pd.read_csv(data_dir+'data_reduced_'+str(n_features)+'.csv')

def get_test_data_reduced(n_features):
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/test/"
    return pd.read_csv(data_dir+'data_reduced_'+str(n_features)+'.csv')

def get_train_data_withManagement():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/train/"
    return pd.read_csv(data_dir+'data_prepared_withManag.csv')

def get_test_data_withManagement():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/test/"
    return pd.read_csv(data_dir+'data_prepared_withManag.csv')

def get_test_data_testsFolder():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Tests/test/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def get_train_data_testsFolder():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Tests/train/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def get_train_data_full():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/train/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def get_test_data_full():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/test/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def get_test_data_withIDs():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/test/"
    return pd.read_csv(data_dir+'data_prepared_withIDs.csv')

if __name__ == '__main__':
    # train_prepared   = get_train_data_reduced(16)
    # test_prepared    = get_test_data_reduced(16)
    train_prepared = get_train_data_testsFolder()
    test_prepared = get_test_data_testsFolder()
    predictors_train = train_prepared.iloc[:,:-1]
    labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
    predictors_test  = test_prepared.iloc[:,:-1]
    labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()

    # Model
    model = BayesianRidge(alpha_1 = 5.661182937742398, alpha_2 = 8.158544161338462, lambda_1 = 7.509288525874375, lambda_2 = 0.08383802954777253)
    # model = SVR(C=2.41199588236654, epsilon=0.07426154697634271, gamma=0.002148615554943889)
    model.fit(predictors_train, labels_train)
    yhat = model.predict(predictors_test)

    # Linear regression
    scatter(yhat, labels_test)
    plt.xlabel("Prediction ML", fontsize=16)
    plt.ylabel("log(Visitation Rate)", fontsize=16)
    m, b = np.polyfit(yhat, labels_test, 1)
    plot(yhat, m * yhat + b)
    X_reg, y_reg = yhat.reshape(-1, 1), labels_test.reshape(-1, 1)
    reg = LinearRegression().fit(X_reg, y_reg)
    reg.score(X_reg, y_reg)

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
    sns.distplot(rel_err)

    # Interactive plot
    # organic_studies = np.array(["Agustin_Saez_Rubus_idaeus_Argentina_2014",
    #                             "Breno_M_Freitas_Gossypium_hirsutum_Brazil_2011",
    #                             "Juliana_Hipolito_Coffea_arabica_Brazil_2014",
    #                             "Luisa_G_Carvalheiro_Mangifera_indica_South_Africa_2009",
    #                             "Virginie_Boreux_Malus_domestica_Germany_2015"])
    organic_studies = np.array(["Yi_Zou_Brassica_napus_China_2015",
                                "Breno_M_Freitas_Gossypium_hirsutum_Brazil_2011"])
    import plotly.express as px
    check_data = get_test_data_withIDs()
    is_organic = pd.DataFrame(check_data.study_id.tolist()).isin(organic_studies).any(1).values
    sel_columns = [col for col in test_prepared.columns if not ('x0_' in col)]
    check_data = check_data[ np.append( sel_columns, ['study_id', 'site_id', 'biome_num'] ) ]
    check_data['is_organic'] = is_organic
    df = pd.concat([ check_data, pd.DataFrame(yhat, columns=['predicted']) ], axis=1)
    df['vr_obs'] = np.exp(df.log_visit_rate)
    df['vr_pred'] = np.exp(df.predicted)
    # fig = px.scatter(df, x="vr_pred", y="vr_obs", hover_data=df.columns, color="is_organic", trendline="ols")
    fig = px.scatter(df, x="predicted", y="log_visit_rate", hover_data=df.columns, color="is_organic", trendline="ols")
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