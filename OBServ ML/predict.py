import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
from matplotlib.pyplot import plot
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats

warnings.filterwarnings('ignore')
from sklearn.svm import SVR

def get_train_predictors_and_labels_rf15():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML_preprocessing/train/"
    return ( pd.read_csv(data_dir+'predictors_red15RF.csv'), np.array(pd.read_csv(data_dir+'labels.csv')).flatten() )

def get_test_predictors_and_labels_rf15():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML_preprocessing/test/"
    return ( pd.read_csv(data_dir+'predictors_red15RF.csv'), np.array(pd.read_csv(data_dir+'labels.csv')).flatten() )

def get_train_predictors_and_labels_svr30():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML_preprocessing/train/"
    return ( pd.read_csv(data_dir+'predictors_red30SVR.csv'), np.array(pd.read_csv(data_dir+'labels.csv')).flatten() )

def get_test_predictors_and_labels_svr30():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML_preprocessing/test/"
    return ( pd.read_csv(data_dir+'predictors_red30SVR.csv'), np.array(pd.read_csv(data_dir+'labels.csv')).flatten() )

if __name__ == '__main__':
    predictors_train, labels_train = get_train_predictors_and_labels_svr30()
    predictors_test,  labels_test  = get_test_predictors_and_labels_svr30()
    predictors_train, labels_train = get_train_predictors_and_labels_rf15()
    predictors_test,  labels_test  = get_test_predictors_and_labels_rf15()

    # Random Forest Regressor
    model = RandomForestRegressor(n_estimators=120, min_samples_split=3, min_samples_leaf=4)  # parameters found in 'model_selection'
    model.fit(predictors_train, labels_train)
    yhat = model.predict(predictors_test)

    # Support Vector Regressor
    model = SVR(gamma=0.1, C=2.0) # parameters found in 'model_selection'
    model.fit(predictors_train, labels_train)
    yhat = model.predict(predictors_test)

    # Linear regression
    scatter(yhat, labels_test)
    plt.xlabel("Prediction ML")
    plt.ylabel("log(Observed abundance)")
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
    mae = mean_absolute_error(labels_test, yhat)