import pandas as pd
import numpy as np
import warnings
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.pyplot import scatter
from matplotlib.pyplot import plot
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy import stats
from sklearn.svm import SVR
warnings.filterwarnings('ignore')

def get_train_data_reduced_58():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/train/"
    return pd.read_csv(data_dir+'data_reduced_58.csv')

def get_test_data_reduced_58():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/test/"
    return pd.read_csv(data_dir+'data_reduced_58.csv')

def get_train_data_reduced_10():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/train/"
    return pd.read_csv(data_dir+'data_reduced_10.csv')

def get_test_data_reduced_10():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/test/"
    return pd.read_csv(data_dir+'data_reduced_10.csv')

def get_train_data_reduced_7():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/train/"
    return pd.read_csv(data_dir+'data_reduced_7.csv')

def get_test_data_reduced_7():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/test/"
    return pd.read_csv(data_dir+'data_reduced_7.csv')

def get_train_data_reduced_27():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/train/"
    return pd.read_csv(data_dir+'data_reduced_27.csv')

def get_test_data_reduced_27():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/test/"
    return pd.read_csv(data_dir+'data_reduced_27.csv')

def get_train_data_reduced_6():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/train/"
    return pd.read_csv(data_dir+'data_reduced_6.csv')

def get_test_data_reduced_6():
    models_repo    = "C:/Users/angel/git/Observ_models/"
    data_dir   = models_repo + "data/ML/Regression/test/"
    return pd.read_csv(data_dir+'data_reduced_6.csv')

if __name__ == '__main__':
    train_prepared   = get_train_data_reduced_7()
    test_prepared    = get_test_data_reduced_7()
    predictors_train = train_prepared.iloc[:,:-1]
    labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()
    predictors_test  = test_prepared.iloc[:,:-1]
    labels_test      = np.array(test_prepared.iloc[:,-1:]).flatten()

    # Support Vector Regressor
    model = SVR(C=1.7, coef0=-0.33, epsilon=0.09, gamma=0.14, kernel='rbf')
    model.fit(predictors_train, labels_train)
    yhat = model.predict(predictors_test)

    # Linear regression
    scatter(yhat, labels_test)
    plt.xlabel("Prediction ML")
    plt.ylabel("log(Observed abundance)")
    plt.xlim(-5, 1)
    plt.ylim(-7, 2)
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

    # Interactive plot
    import plotly.express as px
    df = pd.concat([get_test_data_reduced_27(), pd.DataFrame(yhat, columns=['predicted']) ], axis=1)
    fig = px.scatter(df, x="predicted", y="log_visit_rate", hover_data=df.columns)
    fig.show()