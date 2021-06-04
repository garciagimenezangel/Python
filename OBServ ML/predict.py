import pandas as pd
import numpy as np
import warnings
from sklearn.linear_model import Ridge, LinearRegression
from matplotlib.pyplot import scatter
from matplotlib.pyplot import plot
warnings.filterwarnings('ignore')

# Repository
modelsRepo    = "C:/Users/angel/git/Observ_models/"

# Folders
dataDir   = modelsRepo + "data/ML_preprocessing/"

if __name__ == '__main__':
    df_train = pd.read_csv(dataDir+'data_num_train.csv')
    df_test = pd.read_csv(dataDir+'data_num_test.csv')

    # Test simple model, no tuning of hyperparameters, to see how good the predictions are
    model = Ridge(alpha=1.0)
    data = df_train.values
    X, y = data[:, :-1], data[:, -1]
    model.fit(X, y)
    test = df_test.values
    X_test, y_test = test[:, :-1], test[:, -1]

    # Predict
    yhat = model.predict(X_test)

    # Linear regression
    scatter(yhat, y_test)
    m, b = np.polyfit(yhat, y_test, 1)
    plot(yhat, m * yhat + b)
    X_reg, y_reg = yhat.reshape(-1, 1), y_test.reshape(-1, 1)
    reg = LinearRegression().fit(X_reg, y_reg)
    reg.score(X_reg, y_reg)