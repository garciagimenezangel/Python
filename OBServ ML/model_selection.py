import numpy as np
import pandas as pd
import warnings
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
warnings.filterwarnings('ignore')

def get_predictors_and_labels():
    modelsRepo    = "C:/Users/angel/git/Observ_models/"
    dataDir   = modelsRepo + "data/ML_preprocessing/"
    return ( pd.read_csv(dataDir+'predictors_prepared.csv'), pd.read_csv(dataDir+'labels.csv') )

if __name__ == '__main__':
    predictors, labels = get_predictors_and_labels()

    # LIST OF ESTIMATORS OF TYPE "REGRESSOR" (TRY ALL?)
    from sklearn.utils import all_estimators
    estimators = all_estimators(type_filter='regressor')
    all_regs = []
    for name, RegressorClass in estimators:
        try:
            print('Appending', name)
            reg = RegressorClass()
            all_regs.append(reg)
        except Exception as e:
            print(e)


    tree_reg = DecisionTreeRegressor()
    tree_reg.fit(predictors, labels)

    abundance_predictions = tree_reg.predict(predictors)
    tree_mse = mean_squared_error(labels, abundance_predictions)
    tree_rmse = np.sqrt(tree_mse)