# Because this dataset contains multicollinear features, the permutation importance will show that none of the features
# are important. One approach to handling multicollinearity is by performing hierarchical clustering on the features’
# Spearman rank-order correlations, picking a threshold, and keeping a single feature from each cluster.
# https://scikit-learn.org/stable/auto_examples/inspection/plot_permutation_importance_multicollinear.html#sphx-glr-auto-examples-inspection-plot-permutation-importance-multicollinear-py
import ast
import pickle
from collections import defaultdict
import pandas as pd
import warnings
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr
from scipy.cluster import hierarchy
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
from sklearn.svm import NuSVR

warnings.filterwarnings('ignore')
models_repo = "C:/Users/angel/git/Observ_models/"

def get_train_data_prepared():
    data_dir   = models_repo + "data/ML/Regression/train/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def get_test_data_prepared():
    data_dir   = models_repo + "data/ML/Regression/test/"
    return pd.read_csv(data_dir+'data_prepared.csv')

def get_feature_importance(model, predictors_train, labels_train):
    model.fit(predictors_train, labels_train)
    perm_importance = permutation_importance(model, predictors_train, labels_train, random_state=135, n_jobs=6)
    feature_names = predictors_train.columns
    return pd.DataFrame(sorted(zip(perm_importance.importances_mean, feature_names), reverse=True))

def get_best_models(n_features=0):
    data_dir = models_repo + "data/ML/Regression/hyperparameters/"
    if n_features>0:
        return pd.read_csv(data_dir + 'best_scores_'+str(n_features)+'.csv')
    else:
        return pd.read_csv(data_dir + 'best_scores.csv')

############################################
train_prepared = get_train_data_prepared()

# Get predictors and labels
predictors_train = train_prepared.iloc[:,:-1]
labels_train     = np.array(train_prepared.iloc[:,-1:]).flatten()

# Load custom cross validation
with open('C:/Users/angel/git/Observ_models/data/ML/Regression/train/myCViterator.pkl', 'rb') as file:
    myCViterator = pickle.load(file)

# Define model
df_best_models = get_best_models()
best_model     = df_best_models.loc[df_best_models.model.astype(str) == "NuSVR()"].iloc[0]
d     = ast.literal_eval(best_model.best_params)
model = NuSVR(C=d['C'], coef0=d['coef0'], gamma=d['gamma'], nu=d['nu'], kernel=d['kernel'], shrinking=d['shrinking'])

# Compute correlation matrix
corr             = spearmanr(predictors_train).correlation
    # Select columns where there is variation of the variables (corr != all(nan))
df_corr          = pd.DataFrame(corr)
df_corr.columns  = predictors_train.columns
sel_cols         = df_corr.columns[~df_corr.isna().all()].tolist()
predictors_train = predictors_train[sel_cols]
    # Compute again correlation without the predictors that do not vary and do not provide any information
corr         = spearmanr(predictors_train).correlation
corr_linkage = hierarchy.ward(corr)
    # Clean columns with no variation (corr=nan)

# Cross-validation to find the best threshold
results = []
thresholds = [0.5, 1, 1.25, 1.5, 2, 3]
for t in thresholds:

    # Select features for threshold=t
    cluster_ids = hierarchy.fcluster(corr_linkage, t, criterion='distance')
    cluster_id_to_feature_ids = defaultdict(list)
    for idx, cluster_id in enumerate(cluster_ids):
        cluster_id_to_feature_ids[cluster_id].append(idx)
    selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
    print("N selected features: " + str(np.size(selected_features)))

    # Compute indicators
    mae_t = []
    r2_t  = []
    slope_t = []
    for ind_train, ind_test in myCViterator:
        features_train = predictors_train.iloc[ind_train]
        features_test  = predictors_train.iloc[ind_test]
        target_train   = np.array(train_prepared.iloc[ind_train, -1:]).flatten()
        target_test    = np.array(train_prepared.iloc[ind_test, -1:]).flatten()
        X_train_sel    = features_train.iloc[:, selected_features]
        X_test_sel     = features_test.iloc[:, selected_features]

        # Predict with the selected features
        model.fit(X_train_sel, target_train)
        yhat    = model.predict(X_test_sel)
        X_reg, y_reg = yhat.reshape(-1, 1), target_test.reshape(-1, 1)
        reg     = LinearRegression().fit(X_reg, y_reg)
        mae_t   = np.concatenate([mae_t,   [mean_absolute_error(X_reg, y_reg)]])
        r2_t    = np.concatenate([r2_t,    [reg.score(X_reg, y_reg)]])
        slope_t = np.concatenate([slope_t, [reg.coef_[0][0]]])

    results.append({
        'threshold': t,
        'mae_mean': np.mean(mae_t),
        'mae_std':  np.std(mae_t),
        'r2_mean': np.mean(r2_t),
        'r2_std':  np.std(r2_t),
        'slope_mean': np.mean(slope_t),
        'slope_std': np.std(slope_t),
    })

# Select features with the selected threshold
df_results = pd.DataFrame(results)
best_result = df_results.loc[df_results['mae_mean'] == np.min(df_results['mae_mean'])]
best_t = best_result.threshold.iloc[0]
corr = spearmanr(predictors_train).correlation
corr_linkage = hierarchy.ward(corr)
cluster_ids = hierarchy.fcluster(corr_linkage, best_t, criterion='distance')
cluster_id_to_feature_ids = defaultdict(list)
for idx, cluster_id in enumerate(cluster_ids):
    cluster_id_to_feature_ids[cluster_id].append(idx)
selected_features = [v[0] for v in cluster_id_to_feature_ids.values()]
selected_features = features_train.iloc[:, selected_features].columns # use names instead of indices

# Save reduced datasets
test_prepared   = get_test_data_prepared()
predictors_test = test_prepared.iloc[:,:-1]

predictors_reduced_train = predictors_train[selected_features]
predictors_reduced_test  = predictors_test[selected_features]
data_reduced_train       = pd.concat([predictors_reduced_train, train_prepared.iloc[:,-1:]], axis=1)
data_reduced_test        = pd.concat([predictors_reduced_test,  test_prepared.iloc[:,-1:]], axis=1)

data_reduced_train.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/train/data_reduced_81.csv', index=False)
data_reduced_test.to_csv('C:/Users/angel/git/Observ_models/data/ML/Regression/test/data_reduced_81.csv', index=False)
