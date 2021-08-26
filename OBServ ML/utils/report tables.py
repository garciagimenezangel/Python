import pandas as pd

floralnest = pd.read_csv('C:/Users/angel/git/Observ_models/report/data - version paper/Lookup tables/FloralNest-CGLSCont_DelphiResults_Simplified.csv').drop(columns=['management nesting'])
print(floralnest.to_latex(index=False, float_format='%.2f'))

floralnest_ESTIMAP = pd.read_csv('C:/Users/angel/git/Observ_models/report/data - version paper/Lookup tables/FloralNest-CGLSDisc_ESTIMAP_Simplified.csv').drop(columns=['management nesting'])
floralnest_ESTIMAP['Landcover'] = floralnest_ESTIMAP['descr']
cols = floralnest_ESTIMAP.columns.tolist()
cols = cols[-1:] + cols[:-1]
floralnest_ESTIMAP = floralnest_ESTIMAP[cols]
floralnest_ESTIMAP.drop(columns=['landcover', 'descr'], inplace=True)
print(floralnest_ESTIMAP.to_latex(index=False, float_format='%.2f'))

guilds = pd.read_csv('C:/Users/angel/git/Observ_models/report/data - version paper/Lookup tables/Guilds_SpainAssessment_Simplified.csv')
print(guilds.to_latex(index=False, float_format='%.2f'))

predictors_meta =  pd.read_csv('C:/Users/angel/git/Observ_models/report/tables/metadata_variables.csv')
predictors_meta =  predictors_meta.astype(str)
predictors_meta =  predictors_meta.drop(columns=['Variable'])
predictors_meta.columns = ['Group', 'Variable', 'Dataset','Reference']
predictors_meta =  predictors_meta.replace({'nan': ''}, regex=True)
with pd.option_context("max_colwidth", 2000):
    print (predictors_meta.to_latex(index=False))
