import pandas as pd
import numpy as np

from src.data import util
from joblib import load, dump
import pickle

PATH = '/Users/lucasbaier/OneDrive - KIT/Lucas Baier/Code/TaxiData_Weather/'

df = util.read_all(show=False)
regions = util.largest(20)

pred_static_nw_nh = load(PATH + 'results/predictions_static_no_weather_nh_2009-2011.joblib')
pred_static_w_nh = load(PATH + 'results/predictions_static_weather_nh_2009-2011.joblib')
pred_static_nw_h = load(PATH + 'results/predictions_static_no_weather_h_2009-2011.joblib')
pred_static_w_h = load(PATH + 'results/predictions_static_weather_h_2009-2011.joblib')

pred_yearly_training_nw = load(PATH + 'results/predictions_yearly_training_no_weather_3y.joblib')
pred_yearly_training_nw_h = load(PATH + 'results/predictions_yearly_training_no_weather_h_3y.joblib')
pred_yearly_training_w = load(PATH + 'results/predictions_yearly_training_weather_3y.joblib')
pred_yearly_training_w_h = load(PATH + 'results/predictions_yearly_training_weather_h_3y.joblib')


all_preds = load(PATH + 'results/persistence_scikit_no-weather-yearly_update.joblib')

def compute_errors(df, pred):
    results = {}
    _pred = pred['predictions']
    results['Rmse'] = util.rmse(df, _pred)
    results['Smape'] = util.smape(df, _pred)
    results['Name'] = pred['name']
    return results

pred_list = [pred_static_nw_nh, pred_static_w_nh, pred_static_nw_h, pred_static_w_h,
             pred_yearly_training_nw ,pred_yearly_training_w, pred_yearly_training_nw_h, pred_yearly_training_w_h]
results = [compute_errors(df, i) for i in pred_list]

results_df = pd.DataFrame.from_dict(results).set_index('Name')
print(results_df)


def compute_yearly_errors(df, pred):
    _pred = pred['predictions']
    rmse = []
    smape = []
    year = []
    results = {}
    for i in range(2012, 2019):
        pred_year = _pred.loc[_pred.index.year == i]
        rmse.append(util.rmse(df, pred_year))
        smape.append(util.smape(df, pred_year))
        year.append(i)

    results['Name'] = pred['name']
    results['Rmse'] = rmse
    results['Smape'] = smape
    results['Year'] = year
    return(results)

results = [compute_yearly_errors(df, i) for i in pred_list]

#
# _pred = all_preds['pred_t']
# _df = df[_pred.columns].loc[_pred.index]
# _error = ((_df - _pred) ** 2)
#
# (_error.mean(axis = 1)**0.5).mean()
#
#
# def rmse(df, pred):
#     _pred = pred.clip(lower = 0)
#     _df = df[pred.columns].loc[pred.index]
#     return ((((_df - _pred) ** 2).mean(axis = 1)**0.5).mean())
#




def create_df_yearly(errors, metric):
    results_metric = [pd.Series(data = i[metric], index = i['Year'], name= i['Name']) for i in errors]
    df = pd.concat(results_metric, axis = 1)
    return df

results_df_yearly = create_df_yearly(results, 'Rmse')
print(results_df_yearly)


predictions = {}
for i in pred_list:
    predictions[i['name']] = i['predictions']

dump(predictions, PATH + 'results/all_predictions.joblib')


with open(PATH + 'results/all_predictions.pickle', 'wb') as handle:
    pickle.dump(predictions, handle, protocol=pickle.HIGHEST_PROTOCOL)