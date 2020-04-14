import pandas as pd
import numpy as np

from src.data import util
from src.models import dm_test
from joblib import load, dump
import pickle

PATH = '/Users/lucasbaier/OneDrive - KIT/Lucas Baier/Code/TaxiData_Weather/'

df = util.read_all(show=False)
regions = util.largest(20)

with open(PATH + 'results/all_predictions.pickle', 'rb') as handle:
    predictions = pickle.load(handle)

def switches(pred):
    eq = (pred_c == pred).mean(axis=1)
    return (eq != eq.shift(1)).sum()

results_setup = []

eia_dict = {}
for name, preds in predictions.items():

    print(name)

    # Naive or persistence model --> random walk
    pred_c = preds
    pred_s = df.shift(1)[regions].loc[pred_c.index].astype(float)
    real = df[regions].loc[pred_c.index].astype(float)

    # Compute errors
    error_c = (pred_c - real).abs()
    error_s = (pred_s - real).abs()

    # Optimize span based on validation data in 2012
    error_c_2012 = error_c[error_c.index.year == 2012]
    error_s_2012 = error_s[error_s.index.year == 2012]
    span_error = {}

    for i in range(1, 13):
        index_c_ewma = (error_c_2012 - error_s_2012).mean(axis=1)
        index_c_ewma = (index_c_ewma.ewm(span=i, adjust=True).mean() < 0).shift(1, fill_value=True)
        pred_eia_ewma = pred_c[pred_c.index.year == 2012].copy()
        pred_eia_ewma.loc[index_c_ewma == False] = pred_s.loc[pred_s.index.year == 2012].loc[index_c_ewma == False]

        # print('EIA EWMA:', util.rmse(real, pred_eia_ewma), 'Span: ', i)
        span_error[i] = util.rmse(real, pred_eia_ewma)

    opt_span = min(span_error.items(), key=lambda x: x[1])[0]

    index_c = ((error_c - error_s).mean(axis = 1) < 0)
    pred_eia_optimal = pred_c.copy()
    pred_eia_optimal.loc[index_c == False] = pred_s.loc[index_c == False]

    index_c_ewma = (error_c - error_s).mean(axis = 1)
    index_c_ewma = (index_c_ewma.ewm(span = opt_span, adjust=True).mean() < 0).shift(1, fill_value = True)
    pred_eia_ewma = pred_c.copy()
    pred_eia_ewma.loc[index_c_ewma == False] = pred_s.loc[index_c_ewma == False]

    # Save results
    perf_dict = {}
    perf_dict['Simple'] = util.rmse(real, pred_s)
    perf_dict['Complex'] = util.rmse(real, pred_c)
    perf_dict['EIA EWMA'] = util.rmse(real, pred_eia_ewma)
    perf_dict['Improvement'] = perf_dict['Complex'] - perf_dict['EIA EWMA']
    perf_dict['EIA optimal'] = util.rmse(real, pred_eia_optimal)
    perf_dict['Max improvement'] = perf_dict['Complex'] - perf_dict['EIA optimal']
    perf_dict['Perc'] = perf_dict['Improvement'] / perf_dict['Max improvement']

    perf_dict['Switches optimal'] = switches(pred_eia_optimal)
    perf_dict['Switches EIA'] = switches(pred_eia_ewma)
    perf_dict['Span chosen'] = opt_span


    # print(real.unstack().values.dtype)
    # print(pred_eia_ewma.unstack().values.dtype)
    # print(pred_c.unstack().values.dtype)
    # perf_dict['DM-test'] = dm_test.dm_test(real.unstack().values, pred_eia_ewma.unstack().values, pred_c.unstack().values)[1]

    result = pd.Series(perf_dict, name = name)

    print(result)
    results_setup.append(result)

    eia_dict[name] = pred_eia_ewma

df_results_all_setups = pd.concat(results_setup, axis = 1)

# pickle.dump(eia_dict, open(PATH + 'results/all_eia_predictions.pickle', 'wb'), protocol=pickle.HIGHEST_PROTOCOL)


# Weighted Ensemble

# span = 6 #ewma
# # EWMA of Errors for all regions
# def ew_error(pred):
#     return (pred - real).abs().ewm(span=span).mean().mean(axis=1).shift(1)
#
# ew_error_s = ew_error(pred_s)
# ew_error_c = ew_error(pred_c)
#
#
# # Simple Mean
# pred_mean_s = (pred_s + pred_c) / 2
#
# # Weighted Mean based on EWMA
# weights_ew = ew_error_s / (ew_error_c + ew_error_s)
# pred_mean_ew = (weights_ew * pred_c.T + (1-weights_ew) * pred_s.T).T.fillna(pred_c)


