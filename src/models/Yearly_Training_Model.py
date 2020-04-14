import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.data import util
from src.data.util import ScikitModel

import tqdm
from joblib import load, dump

# Variable declarations
PATH = '/Users/lucasbaier/OneDrive - KIT/Lucas Baier/Code/TaxiData_Weather/'
name = 'yearly_training_no_weather_h_3y'
weather = False
holidays = True

df = util.read_all(show=False)
regions = util.largest(20)
rs = 42

pred_t = []
for year in tqdm.tqdm(range(2009, 2016)):
    start_train = pd.Timestamp('%d-1-1' %(year), tz=df.index.tz)
    end_train = pd.Timestamp('%d-1-1' %(year+3), tz=df.index.tz)
    end_test = pd.Timestamp('%d-1-1' %(year+4), tz=df.index.tz)

    X_train, y_train, index, data = util.gen_data(regions, start_train, end_train, weather=weather, holidays=holidays)

    print('Shape of training data: ', X_train.shape)

    regressor = MLPRegressor(hidden_layer_sizes=(128, 32), tol=4, max_iter=300, verbose=True, random_state=rs)
    regressor.out_activation_ = 'relu'

    scaler = StandardScaler()
    X_scale = scaler.fit_transform(X_train)
    regressor.fit(X_scale, y_train)

    X_test, y_test, index, data = util.gen_data(regions, end_train, end_test, weather= weather, holidays=holidays)
    X_test = scaler.transform(X_test)
    predictions = regressor.predict(X_test)
    index['pred'] = predictions
    predictions = index.set_index(['date', 'region']).unstack()
    predictions.columns = predictions.columns.droplevel()
    pred_t.append(predictions)

    dump(regressor, PATH + 'models/{}_{}-{}.joblib'.format(name, start_train.year, end_train.year))


pred_yt = pd.concat(pred_t, axis = 0)
predictions_to_save = {}
predictions_to_save['name'] = name
predictions_to_save['predictions'] = pred_yt
dump(predictions_to_save, PATH + 'results/predictions_{}.joblib'.format(name))

print('RMSE_all:', util.rmse(df, pred_yt))
print('SMAPE_all:', util.smape(df, pred_yt))

