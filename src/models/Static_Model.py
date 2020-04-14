import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from src.data import util
from joblib import load, dump

PATH = '/Users/lucasbaier/OneDrive - KIT/Lucas Baier/Code/TaxiData_Weather/'

df = util.read_all(show=False)
regions = util.largest(20)

name = 'static_weather_nh_2009-2011'
weather = True
holidays = False
rs = 42
start_train = pd.Timestamp('2009-1-1', tz=df.index.tz)
end_train = pd.Timestamp('2012-1-1', tz=df.index.tz)
end_test = pd.Timestamp('2019-1-1', tz=df.index.tz)

X_train, y_train, index, data = util.gen_data(regions, start_train, end_train, weather=weather, holidays=holidays)

regressor = MLPRegressor(hidden_layer_sizes=(128,32), tol=4, max_iter=300, verbose = True, random_state=rs)
regressor.out_activation_ = 'relu'

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
regressor.fit(X_scaled, y_train)

X_test, y_test, index, data = util.gen_data(regions, end_train, end_test, weather = weather, holidays=holidays)
X_test = scaler.transform(X_test)
predictions = regressor.predict(X_test)
index['pred'] = predictions
predictions = index.set_index(['date','region']).unstack()
predictions.columns = predictions.columns.droplevel()

predictions_to_save = {}
predictions_to_save['name'] = name
predictions_to_save['predictions'] = predictions

# Save model and predictions
dump(regressor, PATH + 'models/model_{}.joblib'.format(name))
dump(predictions_to_save, PATH + 'results/predictions_{}.joblib'.format(name))

print('RMSE_all:', util.rmse(df, predictions))
print('SMAPE_all:', util.smape(df, predictions))

