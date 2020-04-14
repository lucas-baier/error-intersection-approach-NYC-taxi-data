import pandas as pd
import numpy as np
from sklearn.neural_network import MLPRegressor
from xgboost.sklearn import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

from src.data import util
from joblib import load, dump

PATH = '/Users/lucasbaier/OneDrive - KIT/Lucas Baier/Code/TaxiData_Weather/'

df = util.read_all(show=False)
regions = util.largest(20)

name = 'static_weather_nh_2009-2011'
weather = True
holidays = False
regressor = load(PATH + 'models/model_{}.joblib'.format(name))

rs = 42
start_train = pd.Timestamp('2009-1-1', tz=df.index.tz)
end_train = pd.Timestamp('2013-1-1', tz=df.index.tz)
end_test = pd.Timestamp('2019-1-1', tz=df.index.tz)





# Create Training data for meta classifier
X_train_reg, y_train_reg, index, data = util.gen_data(regions, start_train, end_train, weather=weather, holidays=holidays)

scaler_reg = StandardScaler()
X_scaled_reg = scaler_reg.fit_transform(X_train_reg)

predictions_reg_test = regressor.predict(X_scaled_reg)
index['pred'] = predictions_reg_test
predictions_reg_test = index.set_index(['date', 'region']).unstack()
predictions_reg_test.columns = predictions_reg_test.columns.droplevel()
predictions_reg_test = predictions_reg_test.fillna(method ='backfill')

# Create label for classifier
pred_c = predictions_reg_test
pred_s = df.shift(1)[regions].loc[pred_c.index].astype(float).fillna(method = 'backfill')
real = df[regions].loc[pred_c.index].astype(float)

# Compute errors
error_c = (pred_c - real).abs()
error_s = (pred_s - real).abs()
label = ((error_c - error_s) < 0)

label_unstacked = label.unstack().rename('UseComplex').reset_index()
label_unstacked['error_c'] = error_c.unstack().values
label_unstacked['error_s'] = error_s.unstack().values

data_meta = pd.merge(data, label_unstacked, left_on=['date', 'region'], right_on=['date','region'])
y_train = data_meta['UseComplex']
X_train = data_meta.drop(['region', 'region_i', 'y', 'date', 'UseComplex', 'error_c', 'error_s'], axis=1)

scaler_meta = StandardScaler()
X_scaled_reg = scaler_meta.fit_transform(X_train)

#
#
#
classifier = XGBClassifier(n_jobs=8, n_estimators= 1000, verbosity= 1)
classifier.fit(X_scaled_reg, y_train)

dump(classifier, PATH + 'models/model_{}.joblib'.format('meta_classifier'))






# Create Test data for meta classifier
X_test_reg, y_test_reg, index, data_test = util.gen_data(regions, end_train, end_test, weather=weather, holidays=holidays)
X_test_reg = scaler_reg.transform(X_test_reg)

predictions_reg_test = regressor.predict(X_test_reg)
index['pred'] = predictions_reg_test
predictions_reg_test = index.set_index(['date', 'region']).unstack()
predictions_reg_test.columns = predictions_reg_test.columns.droplevel()
predictions_reg_test = predictions_reg_test.fillna(method ='backfill')

# Create label for data
pred_c = predictions_reg_test
pred_s = df.shift(1)[regions].loc[pred_c.index].astype(float).fillna(method = 'backfill')
real = df[regions].loc[pred_c.index].astype(float)

# Compute errors
error_c = (pred_c - real).abs()
error_s = (pred_s - real).abs()
label = ((error_c - error_s) < 0)

label_unstacked = label.unstack().rename('UseComplex').reset_index()
label_unstacked['error_c'] = error_c.unstack().values
label_unstacked['error_s'] = error_s.unstack().values

data_meta = pd.merge(data_test, label_unstacked, left_on=['date', 'region'], right_on=['date','region'])
y_test_meta = data_meta['UseComplex']
X_test_meta = data_meta.drop(['region', 'region_i', 'y', 'date', 'UseComplex', 'error_c', 'error_s'], axis=1)

X_test_meta = scaler_meta.transform(X_test_meta)

predictions_classifier = classifier.predict(X_test_meta)
confusion_matrix(y_test_meta, predictions_classifier)


df_predictions = pd.Series(predictions_classifier, index = [data_meta['date'], data_meta['region']])
df_predictions = df_predictions.unstack(level=-1)

pred_meta = pred_c.copy()
pred_meta[~df_predictions] = pred_s


# Save model and predictions
#dump(predictions_to_save, PATH + 'results/predictions_{}.joblib'.format(name))

print('RMSE_Meta:', util.rmse(df, pred_meta))
print('SMAPE_all:', util.smape(df, pred_meta))

print('RMSE complex:', util.rmse(df, pred_c))

