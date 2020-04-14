import pandas as _pd
import matplotlib.pyplot as _plt
import numpy as _np
from keras.utils import to_categorical
from math import pi
from IPython.core.display import display as _display
from src.data import holiday_reader

_df = None
_shapes = None
_weather = None
figsize = (16,9)



def gen_data(regions, start, end='', lags=24, seasons=4, weather=False, holidays=False, return_df=False):
    global _df
    global _weather
    if weather is True:
        _weather = read_weather()

    if holidays is True:
        _holidays = holiday_reader.read_holidays()

    if end == '':
        end = start

    if type(start) == str:
        start = _pd.Timestamp(start, tz=_df.index.tz)

    data = _df[start - _pd.Timedelta(weeks=seasons):end][regions].unstack().reset_index()
    data.columns = ['region', 'date', 'y']

    # Add Lags
    for lag in range(1, lags + 1):
        data['val_%02d' % lag] = data['y'].shift(lag)
    for season in range(1, seasons + 1):
        data['week_%02d' % season] = data['y'].shift(season * 168)
    data = data[data['date'] >= _pd.Timestamp(start)]

    if weather:
        data = data.fillna(method='backfill').set_index('date').join(_weather).reset_index().rename({'index': 'date'}, axis=1)
        #data = data.dropna().set_index('date').join(_weather).reset_index().rename({'index': 'date'}, axis=1)
    else:
        #data = data.dropna().reset_index(drop=True)
        data = data.fillna(method='backfill').reset_index(drop=True)

    if holidays:
        data = data.set_index('date', drop=False)
        data = data.join(_holidays)

    # Add cyclical Information
    data['hour_sin'] = _np.sin(data['date'].dt.hour / 24 * 2 * pi)
    data['hour_cos'] = _np.cos(data['date'].dt.hour / 24 * 2 * pi)
    data['month_sin'] = _np.sin(data['date'].dt.month / 12 * 2 * pi)
    data['month_cos'] = _np.cos(data['date'].dt.month / 12 * 2 * pi)

    # Add Weekday Category
    weekdays = to_categorical(data['date'].dt.weekday, num_classes=7)
    data = _pd.concat([data, _pd.DataFrame(weekdays, index = data.index)], axis=1)

    # Add Region Category
    region_order = dict(zip(sorted(regions), range(len(regions))))
    data['region_i'] = data['region'].map(region_order)
    regions_hot = to_categorical(data['region_i'], num_classes=len(regions))
    data = _pd.concat([data, _pd.DataFrame(regions_hot, index = data.index)], axis=1)

    if not return_df:
        X = data.drop(['region', 'region_i', 'y', 'date'], axis=1).values
        y = data['y'].values
        index = data[['region', 'date']].copy()

    else:
        X = data.drop(['region', 'region_i', 'y', 'date'], axis=1)
        y = data['y']
        index = data[['region', 'date']].copy()
    return X, y, index, data


def read_weather():
    weather = _pd.read_csv(
        '/Users/lucasbaier/OneDrive - KIT/Lucas Baier/Code/TaxiData_Weather/data/central_park_weather.csv', header=0)
    weather = weather.iloc[:, 2:]
    weather.index = weather.DATE
    weather.loc[weather['AWND'].isna() == True, 'AWND'] = 0

    weather = weather.drop(['DATE', 'AWND'], axis=1)

    weather.index = _pd.to_datetime(weather.index).tz_localize('US/Eastern')
    weather = weather.resample('h').pad()

    weather = weather[weather.index < _pd.Timestamp('2018-7-1', tz='US/Eastern')]

    return weather



def smape(df, pred):
    _pred = pred.clip(lower = 0)
    _df = df[pred.columns].loc[pred.index]
    return ((_pred-_df).abs() / ((_pred+_df).abs()/2)).mean(axis =0).mean()

def rmse(df, pred):
    _pred = pred.clip(lower = 0)
    _df = df[pred.columns].loc[pred.index]
    return ((((_df - pred) ** 2).mean(axis = 1)**0.5).mean())


# def rmse(df, pred):
#     _pred = pred.clip(lower = 0)
#     _df = df[pred.columns].loc[pred.index]
#     return ((((_df - pred) ** 2).mean(axis = 0).mean())**0.5)

#
#
# def rmse(df, pred):
#     _pred = pred.clip(lower = 0)
#     _df = df[_pred.columns].loc[_pred.index]
#     return ((((_df - _pred) ** 2).mean(axis = 0).mean())**0.5)
#
#
# def symmetric_mean_absolute_percentage_error(df, pred):
#     _df = df[pred.columns].loc[pred.index]
#     smape = ((pred - df).abs() / (df + pred + 1)).mean(axis = 1).mean()
#     return smape




def read_single(path, show=True, tz=False):
    """Freq will be set to False if tz is True"""
    _df = _pd.read_csv(path, index_col=0, header=0)
    if tz:
        _df.index = _pd.to_datetime(_df.index, utc=True).tz_convert('US/Eastern')
        #_df = _df.tz_localize('UTC').tz_convert('US/Eastern')
    _df = _df.asfreq('h')
    _df.columns = _df.columns.astype(int)
    if show:
        display(_df.head())
        display(_df.shape)
    return _df


def read_all(show=False, include_na=False):
    """Reads the CSV containing all the data"""
    global _df
    
    if _df is None:
        # _df = read_single('data/all_timezone_corrected.csv', show=False, tz=True)
        _df = read_single('/Users/lucasbaier/OneDrive - KIT/Lucas Baier/Code/TaxiData_Weather/data/all_timezone_corrected.csv', show=False, tz=True)
    if include_na:
        res = _df.copy()
    else:
        res = _df[_df.columns[0:263]].copy()
    
    if show:
        display(res.head())
        display(res.shape)
        
    return res

def largest(n):
    global _df
    return sorted(list(_df.sum().nlargest(n).index))


def align_dfs(df1, df2):
    """If the indices don't match, make sure to only return rows, where the indices exist"""
    return df1[df1.index.isin(df2.index)], df2[df2.index.isin(df1.index)]


def show_region(region, width=16, height=9, graphs=False, query=None):
    """Displays a region on the map, and shows the data (evtl. based on query)"""
    import geopandas as _gpd
    global _shapes
    global _df
    
    _load_zones()
    
    name = '%s - %s' %(region, _shapes[_shapes['LocationID'] == region]['zone'].iloc[0])
    _shapes['highlight'] = _shapes['LocationID'] == region
    
    # highlight
    _display(name)
    _shapes.plot(column='highlight', figsize=(width,height))
    _plt.show()
    
    if graphs:
        read_all(show=False)
            
        if query:
            source = _df.query(query)
        else:
            source = _df
            
        #daily
        df = source[region].copy()
        df.index = df.index.floor('d')
        df.groupby(level=0).sum().plot(figsize=(width,height), title='Daily Sum for region %s' %name)
        _plt.show()

        # monthly
        df = source[region].copy()
        df.index = df.index.floor('d') - _pd.tseries.offsets.MonthBegin(1)
        df.groupby(level=0).sum().plot(figsize=(width,height), title='Monthly Sum for region %s' %name)
        _plt.show()
    
def get_name(region):
    global _shapes
    _load_zones()
    return _shapes[_shapes['LocationID'] == region]['zone'].iloc[0]    
    
def get_shapes():
    global _shapes
    _load_zones()
    return _shapes.copy()

_c2 = None
def baseline(region, begin, end=None, figsize=figsize):
    from sklearn.metrics import mean_squared_error
    global _df
    global _c2
    
    end = begin if end is None else end
    local = _df.copy()[[region]]
    if _c2 is None:
        _c2 = read_single('output/pred_canary2.csv', dtype=_np.float64, show=False)
    
    
    # main-lag
    local['baseline'] = local[region].shift(168)
    local_week = local[begin:end]
    _display('RMSE (week): %s' % mean_squared_error(local_week[region], local_week['baseline']) ** 0.5)
    
    # one year lag
    local['baseline'] = local[region].shift(168*52)
    local_year = local[begin:end]
    _display('RMSE (year): %s' % mean_squared_error(local_year[region], local_year['baseline']) ** 0.5)
    
    # canary2
    local['baseline'] = _c2[[region]]
    local_canary = local[begin:end]
    _display('RMSE (canary2): %s' % mean_squared_error(local_canary[region], local_canary['baseline']) ** 0.5)
    _display(local_canary.plot(figsize=figsize))
    
def _load_zones():
    import geopandas as _gpd
    global _shapes
    if _shapes is None:
        _shapes = _gpd.read_file('data/shapes_fixed/shapes_fixed.shp')
        _shapes.drop(['OBJECTID', "Shape_Area", "Shape_Leng"], axis=1, inplace=True)
        _shapes.crs = {'init': 'epsg:4326'}
        # _shapes = _shapes.to_crs({'init': 'epsg:4326'})
        
def get_model(path):
    from keras.models import load_model
    return load_model(path)


def model_prediction(path, regions, start, end=None, lags=24, seasons=4, weather=False):
    global _df
    end = end or start
    
    model = get_model(path)
    X_pred, _, index = gen_data(regions, start, end, lags=lags, seasons=seasons, weather=weather)
    
    index['pred'] = model.predict(X_pred).reshape(1,-1)[0]
    prediction = index.set_index(['date','region']).unstack()
    prediction.columns = prediction.columns.droplevel()
    
    return prediction.clip(lower=0).astype(int)
    

def _add_weekdays(data):
    from keras.utils import to_categorical
    weekdays = to_categorical(data.index.weekday, num_classes=7).astype(int)
    return data.join(_pd.DataFrame(weekdays, index=data.index, columns=['w_%d'%i for i in range(7)]))

def train_tlnn(regions, start, end, deep):
    from keras.models import Sequential
    from keras.layers import Dense, Dropout
    from keras.optimizers import Adam
    from IPython.core.display import HTML
    from keras_tqdm import TQDMNotebookCallback
    display(HTML("""
        <style>
            .p-Widget.jp-OutputPrompt.jp-OutputArea-prompt:empty {
                  padding: 0;
                  border: 0;
            }
        </style>
    """))
    
    X_test, y_test, _ = gen_data(regions, start, end)
    mod = [Dense(128, kernel_initializer='uniform', input_dim=len(X_test[0]), activation='relu'),
        Dropout(0.1)]
    if deep:
        mod.extend([Dense(32, kernel_initializer='uniform', activation='relu'),
        Dropout(0.1)])
    model = Sequential(mod + [Dense(1, kernel_initializer='uniform', activation='linear')])
    model.compile(optimizer=Adam(), loss='mse')
    batch_size = int(len(X_test)/64)
    model.fit(X_test, y_test, epochs=100, batch_size=batch_size, verbose=0, callbacks=[TQDMNotebookCallback()])
    return model

def mean_absolute_percentage_error(y_true, y_pred): 
        return _np.mean(_np.abs((y_true - y_pred) / (y_true + 1)))

def mean_log_acc(y_true, y_pred):
    return _np.log(y_pred.clip(lower=1) / y_true.clip(lower=1)).mean().abs()

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return ((y_pred - y_true).abs() / (y_true + y_pred + 1)).mean()
    
def mean_squared_error(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()
    
def mean_absolute_error(y_true, y_pred):
    return (y_true - y_pred).abs().mean()

def analyze(pred, start, end='', ret=True):
    global _df
    
    
    if end == '':
        end = start
    regions = pred.columns
    pred = pred[start:end].copy()
    real = _df[start:end][regions].copy()
    
    if len(regions) == 1:
        pred.join(real, rsuffix='real').plot(figsize=(16,9), title=regions[0])
    else:
        _plt.figure(figsize=(16,9))
        _plt.title('Sum of %s' % (','.join(str(r) for r in regions)))
        pred.sum(axis=1).plot()
        real.sum(axis=1).plot()
        _plt.figlegend(['pred', 'real'])
    _plt.show()
    
    #pred['Err on sum'] = pred.sum(axis=1)
    #real['Err on sum'] = real.sum(axis=1)
    
    errors = {}
    
    errors['RMSE'] = mean_squared_error(real, pred) ** 0.5
    errors['MAE'] = mean_absolute_error(real, pred)
    errors['MAPE'] = mean_absolute_percentage_error(real, pred)
    errors['MLAR'] = mean_log_acc(real, pred)
    errors['sMAPE'] = symmetric_mean_absolute_percentage_error(real, pred)
    res = _pd.DataFrame(errors).T
    res['mean'] = res.mean(axis=1)
    # res['std'] = res.std(axis=1)
    display(res['mean'])
    return res


# https://dev.to/nestedsoftware/calculating-standard-deviation-on-streaming-data-253l
class RunningStatsCalculator:
    def __init__(self):
        self.count = 0
        self._mean = 0
        self._dSquared = 0

    def update(self, newValue):
        self.count += 1

        meanDifferential = (newValue - self._mean) / self.count
        newMean = self._mean + meanDifferential
        dSquaredIncrement = (newValue - newMean) * (newValue - self._mean)

        self._mean = newMean
        self._dSquared += dSquaredIncrement

    @property
    def mean(self):
        self.validate()
        return self._mean

    @property
    def dSquared(self):
        self.validate()
        return self._dSquared

    @property
    def populationVariance(self):
        return self.dSquared / self.count

    @property
    def populationStdev(self):
        return self.populationVariance ** 0.5

    @property
    def sampleVariance(self):
        return self.dSquared / (self.count - 1) if self.count > 1 else 0

    @property
    def sampleStdev(self):
        return self.sampleVariance ** 0.5

    def validate(self):
        if (self.count == 0):
            raise Exception('Mean is undefined')


class DetectorWrapper():
    MAP_TORNADO = {'add_element': 'run', 'reset': 'reset'}
    MAP_SCIKIT = {'add_element': 'add_element', 'detected_drift': 'detected_change',
                  'detected_warning': 'detected_warning_zone'}

    def __init__(self, detector, mapping):
        self.detector = detector
        self.mapping = mapping
        self.drift = None
        self.warning = None

    def _run_func(self, name, *args):
        if name not in self.mapping:
            return
        return getattr(self.detector, self.mapping[name])(*args)

    def add_element(self, e):
        res = self._run_func('add_element', e)
        if res is not None:
            # print(res)
            self.warning, self.drift = res
        return res

    def detected_drift(self):
        if self.drift is not None:
            return self.drift
        return self._run_func('detected_drift')

    def detected_warning(self):
        if self.warning is not None:
            return self.warning

        return self._run_func('detected_warning')

    def reset(self):
        return self._run_func('reset')

class BaseModel():
    def __init__(self, history, regions):
        self.history = history[regions].copy()
        self.regions = regions

        self.now = max(self.history.index)
        self.delta_1h = _pd.Timedelta(hours=1)
        self.next = self.now + self.delta_1h

    def predict(self):
        return self.history.loc[self.now].rename(self.next)

    def update(self, observations):
        self.history = self.history.append(observations[self.regions])
        self.now, self.next = self.next, self.next + self.delta_1h

    def relearn(self):
        pass

class CanaryModel(BaseModel):
    def __init__(self, history, regions, num_weeks):
        super().__init__(history, regions)

        self.num_weeks = num_weeks
        self.deltas = [_pd.Timedelta(days=i * 7) for i in range(1, num_weeks + 1)]

    def __repr__(self):
        return 'Canary-%d, %s' % (self.num_weeks, str(self.now))

    def predict(self):
        base = self.history.loc[self.now]
        delta = sum(
            self.history.loc[self.next - d] - self.history.loc[self.now - d] for d in self.deltas) / self.num_weeks
        return (base + delta).clip(lower=0).rename(self.next)

    def update(self, observations):
        super().update(observations)
        # self.history = self.history[self.now - self.deltas[-1]:]


class FastModel(BaseModel):
    """A Model that pre-computes all predictions, and simply returns them when asked"""

    def __init__(self, predictions, regions, next_hour):
        super().__init__(predictions, regions)

        self.next = next_hour
        self.now = self.next - self.delta_1h

    def predict(self):
        return self.history.loc[self.next]

    def update(self, observation):
        self.now, self.next = self.next, self.next + self.delta_1h


class ScikitModel(FastModel):
    def __init__(self, path, regions, start, end,weather):
        from joblib import load
        models = load(path)
        preds = {}
        for region in regions:
            model, scaler = models[region]
            X_pred, _, index = gen_data([region], start, end, weather=weather)
            X_pred = scaler.transform(X_pred)
            index['pred'] = model.predict(X_pred).reshape(1,-1)[0]
            prediction = index.set_index(['date','region']).unstack()
            prediction.columns = prediction.columns.droplevel()
            preds[region] = prediction
        prediction = _pd.concat(preds, axis=1)
        prediction.columns = prediction.columns.droplevel()
        super().__init__(prediction.clip(lower=0), regions, start)



def six_month(ts):
    if ts.month == 1:
        return ts.replace(month=7)
    else:
        return ts.replace(year=ts.year + 1, month=1)


def quarters(ts):
    if ts.month == 10:
        return ts.replace(year=ts.year + 1, month=1)
    else:
        return ts.replace(month=ts.month + 3)