import pandas as pd
import numpy as np
from datetime import date
import holidays

def read_holidays():
    all_holidays = holidays.US(state='NY', years=set(range(2009,2020)))

    index_date = pd.date_range(start='1/1/2009', end='31/12/2019').to_pydatetime()
    index_date = np.array(list(map(lambda t: t.date(), index_date)))
    holiday_series = pd.Series(0, index = index_date)

    for date, name in all_holidays.items():
        holiday_series[holiday_series.index == date] = name

    df_holidays = pd.get_dummies(holiday_series)
    df_holidays.index = pd.to_datetime(df_holidays.index).tz_localize('US/Eastern')
    df_holidays = df_holidays.resample('h').pad()
    df = df_holidays[df_holidays.index < pd.Timestamp('2018-7-1', tz='US/Eastern')]
    return df


