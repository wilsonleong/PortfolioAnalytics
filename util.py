# -*- coding: utf-8 -*-
"""
Created on Sun Dec  6 11:35:31 2020

@author: Wilson Leong

This module does the following:
    - GetStartDate()

"""

import datetime
import dateutil
import setup

date_ranges = ['YTD','1W','1M','3M','6M','1Y','3Y','5Y']

# function to return start dates for YTD 1W 1M 3M 6M 1Y 3Y 5Y 10Y; calculate up to end of yesterday
def GetStartDate(period=None):
    #period='1m'
    if period is None:
        tn = setup.GetAllTransactions()
        supported_instruments = setup.GetListOfSupportedInstruments()
        tn = tn[tn.BBGCode.isin(supported_instruments)]
        start_date = tn.Date.min()    # since inception
    else:
        period = period.upper()
    # supported periods: YTD 1W 1M 3M 6M 1Y 3Y 5Y 10Y; calculate up to end of yesterday (up to start of today)
        today = datetime.datetime.today().date()
        if period=='YTD':
            start_date = datetime.datetime(today.year, 1, 1)
        elif period=='1W':
            start_date = today + datetime.timedelta(days=-7)
        elif period=='1M':
            start_date = today + dateutil.relativedelta.relativedelta(months=-1)
        elif period=='3M':
            start_date = today + dateutil.relativedelta.relativedelta(months=-3)
        elif period=='6M':
            start_date = today + dateutil.relativedelta.relativedelta(months=-6)
        elif period=='1Y' or period=='12M':
            start_date = today + dateutil.relativedelta.relativedelta(years=-1)
        elif period=='3Y':
            start_date = today + dateutil.relativedelta.relativedelta(years=-3)
        elif period=='5Y':
            start_date = today + dateutil.relativedelta.relativedelta(years=-5)
        elif period=='10Y':
            start_date = today + dateutil.relativedelta.relativedelta(years=-10)
        start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
    return start_date


# function to return start dates for YTD 1W 1M 3M 6M 1Y 3Y 5Y 10Y; calculate up to end of yesterday
# function to get start and end dates for 20xx
def GetDates(period=None):
    '''
    Accepted values for period:
        None (start from first ever transaction, end yesterday)
        YTD
        1W
        1M
        3M
        6M
        1Y
        3Y
        5Y
        10Y
        20xx
    '''

    # since inception    
    if period is None:
        tn = setup.GetAllTransactions()
        supported_instruments = setup.GetListOfSupportedInstruments()
        tn = tn[tn.BBGCode.isin(supported_instruments)]
        start_date = tn.Date.min()
        end_date = datetime.datetime.now().date() - datetime.timedelta(days=1)

    # period is a year (2019, 2020, 2021)
    elif len(period)==4 and period.isnumeric():
        start_date = datetime.datetime(int(period), 1, 1)
        if int(period)==datetime.datetime.today().year:
            end_date = end_date = datetime.datetime.now().date() - datetime.timedelta(days=1)
        else:
            end_date = datetime.datetime(int(period), 12, 31)

    else:
        period = period.upper()
        # supported periods: YTD 1W 1M 3M 6M 1Y 3Y 5Y 10Y; calculate up to end of yesterday (up to start of today)
        today = datetime.datetime.today().date()
        if period=='YTD':
            start_date = datetime.datetime(today.year, 1, 1)
        elif period=='1W':
            start_date = today + datetime.timedelta(days=-7)
        elif period=='1M':
            start_date = today + dateutil.relativedelta.relativedelta(months=-1)
        elif period=='3M':
            start_date = today + dateutil.relativedelta.relativedelta(months=-3)
        elif period=='6M':
            start_date = today + dateutil.relativedelta.relativedelta(months=-6)
        elif period=='1Y' or period=='12M':
            start_date = today + dateutil.relativedelta.relativedelta(years=-1)
        elif period=='3Y':
            start_date = today + dateutil.relativedelta.relativedelta(years=-3)
        elif period=='5Y':
            start_date = today + dateutil.relativedelta.relativedelta(years=-5)
        elif period=='10Y':
            start_date = today + dateutil.relativedelta.relativedelta(years=-10)
        
        end_date = datetime.datetime.now().date() - datetime.timedelta(days=1)
        
    start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
    end_date = datetime.datetime.combine(end_date, datetime.datetime.min.time())
    return start_date, end_date


