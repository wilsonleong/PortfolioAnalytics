# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:10:57 2020

@author: Wilson Leong

This module does the following:
    - compute TWRR (time-weighted rate of return), aka Modified Dietz Return
    - compute MWRR (money-weighted raet of return), aka IRR (internal rate of return)
    

"""

import setup
import pandas as pd
import numpy as np
import datetime
import dateutil.relativedelta
import calc_summary
import calc_fx

# use Yahoo Finance API to collect historical prices
import yfinance as yf
yf.pdr_override()
from pandas_datareader import data as pdr
#data = pdr.get_data_yahoo('VWO', start='2020-02-24', end=datetime.datetime.today())
#data = pdr.get_data_yahoo('HKD=X', start='2020-02-24', end=datetime.datetime.today())


# function to return start dates for YTD 1W 1M 3M 6M 1Y 3Y 5Y 10Y; calculate up to end of yesterday
def GetStartDate(period):
    #period='1m'
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
    return start_date


# function to return valuation and cash flows as of particular date
def _GetValuation(start_date):
    #start_date=datetime.datetime(2020,3,3)
    #hist_valuation = CalcPortfolioHistoricalValuation()
    row = hist_valuation[hist_valuation.Date<=start_date]
    if len(row)==0:
        val = 0
    else:
        val = row.ValuationHKD.iloc[-1]
    return val


# get the cashflows from a certain date
def _GetCashFlows(start_date):
    #start_date=datetime.datetime(2020,9,1)
    tn = GetTransactionsETFs()
    tn.drop(['_id'], axis=1, inplace=True)
    cashflows = tn[tn.Date>=start_date]
    return cashflows


# determine which date ranges to collect historical data for
def _GetETFDataDateRanges(bbgcode):
    #bbgcode='SPY US'
    tn = setup.GetTransactionsETFs()
    tn = tn[tn.BBGCode==bbgcode]
    
    # check if security is still in the portfolio, or position is already closed
    DateFrom = tn.Date.min().date()
    if tn.NoOfUnits.sum()==0:
        DateTo = tn.Date.max() + datetime.timedelta(days=1)
        DateTo = DateTo.date()
    else:
        DateTo = datetime.datetime.today().date()
    dates = {}
    dates['DateFrom'] = DateFrom
    dates['DateTo'] = DateTo
    return dates
#_GetETFDataDateRanges('SPY US')


# get list of ETFs and date ranges, and query
def GetHistoricalData(start_date=datetime.datetime(2019,12,31)):
    #list_of_etfs = GetListOfETFs()
    list_of_supported_instruments = setup.GetListOfSupportedInstruments()
    # populate list of ETFs and date ranges
    df = pd.DataFrame(columns=['BBGCode','YFTicker','DateFrom','DateTo'])
    for i in range(len(list_of_supported_instruments)):
        bbgcode = list_of_supported_instruments[i]
        yf_ticker = setup.GetYahooFinanceTicker(bbgcode)
        dates = _GetETFDataDateRanges(bbgcode)
        date_from = dates['DateFrom']
        date_to = dates['DateTo']
        if date_from < start_date.date():
            date_from = start_date.date()
        df = df.append({'BBGCode':bbgcode,'YFTicker': yf_ticker,'DateFrom': date_from,'DateTo': date_to}, ignore_index=True)

    # loop through the list and collect the data from Yahoo
    data = pd.DataFrame()
    for i in range(len(df)):
        row = df.iloc[i]
        tmp = pdr.get_data_yahoo(row.YFTicker, start=row.DateFrom, end=row.DateTo)
        tmp = tmp.reset_index()
        tmp['BBGCode'] = row.BBGCode
        data = data.append(tmp, ignore_index=False)
    
    # NEED TO DEAL WITH HK/US HOLIDAYS MISMATCH
    tmp = data.pivot('Date','BBGCode', values='Close')
    tmp = tmp.fillna(method='ffill')
    tmp = tmp.reset_index()
    tmp2 = pd.melt(tmp, id_vars=['Date'], value_vars=list(data.BBGCode.unique()), value_name='Close')
    tmp2.dropna(inplace=True)
    tmp2.to_csv('HistoricalPrices.csv', index=False)
    return tmp2


historical_data = GetHistoricalData()
# get USDHKD rate
usdhkd = pdr.get_data_yahoo('HKD=X', start='2019-12-31', end=datetime.datetime.today())
usdhkd = usdhkd[['Close']]
usdhkd.columns = ['USDHKDrate']
# calculate the value of a security (returns time series) - this works for US ETFs only
def _CalcValuation(bbgcode):
    #bbgcode='SPY US'
    #bbgcode='SCHSEAI SP'
    hd = historical_data[historical_data.BBGCode==bbgcode]
    hd = hd[['Date','Close']]
    hd = hd.sort_values(['Date'], ascending=True)
    tn = setup.GetTransactionsETFs()
    tn = tn[tn.BBGCode==bbgcode]
    tn = tn[['Date','NoOfUnits']]
    df = pd.merge(hd, tn, how='left', on='Date')
    df.NoOfUnits.fillna(0, inplace=True)
    df['Holdings'] = df.NoOfUnits.cumsum()
    # security currency
    sec_ccy = setup.GetSecurityCurrency(bbgcode)
    ToUSD = calc_fx.GetFXRate('USD', sec_ccy)
    df['Valuation'] = df.Holdings * df.Close
    df['ValuationUSD'] = df.Valuation * ToUSD
    df = df.merge(usdhkd, how='left', on='Date')
    df['USDHKDrate'] = df.USDHKDrate.fillna(method='ffill')
    df['ValuationHKD'] = df.ValuationUSD * df.USDHKDrate
    return df


# calculate the value of the entire portfolio (add up each security in the portfolio)
def CalcPortfolioHistoricalValuation():
    #list_of_etfs = GetListOfETFs()
    list_of_supported_instruments = setup.GetListOfSupportedInstruments()
    df = pd.DataFrame()
    
    # loop through the list
    for i in range(len(list_of_supported_instruments)):
        bbgcode = list_of_supported_instruments[i]
        tmp = _CalcValuation(bbgcode)
        # remove redundant rows
        tmp = tmp[~((tmp.NoOfUnits==0) & (tmp.Holdings==0))]
        tmp['BBGCode'] = bbgcode
        df = df.append(tmp, ignore_index=False)

    # on each unique date, take the last row of unique security to avoid duplicated valuation
    df = df.drop_duplicates(subset=['Date','BBGCode'], keep='last')

    # group the data by date
    agg = df.groupby(['Date']).agg({'ValuationHKD':'sum'})
    agg = agg.reset_index()
    return agg
# return the valuation 
hist_valuation = CalcPortfolioHistoricalValuation()


# compute the Modified Dietz return
def CalcModDietzReturn(platform, bbgcode=None, period=None):
    #period='3M'
    #period='1W'
    #platform='FSM HK'
    #bbgcode=None
    #bbgcode='VGT US'
    #bbgcode='XLE US' # bad example, need to adjust for transferring from SG to HK
    #bbgcode='ALLGAME LX'
    
    # collect the data
    #df = _GetDataset()
    df = setup.GetAllTransactions()

    # filter on date range for the transactions / cash flows
    if period is not None:
        start_date = GetStartDate(period)
        df = df[df.Date >= np.datetime64(start_date)]
    
    # filter the data based on selection criteria (bbgcode, or platform)
    df = df[df.Platform==platform]
    df.drop(['_id'], axis=1, inplace=True)
    
    if bbgcode is not None:
        df = df[df.BBGCode==bbgcode]

    # Buy and Sell
    cf = df[df.Type.isin(['Buy','Sell'])].copy()
    div = df[df.Type=='Dividend']
    
    # calc dates
    date_start = cf.Date.min()
    date_end = np.datetime64(datetime.datetime.today().date())
    date_diff = (date_end - date_start).days

    # calculate No of days for weighting    
    cf.loc[:,'NoOfDays'] = (datetime.datetime.today() - cf.loc[:,'Date']) / pd.Timedelta(days=1)
    cf.loc[:,'NoOfDays'] = cf.loc[:,'NoOfDays'].astype(int)
    cf.loc[:,'WeightedCF'] = cf.loc[:,'CostInPlatformCcy'] * cf.loc[:,'NoOfDays'] / date_diff
    
    # pnl = current value + realised gains/losses
    
    # realised gains/losses (sold securities + dividends)
    pnl_realised = df.RealisedPnL.sum()
    
    #ps = calc_summary.GetPortfolioSummary()
    ps = calc_summary.ps
    
    # unrealised pnl
    if bbgcode is not None:
        if len(ps_active.BBGCode==bbgcode) > 0:
            r = ps_active[ps_active.BBGCode==bbgcode].iloc[0]
            # current value needs to include realised gains & dividends
            current_value = r.CurrentValue + pnl_realised
    else:
        ps_active = ps[ps.CurrentValue!=0]
        r = ps_active[ps_active.Platform==platform]
        # current value needs to include realised gains & dividends
        current_value = pnl_realised + r.CurrentValue.sum()
    
    # withdrawals
    withdrawals = cf[cf.Type=='Sell']
    
    # deposits
    deposits = cf[cf.Type=='Buy']
    
    # numerator: V(1) - V(0) - sum of cash flows
    if period is None:
        beginning_value = 0
    else:
        beginning_value = _GetValuation(np.datetime64(start_date))
    
    net_external_cash_flows = deposits.CostInPlatformCcy.sum() + withdrawals.CostInPlatformCcy.sum()
    num = current_value - beginning_value - net_external_cash_flows
    
    # denominator: V(0) + sum of cash flows weighted
    den = beginning_value + deposits.WeightedCF.sum() + withdrawals.WeightedCF.sum()

    # Modified Dietz Return (cumulative)
    mdr = num/den
    
    # calculate annualised return
    annualised_return = (1 + mdr) ** (365/date_diff) - 1
    
    # object to return
    obj = {}
    obj['DateStart'] = date_start
    obj['DateEnd'] = date_end
    obj['CumulativeReturn'] = mdr
    obj['AnnualisedReturn'] = annualised_return
    return obj
#CalcModDietzReturn('FSM HK', period='3M')
#CalcModDietzReturn('FSM HK', period='1M')  # BUG!!!
#CalcModDietzReturn('FSM HK', period='1W')  # BUG!!!


# calculate the cost of the entire portfolio (add up each transaction in the portfolio)
def CalcPortfolioHistoricalCost(platform='FSM HK', start_date=datetime.datetime(2019,12,31)):
    tn_etf_cost = setup.GetTransactionsETFs()
    tn_etf_cost = tn_etf_cost[tn_etf_cost.Date > start_date]
    #tn_etf_cost['AccumCost'] = tn_etf_cost.CostInPlatformCcy.cumsum()
    agg = tn_etf_cost.groupby(['Date']).agg({'CostInPlatformCcy':'sum'})
    agg = agg.reset_index()
    agg['AccumCostHKD'] = agg.CostInPlatformCcy.cumsum()
    agg.drop(['CostInPlatformCcy'], axis=1, inplace=True)
    return agg


import scipy.optimize
def _xnpv(rate, values, dates):
    '''Equivalent of Excel's XNPV function.

    >>> from datetime import date
    >>> dates = [date(2010, 12, 29), date(2012, 1, 25), date(2012, 3, 8)]
    >>> values = [-10000, 20, 10100]
    >>> xnpv(0.1, values, dates)
    -966.4345...
    '''
    if rate <= -1.0:
        return float('inf')
    d0 = dates[0]    # or min(dates)
    return sum([ vi / (1.0 + rate)**((di - d0).days / 365.0) for vi, di in zip(values, dates)])

def _xirr(values, dates):
    '''Equivalent of Excel's XIRR function.

    >>> from datetime import date
    >>> dates = [datetime.datetime(2010, 12, 29), datetime.datetime(2012, 1, 25), datetime.datetime(2012, 3, 8)]
    >>> values = [-10000, 20, 10100]
    >>> xirr(values, dates)
    0.0100612...
    '''
    try:
        return scipy.optimize.newton(lambda r: _xnpv(r, values, dates), 0.0)
    except RuntimeError:    # Failed to converge?
        return scipy.optimize.brentq(lambda r: _xnpv(r, values, dates), -1.0, 1e10)
# dates = [datetime.datetime(2020, 2, 24),
#          datetime.datetime(2020, 4, 21),
#          datetime.datetime(2020, 9, 22)]
# values = [-119153.86,
#           -120911.58,
#           305547.88]


# calculate the IRR
# WORK IN PROGRESS: NEED TO TAKE INTO ACCOUNT BALANCE BROUGHT FORWARD!!!
def CalcIRR(platform=None, bbgcode=None, period=None):
    #platform = 'FSM HK'
    #bbgcode = 'SPY US'
    #period = None
    #platform,bbgcode,period=None,None,None
    df = setup.GetAllTransactions()
    list_of_supported_securities = setup.GetListOfSupportedInstruments()

    # filter the data based on selection criteria (bbgcode, platform)
    df.drop(['_id'], axis=1, inplace=True)
    df = df[df.BBGCode.isin(list_of_supported_securities)]
    if platform is not None:
        df = df[df.Platform==platform]
    if bbgcode is not None:
        df = df[df.BBGCode==bbgcode]






    # start date as 1 Jan 2020 (HARDCODED)
    start_date = datetime.datetime(2020,1,1)
    df = df[df.Date>start_date]
    # need to add balance brought forward
    # add cost of existing holding
    # add valuation of existing holdings even if no trading in the year












    # filter on date range for the transactions / cashflows
    if period is not None:
        start_date = GetStartDate(period)
        # check if the date range covers any data
        hasPrevData = False
        if len(df[df.Date <= np.datetime64(start_date)]) > 1:
            hasPrevData = True
    
    perform_calc = True
    if period is not None:
        if hasPrevData:
            df = df[df.Date >= np.datetime64(start_date)].copy()
        else:
            irr = np.nan
            perform_calc = False

    if perform_calc:
        # calculation cashflow direction for XIRR computation
        df.loc[df.Type=='Buy', 'Cashflow'] = df.loc[df.Type=='Buy', 'CostInPlatformCcy'] * -1
        df.loc[df.Type=='Sell', 'Cashflow'] = df.loc[df.Type=='Sell', 'CostInPlatformCcy'] * -1
        df.loc[df.Type=='Dividend', 'Cashflow'] = df.loc[df.Type=='Dividend', 'RealisedPnL']
        
        # get platform and currency
        platforms = list(df.Platform.unique())
        currencies = [setup.GetPlatformCurrency(x) for x in platforms]
        platform_ccy = {platforms[x]: currencies[x] for x in range(len(platforms))}
        df['PlatformCcy'] = df.Platform.map(platform_ccy)
        df = df[['Date','PlatformCcy','Cashflow']]
        # calculate HKD equivalent
        SGDHKD = calc_fx.GetFXRate('HKD','SGD')
        ToHKD = {'HKD':1, 'SGD':SGDHKD}
        df['CashflowInHKD'] = df.PlatformCcy.map(ToHKD) * df.Cashflow
    
        # get valuations (beginning, ending)
        if bbgcode is None:
            val = CalcPortfolioHistoricalValuation()
            val.rename(columns={'ValuationHKD':'Cashflow'}, inplace=True)
        else:
            val = _CalcValuation(bbgcode) # bug: if bbgcode is None....??????
            val.rename(columns={'ValuationHKD':'Cashflow'}, inplace=True)
            val = val[['Date','Cashflow']]
        # latest valuation up to start date
        if period is not None:
            val_start = (val[(val.Date<=np.datetime64(start_date)) & (val.index==val[val.Date<=np.datetime64(start_date)].index.max())]).copy()
            val_start.loc[:,'Cashflow'] = val_start.loc[:,'Cashflow'] * -1
            # add beginning valuation as first cashflow (only if there are transactions in the period)
            #if len(df) > 0:
            if hasPrevData:
                df = df.append(val_start)
        
        # latest valuation
        val_end = val[val.index==val.index.max()]
            
        # add latest valuation as final cashflow (only if there are still holdings)
        #if (df.Date.iloc[-1] != val.Date.iloc[0]) and val.Cashflow.iloc[0]!=0:
        if val.Cashflow.iloc[0]!=0:
            df = df.append(val_end)
    
        df = df.sort_values(['Date'])
        df = df.reset_index(drop=True)
    
        # annualised return
        annualised_irr = _xirr(values=df.Cashflow.to_list(), dates=df.Date.to_list())
        
        # convert back to period
        no_of_days = (pd.to_datetime(df.iloc[-1].Date) - pd.to_datetime(df.iloc[0].Date)).days
        irr = (1+annualised_irr)**(no_of_days/365)-1
    
    return irr
#CalcIRR('FSM HK')
#CalcIRR('FSM HK', period='1w')
#CalcIRR('FSM HK', period='1m')
#CalcIRR('FSM HK', period='3m')
#CalcIRR('FSM HK', period='6m')
#CalcIRR('FSM HK', period='1y')
#CalcIRR('FSM HK', period='ytd')
