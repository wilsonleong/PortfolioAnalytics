# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 15:10:57 2020

@author: Wilson Leong

This module does the following:
    - compute TWRR (time-weighted rate of return), aka Modified Dietz Return
    - compute MWRR (money-weighted rate of return), aka IRR (internal rate of return)
    

"""

import datetime
import setup
import pandas as pd
import numpy as np
import dateutil.relativedelta
import calc_summary
import calc_fx
import mdata
import util


# function to get the balance brought forward (of supported instruments)
# returns list of holdings, cost and valuation in HKD at the given start date
def _GetExistingHoldings(start_date, bbgcode=None, platform=None, base_ccy='HKD'):
    tn = setup.GetAllTransactions()
    tn = tn[tn.Date<start_date]
    if platform is not None:
        tn = tn[tn.Platform==platform]
    # only include the supported instruments
    support_instruments = setup.GetListOfSupportedInstruments()
    tn = tn[tn.BBGCode.isin(support_instruments)]
    holdings = tn.groupby(['Platform','BBGCode']).agg({'NoOfUnits':'sum','CostInPlatformCcy':'sum'})
    holdings = holdings[holdings.NoOfUnits!=0]
    holdings = holdings.reset_index()
    # calculate the cost and valuation in base ccy equivalent (cost in platform ccy, val in sec ccy)
    historical_data = mdata.GetHistoricalData()
    val = historical_data.copy()
    val = val[val.Date<start_date]

    for i in range(len(holdings)):
        row = holdings.iloc[i]
        holdings.loc[i,'PlatformCcy'] = setup.GetPlatformCurrency(row.Platform)
        holdings.loc[i,'SecurityCcy'] = setup.GetSecurityCurrency(row.BBGCode)
        
        # add valuation here
        v = val[val.BBGCode==row.BBGCode]
        if len(v)==0:
            print ('WARNING: no market data - check feed/date range')
        holdings.loc[i,'Close'] = v.iloc[-1].Close
        holdings.loc[i,'ValuationInSecCcy'] = holdings.loc[i,'Close'] * row.NoOfUnits
        
        # calc base ccy equivalent
        # optimise FX query (if platform ccy = security ccy then use same fx rate)
        same_ccy = holdings.loc[i,'PlatformCcy']==holdings.loc[i,'SecurityCcy']
        if same_ccy:
            fxrate = calc_fx.ConvertFX(holdings.loc[i,'SecurityCcy'],base_ccy)
            holdings.loc[i,'CostInBaseCcy'] = fxrate * row.CostInPlatformCcy
            holdings.loc[i,'ValuationInBaseCcy'] = fxrate * holdings.loc[i,'ValuationInSecCcy']
        else:
            holdings.loc[i,'CostInBaseCcy'] = calc_fx.ConvertTo(base_ccy, holdings.loc[i,'PlatformCcy'], row.CostInPlatformCcy)
            holdings.loc[i,'ValuationInBaseCcy'] = calc_fx.ConvertTo(base_ccy, holdings.loc[i,'SecurityCcy'], holdings.loc[i,'ValuationInSecCcy'])

    return holdings


# function to return valuation and cash flows as of particular date
def _GetValuation(start_date):
    row = hist_valuation[hist_valuation.Date<=start_date]
    if len(row)==0:
        val = 0
    else:
        val = row.ValuationHKD.iloc[-1]
    return val


# calculate the value of a security (returns time series) - this works only when bbgcode is specified
def _CalcValuation(bbgcode, platform=None, start_date=None):
    # assumes bbgcode can only be on 1 platform (exception VWO XLE)
    #bbgcode='XLE US'
    #platform='FSM HK'
    #bbgcode='SCHSEAI SP'
    
    tn = setup.GetAllTransactions()
    # filter by platform and bbgcode
    if platform is not None:
        tn = tn[tn.Platform==platform]
    
    tn = tn[tn.BBGCode==bbgcode]
    
    if start_date is None:
        #tn = setup.GetAllTransactions()
        supported_instruments = setup.GetListOfSupportedInstruments()
        tn = tn[tn.BBGCode.isin(supported_instruments)]
        
        #if bbgcode is not None:
        tn = tn[tn.BBGCode==bbgcode]

        start_date = tn.Date.min()
    
    hd = mdata.GetHistoricalData(bbgcode=bbgcode)
    hd = hd[['Date','Close']]
    hd_prev = hd[hd.Date<start_date].copy()
    hd_prev = hd_prev.tail(1)
    
    # filter by selected date range
    hd = hd[hd.Date>=start_date]
    # filter by date until its no longer held
    if tn.NoOfUnits.sum()==0:
        hd = hd[hd.Date<=tn.Date.max()]
    # add back last valuation before beginning of date range
    hd = hd.append(hd_prev)
    
    hd = hd.sort_values(['Date'], ascending=True)
    
    tn = tn[['Date','NoOfUnits']]
    tn = tn[tn.Date>=start_date]
    
    # CAREFUL: if the transaction date is a holiday where there is no market data, the holdings will be missed
    
    # add balance brought forward
    bf = _GetExistingHoldings(start_date, platform=platform)
    bf = bf[bf.BBGCode==bbgcode]
    df = pd.merge(hd, tn, how='left', on='Date')
    # if there is balance b/f, then add it
    if len(bf)>0:
        df.loc[0,'NoOfUnits'] = bf.iloc[0].NoOfUnits
    df.NoOfUnits.fillna(0, inplace=True)
    df['Holdings'] = df.NoOfUnits.cumsum()
    # security currency
    sec_ccy = setup.GetSecurityCurrency(bbgcode)
    ToUSD = calc_fx.GetFXRate('USD', sec_ccy)
    df['Valuation'] = df.Holdings * df.Close
    df['ValuationUSD'] = df.Valuation * ToUSD
    # filter out unused rows
    
    # load historical USDHKD exchange rates from cache
    usdhkd = mdata.GetHistoricalUSDHKD()
    
    df = df.merge(usdhkd, how='left', on='Date')
    df['USDHKDrate'] = df.USDHKDrate.fillna(method='ffill')
    df['ValuationHKD'] = df.ValuationUSD * df.USDHKDrate
    return df
#_CalcValuation(bbgcode='XLE US', platform='FSM HK')
#_CalcValuation(bbgcode='XLE US', platform='FSM SG')
#_CalcValuation(bbgcode='XLE US')


# calculate the value of the entire portfolio (add up each security in the portfolio)
def CalcPortfolioHistoricalValuation(platform=None, bbgcode=None, start_date=None):
    # only applies to instruments supported by Yahoo Finance
    supported_instruments = setup.GetListOfSupportedInstruments()
    tn = setup.GetAllTransactions()
    tn_in_scope = tn[tn.BBGCode.isin(supported_instruments)]
    instruments_in_scope = supported_instruments
    
    # if platform is specified, check which instruments were actually on the platform
    if platform is not None:
        tn_in_scope = tn_in_scope[tn_in_scope.Platform==platform]
        instruments_in_scope = list(tn_in_scope.BBGCode.unique())
    
    # if bbgcode is specified, then restrict to just the instrument
    if bbgcode is not None:
        if bbgcode in instruments_in_scope:
            instruments_in_scope = [bbgcode]
    
    # if start date is not defined, start from earliest transaction in scope
    if start_date is None:
        start_date = tn_in_scope.Date.min()

    df = pd.DataFrame()
    # loop through the list
    for i in range(len(instruments_in_scope)):
        bbgcode = instruments_in_scope[i]
        tmp = _CalcValuation(bbgcode=bbgcode, platform=platform, start_date=start_date)
        # remove redundant rows
        tmp = tmp[~((tmp.NoOfUnits==0) & (tmp.Holdings==0))]
        tmp['BBGCode'] = bbgcode
        df = df.append(tmp, ignore_index=False)

    # on each unique date, take the last row of unique security to avoid duplicated valuation
    df.sort_values(['Date','BBGCode'], inplace=True)
    df = df.drop_duplicates(subset=['Date','BBGCode'], keep='last')

    # group the data by date
    agg = df.groupby(['Date']).agg({'ValuationHKD':'sum'})
    agg = agg.reset_index()
    return agg


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
        start_date = util.GetStartDate(period)
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
    
    ps = calc_summary.GetPortfolioSummary()
    ps = ps['Original']
    #ps = calc_summary.ps_original.copy()
    
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
def CalcPortfolioHistoricalCost(platform=None, start_date=None, base_ccy='HKD'):
    if start_date is None:
        tn = setup.GetAllTransactions()
        supported_instruments = setup.GetListOfSupportedInstruments()
        tn = tn[tn.BBGCode.isin(supported_instruments)]
        start_date = tn.Date.min()
    
    #platform='FSM HK'
    tn_cost = setup.GetTransactionsETFs()
    tn_cost = tn_cost[tn_cost.Date > start_date]
    
    # need to add balance brought forward
    bf = _GetExistingHoldings(start_date)

    if platform is not None:
        tn_cost = tn_cost[tn_cost.Platform==platform]
        bf = bf[bf.Platform==platform]
    
    for i in range(len(bf)):
        row = bf.iloc[i]
        dic = {'Platform':row.Platform,
               'Date':start_date,
               'Type':'Buy',
               'BBGCode':row.BBGCode,
               'CostInPlatformCcy':row.CostInPlatformCcy,
               'NoOfUnits':row.NoOfUnits
               }
        tn_cost = tn_cost.append(dic, ignore_index=True)
    tn_cost.sort_values(['Date','BBGCode'], inplace=True)

    # convert all values into HKD before aggregating (need to convert platform ccy to HKD)
    platforms = list(tn_cost.Platform.unique())
    platform_ccys = [setup.GetPlatformCurrency(x) for x in platforms]
    platform_ccy_mapping = {platforms[i]: platform_ccys[i] for i in range(len(platforms))}
    tn_cost['PlatformCcy'] = tn_cost.Platform.map(platform_ccy_mapping)
    ccys = tn_cost.PlatformCcy.unique()
    fx_rate = []
    for i in range(len(ccys)):
        ccy = ccys[i]
        if ccy==base_ccy:
            rate = 1
        else:
            rate = calc_fx.ConvertFX(ccy,base_ccy)
        fx_rate.append(rate)
    ToBaseCcyRate = {ccys[i]:fx_rate[i] for i in range(len(ccys))}
    tn_cost['ToHKDrate'] = tn_cost.PlatformCcy.map(ToBaseCcyRate)
    tn_cost['CostInBaseCcy'] = tn_cost.ToHKDrate * tn_cost.CostInPlatformCcy
    
    agg = tn_cost.groupby(['Date']).agg({'CostInBaseCcy':'sum'})
    agg = agg.reset_index()
    agg['AccumCostHKD'] = agg.CostInBaseCcy.cumsum()
    agg.drop(['CostInBaseCcy'], axis=1, inplace=True)
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
def CalcIRR(platform=None, bbgcode=None, period=None):
    #platform = 'FSM HK'
    #bbgcode = 'ARKK US'
    #period = None      #since inception
    #period = 'YTD'
    #period = '1Y'
    #period = '3M'
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

    # get the start date for cashflows (the sum of anything before needs to be added as a single cashflow)
    date_range_start = util.GetStartDate(period)
    
    # apply the start date from applicable transactions
    earliest_transaction_date = df.Date.min()
    date_range_start_dt = earliest_transaction_date
    
    PerformCalc = True
    # determine if there is previous data (i.e. whether to add cost brought forward as cashflow)
    if period is None:
        # if period is not defined (i.e. since inception), take the earliest transaction date
        hasPrevData = False
        date_range_start_dt = earliest_transaction_date
    else:
        # if period is defined (e.g. 3Y), check whether there are transactions before 3Y
        hasPrevData = (len(df[df.Date<date_range_start_dt]) > 0)
        date_range_start_dt = datetime.datetime.combine(date_range_start, datetime.datetime.min.time())
        df = df[df.Date >= date_range_start_dt]
        if earliest_transaction_date > date_range_start:
            # if the first transaction is after the beginning of specified period, no need to calc IRR
            irr = np.nan
            PerformCalc = False
            dic = {'StartDate':date_range_start_dt,
                   'InitialCashflow':None,
                   'FinalCashflow':None,
                   'IRR':irr}

    if PerformCalc:
        # process cashflows
        cf = df[df.Date >= date_range_start_dt].copy()
        cf.loc[cf.Type=='Buy', 'Cashflow'] = cf.loc[cf.Type=='Buy', 'CostInPlatformCcy'] * -1
        # realised PnL needs to be taken into account to the cashflow calculation too!
        cf.loc[cf.Type=='Sell', 'Cashflow'] = cf.loc[cf.Type=='Sell', 'CostInPlatformCcy'] * -1 + cf.loc[cf.Type=='Sell', 'RealisedPnL']
        #+ cf.loc[cf.Type=='Sell', 'RealisedPnL'] * -1
        cf.loc[cf.Type=='Dividend', 'Cashflow'] = cf.loc[cf.Type=='Dividend', 'RealisedPnL']

        # get platform and currency
        platforms = list(cf.Platform.unique())
        currencies = [setup.GetPlatformCurrency(x) for x in platforms]
        platform_ccy = {platforms[x]: currencies[x] for x in range(len(platforms))}
        cf['PlatformCcy'] = cf.Platform.map(platform_ccy)
        cf = cf[['Date','PlatformCcy','Cashflow']]
        # calculate HKD equivalent
        SGDHKD = calc_fx.GetFXRate('HKD','SGD')
        ToHKD = {'HKD':1, 'SGD':SGDHKD}
        cf['CashflowInHKD'] = cf.PlatformCcy.map(ToHKD) * cf.Cashflow
    
        # need to add initial and final cashflows (valuation at beginning, valuation at end)
        # get valuations (beginning, ending)
        if bbgcode is None:
            val = CalcPortfolioHistoricalValuation(platform=platform, bbgcode=bbgcode, start_date=date_range_start_dt)
            val.rename(columns={'ValuationHKD':'Cashflow'}, inplace=True)
        else:
            val = _CalcValuation(bbgcode=bbgcode, start_date=date_range_start_dt)
            val.rename(columns={'ValuationHKD':'Cashflow'}, inplace=True)
            val = val[['Date','Cashflow']]
    
        # valuation as of start date
        if period is not None:
            val_start = (val[(val.Date<=np.datetime64(date_range_start_dt)) & (val.index==val[val.Date<=np.datetime64(date_range_start_dt)].index.max())]).copy()
            val_start.loc[:,'Cashflow'] = val_start.loc[:,'Cashflow'] * -1
            val_start.rename(columns={'Cashflow':'CashflowInHKD'}, inplace=True)
            cf = cf.append(val_start)
        else:
            val_start = pd.DataFrame(data={'Date':date_range_start_dt,'CashflowInHKD':0},
                                     columns=['Date','CashflowInHKD'],
                                     index=[0])
        
        # latest valuation
        val_end = val[val.index==val.index.max()].copy()
        val_end.rename(columns={'Cashflow':'CashflowInHKD'}, inplace=True)
    
        # add latest valuation as final cashflow (only if there are still holdings)
        #if (cf.Date.iloc[-1] != val.Date.iloc[0]) and val.Cashflow.iloc[0]!=0:
        if val.Cashflow.iloc[-1]!=0:
            cf = cf.append(val_end, ignore_index=True)
    
        cf = cf.sort_values(['Date'])
        cf = cf.reset_index(drop=True)
    
        # annualised return
        annualised_irr = _xirr(values=cf.CashflowInHKD.to_list(), dates=cf.Date.to_list())
        
        # convert back to period if period is < a year
        no_of_days = (pd.to_datetime(cf.iloc[-1].Date) - pd.to_datetime(cf.iloc[0].Date)).days
        if no_of_days < 365:
            irr = (1+annualised_irr)**(no_of_days/365)-1
        else:
            irr = annualised_irr

        # return the calc results
        dic = {'StartDate':date_range_start_dt,
               'EndDate':cf.tail(1).Date.iloc[0],
               'InitialCashflow':val_start.CashflowInHKD.iloc[0],
               'FinalCashflow':val_end.CashflowInHKD.iloc[0],
               'IRR':irr}
    
    return dic
#CalcIRR('FSM HK')
#CalcIRR('FSM SG', bbgcode='SCHSEAI SP')
#CalcIRR('FSM SG')
#CalcIRR('FSM HK', period='1w')
#CalcIRR('FSM HK', period='1m')
#CalcIRR('FSM HK', period='3m')
#CalcIRR('FSM HK', period='6m')
#CalcIRR('FSM HK', period='1y')
#CalcIRR('FSM HK', period='ytd')
#CalcIRR()
#CalcIRR(period='YTD')
#CalcIRR(period='1W')
#CalcIRR(period='1M')
# print (CalcIRR(period='3M'))
# print (CalcIRR(period='6M'))
# print (CalcIRR(period='1Y'))
# print (CalcIRR(period='3Y'))


# Calculates SPX performance: YTD 1W 1M 3M 6M 1Y 3Y 5Y
def GetSPXReturns():
    spx = mdata.GetHistoricalSPX()
    date_ranges = ['YTD','1W','1M','3M','6M','1Y','3Y','5Y']
    start_dates = {}
    for i in range(len(date_ranges)):
        start_dates[date_ranges[i]] = util.GetStartDate(date_ranges[i])
    
    # gets the last price before the specified date
    def _GetPrice(dt):
        price = spx[spx.Date<=dt].tail(1).SPX.iloc[0]
        return price

    spx_prices = {}
    for i in range(len(date_ranges)):
        spx_prices[date_ranges[i]] = _GetPrice(start_dates[date_ranges[i]])
        
    df = pd.DataFrame(data=spx_prices, index=[0])
    df = df.melt(var_name='DateRange',value_name='Price')
    df['LatestPrice'] = spx.tail(1).SPX.iloc[0]
    df['CumulativeReturn'] = df.LatestPrice / df.Price - 1
    
    # annualise returns for those beyond 1Y
    def _AnnualiseReturn(cum_return: float, date_range: str) -> float:
        if date_range[-1]=='Y':
            years = int(date_range[:-1])
            ar = ((cum_return + 1)**(1 / years)) - 1
        else:
            ar = cum_return
        return ar
    
    # apply the annualised return
    for i in range(len(df)):
        df.loc[i,'AnnualisedReturn'] = _AnnualiseReturn(df.loc[i,'CumulativeReturn'], df.loc[i,'DateRange'])
    
    df.set_index('DateRange', inplace=True)
    return df


# get the returns of each supported instruments
#setup.GetListOfSupportedInstruments()



