# -*- coding: utf-8 -*-
"""
PORTFOLIO TRACKER - CALCULATIONS MODULE


Created on Sat Sep 26 12:58:22 2020

@author: Wilson Leong


Disadvantages of the Modified Dietz Return
The Modified Dietz Return formula exhibits disadvantages when one or more large cash flows occur during the investment period or when the investment is very volatile, and experiences returns that are significantly non-linear. Another disadvantage is that the investor needs to know the value of the investment both at the beginning and end of the investment horizon.

Additionally, the investor must adopt a way to keep track of the cash flows coming in and out of the portfolio. It is important to know when to use the Modified Dietz Return to get an accurate understanding of how the investment portfolio performed.


"""

from setup import *
import pandas as pd
import datetime
import dateutil.relativedelta
import mdata

_special_ccys = ['AUD','NZD','EUR','GBP']    # those ccys that don't use USD as base ccy
_last_nav_file = mdata.LastNavFilePath


# function to get latest FX rate collected on mongodb
# def _GetLatestFXrate(ccypair):
#     # requirement 1: ccypair must be in standard market convention
#     # requirement 2: ccypair must be in the list collected from bbg
#     ccypair = ccypair.upper()

#     # get the last updated rates from mongodb (the update should be run as part of the main)
#     db = ConnectToMongoDB()
#     fx_ccypair = pd.DataFrame(list(db.FX.find({})))
#     fx_ccypair = fx_ccypair[fx_ccypair.Ccypair==ccypair]

#     if len(fx_ccypair):
#         fx = fx_ccypair.PX_LAST.iloc[0]
#     else:
#         fx = None
#         print ('\nERROR: %s needs to be added to the list' % ccypair)
#     return fx

# function to get latest FX rate from Yahoo Finance
def _GetLatestFXrate(ccypair):
    # requirement 1: ccypair must be in standard market convention
    # requirement 2: ccypair must be in the list collected from bbg
    #ccypair='USDHKD'
    ccypair = ccypair.upper()
    tmp = mdata.GetLatestPrice(mdata.Ccypair2YFTicker(ccypair))
    fx = tmp['last_price']
    return fx


# function to invert ccypair
def _InvertCcypair(ccypair):
    ccy1 = ccypair[:3]
    ccy2 = ccypair[-3:]
    ccypair = ccy2 + ccy1
    return ccypair


# function to get conversion pair
def _GetConversionPair(ccy1, ccy2):
    if ccy1 != 'USD' and ccy2 !='USD':
        # requirement: ccy1 or ccy2 must be USD
        print ('ERROR: this function does not support decrossing. USD must be ccy1 or ccy2')
        ccypair = None
    else:
        ccypair = ccy1 + ccy2
        # if one of the ccy is one of those that don't use USD as base ccy
        if ccy2 in _special_ccys:
            ccypair = _InvertCcypair(ccypair)
    return ccypair


# function to work out which ccypair rate: JPY to HKD --> USDHKD / USDJPY
def ConvertFX(ccy_source, ccy_target):
    # usage: ccy_target = returned fx rate * ccy_target
    #ccy_source='EUR'
    #ccy_target='SGD'

    # if no conversion required, return 1
    if ccy_source==ccy_target:
        fx = 1
    else:
        # if one of the ccys is USD then no decrossing required
        if ccy_source=='USD' or ccy_target=='USD':
            ccypair = ccy_source + ccy_target

            # check if its inverted (normally USD as base, unless special ccys)
            if ccy_source != 'USD':
                # invert if ccy1 isnt a special ccy
                if ccy_source not in _special_ccys:
                    ccypair = _InvertCcypair(ccypair)
                    fx = 1/_GetLatestFXrate(ccypair)
                else:
                    fx = _GetLatestFXrate(ccypair)

            else:
                # if source is USD, check if target is a special ccy (if so, invert it)
                if ccy_target in _special_ccys:
                    ccypair = _InvertCcypair(ccypair)
                    fx = 1/_GetLatestFXrate(ccypair)
                else:
                    fx = _GetLatestFXrate(ccypair)

        else:
            # decrossing required
            ccypair1 = 'USD' + ccy_source
            if ccy_source in _special_ccys:
                fx1 = 1/_GetLatestFXrate(_InvertCcypair(ccypair1))
            else:
                fx1 = _GetLatestFXrate(ccypair1)
    
            ccypair2 = 'USD' + ccy_target
            if ccy_target in _special_ccys:
                fx2 = 1/_GetLatestFXrate(_InvertCcypair(ccypair2))
            else:
                fx2 = _GetLatestFXrate(ccypair2)

            # get the conversion rate
            fx = fx2/fx1

    return fx


# Get latest NAV and last updated timestamp of existing holdings across all platforms
def GetLastNAV():
    # get all transactions from MongoDB
    tran = GetAllTransactions()
    #tran['NoOfUnits'] = tran.NoOfUnits.astype(np.float32)
    tran_summary = tran.groupby(['Platform','BBGCode']).aggregate({'NoOfUnits':'sum'})
    tran_summary = tran_summary.reset_index(drop=False)
    # filter transactions
    tran_summary = tran_summary[tran_summary.NoOfUnits>0.001]
    tran_summary = tran_summary[tran_summary.Platform.str[:4]!='Cash']
    # exclude Singapore cash fund
    tran_summary = tran_summary[tran_summary.BBGCode!='CASHFND SP']
    
    # enrich with platform and security currency
    dfPlatforms = pd.DataFrame()
    dfPlatforms['Platform'] = tran_summary.Platform.unique()
    dfPlatforms['PlatformCcy'] = [GetPlatformCurrency(x) for x in list(tran_summary.Platform.unique())]
    tran_summary = tran_summary.merge(dfPlatforms, how='left', left_on='Platform', right_on='Platform')
    secs = GetSecurities()
    tran_summary = tran_summary.merge(secs[['BBGCode','Currency']], how='left', left_on='BBGCode', right_on='BBGCode')
    tran_summary.rename(columns={'Currency':'SecurityCcy'}, inplace=True)
    
    # enrich with last NAV
    lastnav = pd.read_excel(mdata.LastNavFilePath)
    lastnav.rename(columns={'Ticker_BBG':'BBGCode'}, inplace=True)
    tran_summary = tran_summary.merge(lastnav[['BBGCode','LastNAV','LastUpdated']], how='left', left_on='BBGCode', right_on='BBGCode')

    # calculate FX rate
    for i in range(len(tran_summary)):
        row = tran_summary.iloc[i]
        if row.PlatformCcy==row.SecurityCcy:
            tran_summary.loc[i,'FXRate'] = 1
        else:
            tran_summary.loc[i,'FXRate'] = GetFXRate(row.PlatformCcy, row.SecurityCcy)

    # format output columns
    tran_summary.rename(columns={'SecurityCcy':'SecurityCurrency'}, inplace=True)
    tran_summary.drop(columns=['PlatformCcy','NoOfUnits'], inplace=True)

    #print ('(%d securities successfully collected from Excel spreadsheet)' % len(tran_summary))
    #tran_summary.to_csv('LastNAV.csv', index=False)

    # save results on MongoDB
    db = ConnectToMongoDB()
    LastNAV = db['LastNAV']
    LastNAV.delete_many({})
    LastNAV.insert_many(tran_summary[['BBGCode','LastNAV','SecurityCurrency','FXRate','LastUpdated']].to_dict('records'))
    return tran_summary



def GetPortfolioSummary():
    # get a summary of transactions in the portfolio
    tn = GetAllTransactions()
    tn.CostInPlatformCcy = tn.CostInPlatformCcy.round(2)
    tn.drop('_id', axis=1, inplace=True)
    sec = GetSecurities()
    sec.drop('_id', axis=1, inplace=True)

    # enrich transactions with Security metadata
    tn = pd.merge(tn, sec, how='left', left_on='BBGCode', right_on='BBGCode')

    agg = {'NoOfUnits':sum, 'CostInPlatformCcy':sum, 'RealisedPnL':sum} 
    summary = tn.groupby(['Platform','Name','BBGCode','BBGPriceMultiplier','Currency']).agg(agg)
    summary.reset_index(inplace=True)
    summary.rename(columns={'Currency':'SecurityCurrency'},inplace=True)

    # enrich with platforms
    db = ConnectToMongoDB()
    platforms = pd.DataFrame(list(db.Platform.find()))
    summary = pd.merge(summary, platforms, how='left', left_on='Platform', right_on='PlatformName')
    summary.drop(['PlatformName','_id','SecurityCurrency'], axis=1, inplace=True)

    # enrich transactions with the latest price
    lastnav = GetLastNAV()
    
    summary = summary.merge(lastnav[['BBGCode','LastNAV','SecurityCurrency']], how='left', left_on='BBGCode', right_on='BBGCode')
    #summary = pd.merge(summary, lastnav, how='left', left_on='BBGCode', right_on='BBGCode')
    #summary.drop(['FXRate'], axis=1, inplace=True)
    
    # added 22 Nov 2018 (remove unused stock code)
    summary = summary[summary.SecurityCurrency.notnull()]
    summary.reset_index(inplace=True, drop=True)

    for i in range(len(summary)):
        #summary.loc[i,'FXConversionRate'] = GetFXRate(summary.loc[i,'PlatformCurrency'], summary.loc[i,'SecurityCurrency'])
        summary.loc[i,'FXConversionRate'] = ConvertFX(summary.loc[i,'SecurityCurrency'], summary.loc[i,'PlatformCurrency'])
        summary['CurrentValue'] = summary.NoOfUnits * summary.LastNAV * summary.FXConversionRate / summary.BBGPriceMultiplier
#    summary['CurrentValue'] = summary.NoOfUnits * summary.LastNAV * summary.FXRate / summary.BBGPriceMultiplier
    summary.CurrentValue = summary.CurrentValue.round(2)
    summary['PnL'] = summary.CurrentValue - summary.CostInPlatformCcy #- summary.RealisedPnL
    summary.PnL = summary.PnL.round(2)
#    summary_WithLastNAVTimestamp = summary[['BBGCode','LastNAVTimestamp']]
    agg2 = {'NoOfUnits':sum, 'CostInPlatformCcy':sum, 'CurrentValue':sum, 'PnL':sum, 'RealisedPnL':sum}
    #agg2 = {'NoOfUnits':sum, 'CostInPlatformCcy':sum, 'CurrentValue':sum, 'PnL':sum}
    ps = summary.groupby(['Platform','PlatformCurrency','Name','BBGCode','LastNAV']).agg(agg2)
    ps.reset_index(inplace=True)
#    ps = pd.merge(ps, summary_WithLastNAVTimestamp, how='left', left_on='BBGCode', right_on='BBGCode')

    ps['PnLPct'] = ps.PnL / ps.CostInPlatformCcy

    # export Portfolio Summary for later use
    ps.to_csv('PortfolioSummary.csv', index=False)
    UploadLatestPortfolioSummary(ps)

    return ps


def _GetPlatformDef():
    db = ConnectToMongoDB()
    coll = db['Platform']
    df = pd.DataFrame(list(coll.find()))
    df.drop('_id', axis=1, inplace=True)
    return df


def GetHistoricalRealisedPnL():
    tn = GetAllTransactions()
    tn.CostInPlatformCcy = tn.CostInPlatformCcy.round(2)
    tn.drop('_id', axis=1, inplace=True)
    
    # enrich transactions with Platform metadata
    pl = _GetPlatformDef()
    tn = pd.merge(tn, pl, how='left', left_on='Platform', right_on='PlatformName')
    

    # enrich transactions with Security metadata
    sec = GetSecurities()
    sec.drop(['_id', 'Currency'], axis=1, inplace=True)
    tn = pd.merge(tn, sec, how='left', left_on='BBGCode', right_on='BBGCode')

    hist = tn[tn.RealisedPnL.notnull()]
    pnl = hist.groupby(['Platform','Name','Type','PlatformCurrency']).RealisedPnL.sum()
    return pnl


def _GetSecurityCategory(name):
    #name='Cash NZD'
    db = ConnectToMongoDB()
    coll = db['Security']
    df = pd.DataFrame(list(coll.find()))
    match = df[df.Name==name]
    category = match.iloc[0].Category
    return category


def GetFXRate(target_ccy, original_ccy):
    #target_ccy, original_ccy ='HKD','GBP'
    if target_ccy==original_ccy:
        fxrate = 1
    else:
        ccypair = original_ccy + target_ccy
        # use exchange rate in LastNAV
        db = ConnectToMongoDB()
        coll = db['FX']
        df = pd.DataFrame(list(coll.find()))
        
        # check if there is data, if not try inverted
        if len(df[df.Ccypair==ccypair]) > 0:
            row = df[df.Ccypair==ccypair].iloc[0]
            fxrate = row.PX_LAST
        else:
            ccypair = ccypair[-3:] + ccypair[:3]
            row = df[df.Ccypair==ccypair].iloc[0]
            fxrate = 1/row.PX_LAST

        if original_ccy=='JPY':
            fxrate = 1/fxrate
    return fxrate


def ConvertTo(target_ccy, original_ccy, original_amount):
    #target_ccy, original_ccy, original_amount = 'HKD','SGD',1000
#    target_ccy = 'HKD'
#    original_ccy = 'SGD'
#    original_amount = 87421.95
    rate = GetFXRate(target_ccy, original_ccy)
    target_ccy_amount = original_amount * rate
    return target_ccy_amount
#ConvertTo('HKD','SGD',1000)


def GetPnLUnrealised():
    #ps = GetPortfolioSummary()
    ps_active = ps[ps.CurrentValue!=0]
    PnLByPlatformAndAccount = ps_active.groupby(['Platform','Name','PlatformCurrency']).agg({'CurrentValue':sum,'PnL':sum})
    PnLByPlatform = ps_active.groupby(['PlatformCurrency','Platform']).agg({'CostInPlatformCcy':sum,'CurrentValue':sum,'PnL':sum})

    obj = {}
    obj['PnLByPlatformAndAccount'] = PnLByPlatformAndAccount
    obj['PnLByPlatform'] = PnLByPlatform
    return obj


def GetPortfolioComposition(target_ccy='HKD'):
    PnLByPlatformAndAccount = GetPnLUnrealised()['PnLByPlatformAndAccount']
    pcr = PnLByPlatformAndAccount.reset_index()
    for i in range(len(pcr)):
        row = pcr.loc[i]
        
        # get HKD equivalent amount
        ccy = row.PlatformCurrency
        value_ccy = row.CurrentValue
        pcr.loc[i, 'CurrentValueInHKD'] = ConvertTo(target_ccy, ccy, value_ccy)
        
        # get Category
        sec_name = row.Name
        pcr.loc[i, 'Category'] = _GetSecurityCategory(sec_name)
    return pcr


def UploadLatestPortfolioSummary(ps):
    print ('\nUpdating Portfolio Summary on MongoDB...')
    db = ConnectToMongoDB()
    coll = db['LatestPortfolioSummary']
    coll.delete_many({})
    coll.insert_many(ps.to_dict('records'))
    print ('(update completed)')


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
    df = GetAllTransactions()

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
    #cf['NoOfDays'] = (datetime.datetime.today() - cf.Date) / pd.Timedelta(days=1)
    cf.loc[:,'NoOfDays'] = cf.loc[:,'NoOfDays'].astype(int)
    #cf['NoOfDays'] = cf['NoOfDays'].astype(int)
    cf.loc[:,'WeightedCF'] = cf.loc[:,'CostInPlatformCcy'] * cf.loc[:,'NoOfDays'] / date_diff
    #cf['WeightedCF'] = cf.CostInPlatformCcy * cf['NoOfDays'] / date_diff
    
    # pnl = current value + realised gains/losses
    
    # realised gains/losses (sold securities + dividends)
    pnl_realised = df.RealisedPnL.sum()
    
    # unrealised pnl
    if bbgcode is not None:
        if len(ps_active.BBGCode==bbgcode) > 0:
            r = ps_active[ps_active.BBGCode==bbgcode].iloc[0]
            # current value needs to include realised gains & dividends
            current_value = r.CurrentValue + pnl_realised
    else:
        #ps = GetPortfolioSummary()
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


def TopHoldings(ps):
    df = ps.copy()
    # need to convert all to HKD first
    for i in range(len(df)):
        row = df.iloc[i]
        if row.PlatformCurrency=='HKD':
            df.loc[i, 'CurrentValueHKD'] = df.loc[i, 'CurrentValue']
        else:
            df.loc[i, 'CurrentValueHKD'] = ConvertTo('HKD', df.loc[i, 'PlatformCurrency'], df.loc[i, 'CurrentValue'])
    df.loc[:,'PortfolioPct'] = df.loc[:,'CurrentValueHKD'] / df.CurrentValueHKD.sum()
    df = df.sort_values(['CurrentValueHKD'], ascending=False)[['Name','CurrentValueHKD','PortfolioPct']].head(10)
    df = df.reset_index(drop=True)
    return df


# plot investment cost time series
def GetListOfETFs():
    sec = GetSecurities()
    etf = sec[sec.AssetType=='ETF']
    #etf = sec[sec.AssetType.isin(['Stock','ETF'])]
    list_of_etfs = list(etf.BBGCode.unique())
    return list_of_etfs


# get ETF transactions on FSM HK only
def GetTransactionsETFs():
    list_of_etfs = GetListOfETFs()
    tn = GetAllTransactions()
    tn_etf = tn[(tn.BBGCode.isin(list_of_etfs)) & (tn.Platform=='FSM HK')]
    tn_etf_cost = tn_etf[tn_etf.Type.isin(['Buy','Sell'])]
    return tn_etf_cost



# use Yahoo Finance API to collect historical prices
import yfinance as yf
yf.pdr_override()
from pandas_datareader import data as pdr
#data = pdr.get_data_yahoo('VWO', start='2020-02-24', end=datetime.datetime.today())
#data = pdr.get_data_yahoo('HKD=X', start='2020-02-24', end=datetime.datetime.today())


# determine which date ranges to collect historical data for
def _GetETFDataDateRanges(bbgcode):
    #bbgcode='SPY US'
    tn = GetTransactionsETFs()
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
#_GetETFDataDateRanges('IVV US')
#_GetETFDataDateRanges('XLE US')


# get list of ETFs and date ranges, and query
def GetHistoricalData():
    list_of_etfs = GetListOfETFs()
    # populate list of ETFs and date ranges
    df = pd.DataFrame(columns=['BBGCode','YFTicker','DateFrom','DateTo'])
    for i in range(len(list_of_etfs)):
        bbgcode = list_of_etfs[i]
        yf_ticker = GetYahooFinanceTicker(bbgcode)
        dates = _GetETFDataDateRanges(bbgcode)
        date_from = dates['DateFrom']
        date_to = dates['DateTo']
        df = df.append({'BBGCode':bbgcode,'YFTicker': yf_ticker,'DateFrom': date_from,'DateTo': date_to}, ignore_index=True)

    # loop through the list and collect the data from Yahoo
    data = pd.DataFrame()
    for i in range(len(df)):
        row = df.iloc[i]
        tmp = pdr.get_data_yahoo(row.YFTicker, start=row.DateFrom, end=row.DateTo)
        tmp = tmp.reset_index()
        tmp['BBGCode'] = row.BBGCode
        data = data.append(tmp, ignore_index=False)
    data.to_csv('HistoricalPrices.csv', index=False)
    return data
historical_data = GetHistoricalData()
usdhkd = pdr.get_data_yahoo('HKD=X', start='2020-02-24', end=datetime.datetime.today())
usdhkd = usdhkd[['Close']]
usdhkd.columns = ['USDHKDrate']

# returns the USDHKD exchange rate
def _USDHKDrate(date=None):
    # if no date specified, return the latest rate
    if date is None:
        rate = usdhkd.iloc[-1].Close
    else:
        rate = usdhkd[usdhkd.index==datetime.datetime(2020, 2, 25)].Close.iloc[0]
    return rate

# calculate the value of a security (returns time series) - this works for US ETFs only
def _CalcValuation(bbgcode):
    #bbgcode='SPY US'
    #bbgcode='VWO US'
    hd = historical_data[historical_data.BBGCode==bbgcode]
    hd = hd[['Date','Close']]
    hd = hd.sort_values(['Date'], ascending=True)
    tn = GetTransactionsETFs()
    tn = tn[tn.BBGCode==bbgcode]
    tn = tn[['Date','NoOfUnits']]
    df = pd.merge(hd, tn, how='left', on='Date')
    df.NoOfUnits.fillna(0, inplace=True)
    df['Holdings'] = df.NoOfUnits.cumsum()
    df['ValuationUSD'] = df.Holdings * df.Close
    
    # get USDHKD rate
    df = df.merge(usdhkd, how='left', on='Date')
    df['USDHKDrate'] = df.USDHKDrate.fillna(method='ffill')
    df['ValuationHKD'] = df.ValuationUSD * df.USDHKDrate
    return df


# calculate the value of the entire portfolio (add up each security in the portfolio)
def CalcPortfolioHistoricalValuation():
    list_of_etfs = GetListOfETFs()
    df = pd.DataFrame()
    
    # loop through the list
    for i in range(len(list_of_etfs)):
        bbgcode = list_of_etfs[i]
        tmp = _CalcValuation(bbgcode)
        # remove redundant rows
        tmp = tmp[~((tmp.NoOfUnits==0) & (tmp.Holdings==0))]
        tmp['BBGCode'] = bbgcode
        df = df.append(tmp, ignore_index=False)

    # group the data by date
    agg = df.groupby(['Date']).agg({'ValuationHKD':'sum'})
    agg = agg.reset_index()
    return agg


# calculate the cost of the entire portfolio (add up each transaction in the portfolio)
def CalcPortfolioHistoricalCost():
    tn_etf_cost = GetTransactionsETFs()
    #tn_etf_cost['AccumCost'] = tn_etf_cost.CostInPlatformCcy.cumsum()
    agg = tn_etf_cost.groupby(['Date']).agg({'CostInPlatformCcy':'sum'})
    agg = agg.reset_index()
    agg['AccumCostHKD'] = agg.CostInPlatformCcy.cumsum()
    agg.drop(['CostInPlatformCcy'], axis=1, inplace=True)
    return agg


# return the valuation 
hist_valuation = CalcPortfolioHistoricalValuation()     # this is for US ETFs only

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
    row = hist_valuation[hist_valuation.Date<=start_date]
    if len(row)==0:
        val = 0
    else:
        val = row.ValuationHKD.iloc[-1]
    return val

def _GetCashFlows(start_date):
    #start_date=datetime.datetime(2020,9,1)
    tn = GetTransactionsETFs()
    tn.drop(['_id'], axis=1, inplace=True)
    cashflows = tn[tn.Date>=start_date]
    return cashflows


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


def CalcIRR(platform='FSM HK', bbgcode=None, period=None):
    #platform = 'FSM HK'
    #bbgcode = 'SPY US'
    #period = None
    df = GetAllTransactions()
    list_of_etfs = GetListOfETFs()
            
    # filter the data based on selection criteria (bbgcode, platform)
    df.drop(['_id'], axis=1, inplace=True)
    df = df[df.BBGCode.isin(list_of_etfs)]
    df = df[df.Platform==platform]
    if bbgcode is not None:
        df = df[df.BBGCode==bbgcode]

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
        df = df[['Date','Cashflow']]
    
        # get valuations (beginning, ending)
        if bbgcode is None:
            val = CalcPortfolioHistoricalValuation()
            val.rename(columns={'ValuationHKD':'Cashflow'}, inplace=True)
        else:
            val = _CalcValuation(bbgcode) # bug: if bbgcode is NOne....??????
            val.rename(columns={'ValuationHKD':'Cashflow'}, inplace=True)
            val = val[['Date','Cashflow']]
        # latest valuation up to start date
        if period is  not None:
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








# get calculations for other modules to use
ps = GetPortfolioSummary()
top_holdings = TopHoldings(ps)
pnl_unrealised = GetPnLUnrealised()







