# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 21:00:31 2020

@author: Wilson Leong
"""

### Parameters ###
LastNavFilePath = r'D:\Wilson\Documents\Personal Documents\Investments\PortfolioTracker\LastNAV.xlsx'
### End of parameters ###



import datetime
import pandas as pd
import yfinance as yf
yf.pdr_override()
from pandas_datareader import data as pdr
import setup


# gets the latest price & updated timestamp for a ticker (stock, ETF, currency pair, etc.)
def GetLatestPrice(ticker, display_log=False):
    #ticker='EURUSD=X'
    #ticker='0P00006G0B.SI'
    if display_log:
        print ('Getting latest price for "%s"' % ticker)
    t = yf.Ticker(ticker)
    #last_price_row = t.history(period='1w')
    last_price_row = t.history()
    last_price = last_price_row.Close.iloc[-1]
    last_updated = last_price_row.index[-1]
    if display_log:
        print ('> latest price for "%s": %s' % (ticker, last_price))
    data = {}
    data['last_price'] = last_price
    data['last_updated'] = last_updated
    return data
#_GetLatestPrice('0P00006G0B.SI')


# determines the Yahoo Finance ticker symbol based on currency pair
def Ccypair2YFTicker(ccypair):
    #ccypair='EURUSD'
    #ccypair='USDHKD'
    ccy1 = ccypair[:3]
    ccy2 = ccypair[3:]
    if ccy1=='USD':
        ticker = ccy2 + '=X'
    else:
        ticker = ccypair + '=X'
    return ticker


# update the last NAV file with latest prices from Yahoo Finance
def UpdateLastNAV():
    # reads the file, segregate the manual pricing source
    print ('\nUpdating latest NAV - manual & Yahoo Finance API...')
    df = pd.read_excel(LastNavFilePath)
    df_manual = df[df.Ticker_YF.isnull()].copy()
    df_api = df[~df.Ticker_YF.isnull()].copy()
    
    # loop through each item in the list and update the price
    df_api = df_api.reset_index(drop=True)
    for i in range(len(df_api)):
        row = df_api.iloc[i]
        ticker = row.Ticker_YF
        tmp = GetLatestPrice(ticker)
        last_price = tmp['last_price']
        last_updated = tmp['last_updated']
        df_api.loc[i,'LastNAV'] = last_price
        df_api.loc[i,'LastUpdated'] = last_updated
    df2 = df_manual.append(df_api)
    df2 = df2.reset_index(drop=True)
    df2.to_excel(LastNavFilePath, index=False)
    print ('(updated latest NAV on XLSX)')


# Collect historical market data from Yahoo Finance; fill values for closed markets; cache on DB
def ProcessHistoricalMarketData(bbgcode=None, platform=None, start_date=None):
    print ('\nProcessing historical market data...')
    tn = setup.GetAllTransactions()
    # filter by bbgcode and platform
    if bbgcode is not None:
        tn = tn[tn.BBGCode==bbgcode]
    if platform is not None:
        tn = tn[tn.Platform==platform]
    
    if start_date is None:
        supported_instruments = setup.GetListOfSupportedInstruments()
        tn = tn[tn.BBGCode.isin(supported_instruments)]
        start_date = tn.Date.min()
    
    #list_of_etfs = GetListOfETFs()
    list_of_supported_instruments = setup.GetListOfSupportedInstruments()
    
    if bbgcode is not None:
        list_of_supported_instruments = [bbgcode]
    
    # populate list of ETFs and date ranges
    df = pd.DataFrame(columns=['BBGCode','YFTicker','DateFrom','DateTo'])
    for i in range(len(list_of_supported_instruments)):
        bbgcode = list_of_supported_instruments[i]
        yf_ticker = setup.GetYahooFinanceTicker(bbgcode)
        dates = setup.GetETFDataDateRanges(bbgcode)
        date_from = dates['DateFrom']
        date_to = dates['DateTo']       # this results in incorrect values for securites no longer held
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
    
    # added 15 Dec 2020: Yahoo Finance null rows?
    data = data[~data.Close.isnull()]
    data.drop_duplicates(['BBGCode','Date'], inplace=True)
    
    # NEED TO DEAL WITH HK/US HOLIDAYS MISMATCH - this process is also adding incorrect values for securities no longer held
    tmp = data.pivot('Date','BBGCode', values='Close')
    tmp = tmp.fillna(method='ffill')
    tmp = tmp.reset_index()
    tmp2 = pd.melt(tmp, id_vars=['Date'], value_vars=list(data.BBGCode.unique()), value_name='Close')
    tmp2.dropna(inplace=True)
    #tmp2.to_csv('HistoricalPrices.csv', index=False)
    
    # save to mongodb
    db = setup.ConnectToMongoDB()
    
    coll = db['HistoricalMarketData']
    # clear all previous transactions
    coll.delete_many({})

    # insert rows into the db
    coll.insert_many(tmp2.to_dict('records'))
    #return tmp2
    print ('(updated %s records on MongoDB)' % len(tmp2))


# get historical NAV from cache (MongoDB)
def GetHistoricalData(bbgcode=None, start_date=None):
    db = setup.ConnectToMongoDB()
    coll = db['HistoricalMarketData']
    df = pd.DataFrame(list(coll.find()))
    df.drop(['_id'], axis=1, inplace=True)
    if bbgcode is not None:
        df = df[df.BBGCode==bbgcode]
    if start_date is not None:
        df = df[df.Date >= start_date]
    return df


# Collect USDHKD historical rates and cache on DB
def ProcessHistoricalUSDHKD():
    print ('\nProcessing historical USDHKD rates...')
    # collect from Yahoo Finance
    usdhkd = pdr.get_data_yahoo('HKD=X', start='2015-07-01', end=datetime.datetime.today())
    usdhkd = usdhkd[['Close']]
    usdhkd.columns = ['USDHKDrate']
    usdhkd = usdhkd.reset_index()
    
    # store on DB
    db = setup.ConnectToMongoDB()
    coll = db['USDHKD']
    coll.delete_many({})
    coll.insert_many(usdhkd.to_dict('records'))
    print ('(updated %s records on MongoDB)' % len(usdhkd))


# Get historical USDHKD from cache (MongoDB)
def GetHistoricalUSDHKD():
    db = setup.ConnectToMongoDB()
    coll = db['USDHKD']
    df = pd.DataFrame(list(coll.find()))
    df.drop(['_id'], axis=1, inplace=True)
    return df


# Collect USDHKD historical rates and cache on DB
def ProcessHistoricalSPX():
    print ('\nProcessing historical S&P 500 prices...')
    # collect from Yahoo Finance
    spx = pdr.get_data_yahoo('^GSPC', start='2015-07-01', end=datetime.datetime.today())
    spx = spx[['Close']]
    spx.columns = ['SPX']
    spx = spx.reset_index()
    
    # store on DB
    db = setup.ConnectToMongoDB()
    coll = db['SPX']
    coll.delete_many({})
    coll.insert_many(spx.to_dict('records'))
    print ('(updated %s records on MongoDB)' % len(spx))


# Get historical SPX from cache (mongodb)
def GetHistoricalSPX():
    db = setup.ConnectToMongoDB()
    coll = db['SPX']
    df = pd.DataFrame(list(coll.find()))
    df.drop(['_id'], axis=1, inplace=True)
    return df
