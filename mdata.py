# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 21:00:31 2020

@author: Wilson Leong
"""

### Parameters ###
LastNavFilePath = r'D:\Wilson\Documents\Personal Documents\Investments\PortfolioTracker\LastNAV.xlsx'
### End of parameters ###



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


def ProcessHistoricalMarketData(bbgcode=None, platform=None, start_date=None):
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
    #tmp2.to_csv('HistoricalPrices.csv', index=False)
    
    # save to mongodbo
    db = setup.ConnectToMongoDB()
    
    coll = db['HistoricalMarketData']
    # clear all previous transactions
    coll.delete_many({})

    # insert rows into the db
    coll.insert_many(tmp2.to_dict('records'))
    return tmp2
