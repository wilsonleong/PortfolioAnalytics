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


