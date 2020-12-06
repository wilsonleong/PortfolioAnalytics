# -*- coding: utf-8 -*-
"""
PORTFOLIO TRACKER - SETUP MODULE


Created on Sat Sep 26 12:58:18 2020

@author: Wilson Leong
"""

##### PARAMETERS #####
_Currencies = ['GBP','USD','EUR','SGD','HKD','AUD','MOP','CNY','JPY']
_setupfile = r'D:\Wilson\Documents\Personal Documents\Investments\PortfolioTracker\setup.xlsx'
_FXfile = r'D:\Wilson\Documents\Personal Documents\Investments\PortfolioTracker\FX.xlsx'
##### END OF PARAMETERS #####


import pymongo
import numpy as np
import pandas as pd
import datetime
import mdata


def ConnectToMongoDB():
    MongoServer='localhost:27017'
    client = pymongo.MongoClient(MongoServer)
    MongoDatabaseName = 'investments'
    db = client[MongoDatabaseName]
    # collections in the "investment" db:
        # Transactions
        # Security
        # Platform
    return db


def InsertPlatform(PlatformName, Currency):
    db = ConnectToMongoDB()
    Platform = db['Platform']

    dic = {'PlatformName':PlatformName, 'PlatformCurrency':Currency}
    Platform.insert_one(dic)
    print ('(Platform "%s (%s)" added)' % (PlatformName, Currency))


def InsertSecurity(db,
                   SecurityCode, 
                   SecurityAssetClass, 
                   SecurityAssetType,
                   SecurityCategory,
                   SecurityName, 
                   SecurityCcy, 
                   SecurityFXCode,
                   BBGPriceMultiplier, 
                   FundHouse,
                   YahooFinanceTicker
                   ):
    Security = db['Security']

    # data validation
    validated = True
    if SecurityCcy not in _Currencies:
        validated = False
        print ('("%s" is not a valid currency)' % SecurityCcy)

    dic = {'BBGCode':SecurityCode,
           'AssetClass':SecurityAssetClass,
           'AssetType':SecurityAssetType,
           'Category':SecurityCategory,
           'Name':SecurityName,
           'Currency':SecurityCcy,
           'FXCode':SecurityFXCode,
           'BBGPriceMultiplier':BBGPriceMultiplier,
           'FundHouse':FundHouse,
           'YahooFinanceTicker':YahooFinanceTicker
           }

    if validated:
        Security.insert_one(dic)
        print ('(Security "%s" added)' % SecurityCode)


def GetPlatformCurrency(platform):
    # this functiton takes the platform and returns the currency
    db = ConnectToMongoDB()
    platforms = pd.DataFrame(list(db['Platform'].find()))
    Currency = platforms[platforms.PlatformName==platform].PlatformCurrency.iloc[0]
    return Currency


def GetSecurityCurrency(security):
    # this functiton takes the security and returns the currency
    #security = 'JFJAPNI HK'
    db = ConnectToMongoDB()
    securities = pd.DataFrame(list(db['Security'].find()))
    Currency = securities[securities.BBGCode==security].Currency.iloc[0]
    return Currency


def GetWeightedAvgCost(TransactionDate, BBGCode, Quantity, Platform):
    # weighted average cost (in platform currency)
#    BBGCode = 'PBISEAS ID'
#    Quantity = -127.692
#    TransactionDate = datetime.datetime(2018,5,8)
#    Platform = 'FSM SG'
    tns = GetAllTransactions()
    tns = tns[(tns.BBGCode==BBGCode) & (tns.Date <= TransactionDate) & (tns.Platform==Platform)]
    wac = tns.CostInPlatformCcy.sum() / tns.NoOfUnits.sum() * Quantity
    return wac


def InsertTransaction(db,
                      Platform,
                      Date,
                      Type,
                      BBGCode,
                      CostInPlatformCcy,
                      PriceInSecurityCcy,
                      Quantity,
                      Dividend=None,
                      Comment=None
                      ):

#    Platform = 'FSM SG'
#    Type = 'Buy'
#    Date = datetime.datetime(2017,04,16)
#    BBGCode = 'FIEMEAU LX'
#    CostInPlatformCcy = 10021.15
#    PriceInSecurityCcy = 13.14
#    Quantity = 542.73
#    Comment = 'Switch buy'

    # data validation
    validated = True
    if not isinstance(Date, datetime.date):
        validated = False
        print ('("%s" is not a valid date)' % Date)

    # get platform and security currencies
    PlatformCcy = GetPlatformCurrency(Platform)
    SecurityCcy = GetSecurityCurrency(BBGCode)

#    db = ConnectToMongoDB()
    Transactions = db['Transactions']

    # special treatments for Sell and Dividend
    if Type=='Sell':
        # if sell, force the cost and quantity to be negative
        CostInPlatformCcy = abs(CostInPlatformCcy) * -1
        Quantity = abs(Quantity) * -1

    # calculate PriceInPlaformCcy, CostInSecurityCcy, FXRate
    if PlatformCcy == SecurityCcy:
        FXRate = 1
        CostInSecurityCcy = CostInPlatformCcy
        PriceInPlatformCcy = PriceInSecurityCcy
    else:
        CostInSecurityCcy = PriceInSecurityCcy * Quantity
        PriceInPlatformCcy = CostInPlatformCcy / Quantity
        FXRate = CostInPlatformCcy / CostInSecurityCcy

    dic = {'Platform':Platform,
           'Date':Date,
           'Type':Type,
           'BBGCode':BBGCode,
           'CostInPlatformCcy':CostInPlatformCcy,
           'CostInSecurityCcy':CostInSecurityCcy,
           'PriceInPlatformCcy':PriceInPlatformCcy,
           'PriceInSecurityCcy':PriceInSecurityCcy,
           'FXRate':FXRate,
           'NoOfUnits':Quantity,
           'Comment':Comment
           }

    if Type=='Sell':
        # if selling, then Realised PnL needs to be calculated (req. weighted avg cost of acquiring)
        # added 29 Jul 2018: Realised PnL should be in Platform CCY not security CCY
        wac = GetWeightedAvgCost(Date, BBGCode, Quantity,Platform)
        RealisedPnL = round(wac - CostInPlatformCcy,2)
        dic['RealisedPnL'] = RealisedPnL
        dic['CostInPlatformCcy'] = dic['CostInPlatformCcy'] + RealisedPnL
    elif Type=='Dividend':
        RealisedPnL = Dividend
        dic['RealisedPnL'] = RealisedPnL

    if validated:
        Transactions.insert_one(dic)
        print ('(Transaction added: %s | %s: %s)' % (Date, "{:,.2f}".format(CostInPlatformCcy), BBGCode))


def UpdateLatestFXrates():
    # get the list of ccypairs
    print ('\nCollecting latest FX rates from Yahoo Finance...')
    df = pd.read_excel(_FXfile, sheet_name='Sheet1')

    # loop through each ticker, get price
    for i in range(len(df)):
        row = df.iloc[i]
        tmp = mdata.GetLatestPrice(mdata.Ccypair2YFTicker(row.Ccypair))
        df.loc[i,'Rate'] = tmp['last_price']
        df.loc[i,'LastUpdated'] = tmp['last_updated']
    
    # save latest rates into file
    df.to_excel(_FXfile, index=False)

    # changed 21 Nov 2018 (without BBG connection for market data) - rename before pushing into MongoDB
    df.rename({'Rate':'PX_LAST'}, axis=1, inplace=True)
    
    # remove old records, add new ones back
    db = ConnectToMongoDB()
    coll = db['FX']
    coll.delete_many({})
    coll.insert_many(df.to_dict('records'))
    print ('(updated latest FX rates on mongodb)')


def InitialSetup():
    print ('\nRunning initial setup...')
    # connect to MongoDB
    db = ConnectToMongoDB()
    
    # clear previous setup
    db['Platform'].delete_many({})
    db['Security'].delete_many({})
    
    SetupFile = _setupfile

    # initial setup of platforms
    dfPlatforms = pd.read_excel(SetupFile, sheet_name='Platform')
    for i in range(len(dfPlatforms)):
        row = dfPlatforms.iloc[i]
        name = row.PlatformName
        ccy = row.PlatformCurrency
        InsertPlatform(name, ccy)

    # initial setup of securities
    df = pd.read_excel(SetupFile, sheet_name='Security')
    for i in range(len(df)):
        row = df.iloc[i]
        bbgcode = row.BBGCode
        assetclass = row.AssetClass
        assettype = row.AssetType
        category = row.Category
        name = row.Name
        ccy = row.Currency
        fxcode = row.FXCode if row.FXCode is not np.nan else None
        multiplier = int(row.BBGPriceMultiplier)
        fm = row.FundHouse
        yf_ticker = row.YahooFinanceTicker if row.YahooFinanceTicker is not np.nan else None
        InsertSecurity(db, bbgcode, assetclass, assettype, category, name, ccy, fxcode, multiplier, fm, yf_ticker)


def InsertHistTransactions():
    print ('\nImporting historical transactions...')
    # connect to mongodb
    db = ConnectToMongoDB()
    
    # clear all previous transactions
    db['Transactions'].delete_many({})

    # load historical transactions
    transfile = _setupfile
    t = pd.read_excel(transfile, sheet_name='Transactions')
    t.drop(['SecurityName'], axis=1, inplace=True)

    ns = 1e-9
    for i in range(len(t)):
        Platform = t.iloc[i].Platform
        Date = t.iloc[i].Date
        Date = datetime.datetime.utcfromtimestamp(Date.astype(datetime.datetime)*ns)
        Type = t.iloc[i].Type
        BBGCode = t.iloc[i].BBGCode
        CostInPlatformCcy = round(t.iloc[i].CostInPlatformCcy,2)
        PriceInSecurityCcy = t.iloc[i].PriceInSecurityCcy
        Quantity = t.iloc[i].Quantity
        Dividend = t.iloc[i].Dividend if str(t.iloc[i].Dividend)!='nan' else None
        Comment = t.iloc[i].Comment if str(t.iloc[i].Comment)!='nan' else None
        InsertTransaction(db, Platform, Date, Type, BBGCode, CostInPlatformCcy, PriceInSecurityCcy, Quantity, Dividend, Comment)
    print ('(%d transactions added)' % len(t))


def GetAllTransactions():
    db = ConnectToMongoDB()
    Transactions = db['Transactions']
    df = pd.DataFrame(list(Transactions.find()))
    return df


def GetSecurities():
    db = ConnectToMongoDB()
    Security = db['Security']
    df = pd.DataFrame(list(Security.find()))
    return df


# get Yahoo Finance ticker
def GetYahooFinanceTicker(bbgcode):
    #bbgcode = 'VGT US'
    sec = GetSecurities()
    df = sec[sec.BBGCode==bbgcode]
    ticker = df.YahooFinanceTicker.iloc[0]
    return ticker


# get the list of securities supported by Yahoo Finance (regardless of whether there are current holdings)
def GetListOfSupportedInstruments():
    sec = GetSecurities()
    supported = sec[sec.YahooFinanceTicker.notnull()]
    list_of_supported_instruments = list(supported.BBGCode.unique())
    return list_of_supported_instruments


# get transactions of supported instruments (previously applicable to ETFs on FSM HK only)
def GetTransactionsETFs():
    list_of_supported_instruments = GetListOfSupportedInstruments()
    tn = GetAllTransactions()
    tn_etf = tn[tn.BBGCode.isin(list_of_supported_instruments)]
    tn_etf_cost = tn_etf[tn_etf.Type.isin(['Buy','Sell'])]
    return tn_etf_cost


# get bank and cash balances
def GetBankAndCashBalances():
    df = pd.read_excel(_setupfile, sheet_name='Cash')
    return df


# determine which date ranges to collect historical data for
def GetETFDataDateRanges(bbgcode):
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
