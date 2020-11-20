# -*- coding: utf-8 -*-
"""
PORTFOLIO TRACKER - PAD MODULE
For BNPP Singapore PAD requirements


Created on Sat Sep 26 13:38:19 2020

@author: Wilson Leong
"""


from setup import *
import pandas as pd
import datetime


def GetTransactionList(DateFrom, DateTo):
    #DateFrom = datetime.datetime(2017,4,1)
    #DateTo = datetime.datetime(2017,9,30)
    db = ConnectToMongoDB()
    coll = db['Transactions']
    df = pd.DataFrame(list(coll.find({'Date': {'$gte': DateFrom, '$lt': DateTo}, 
                                      'Type': {'$in': ['Buy', 'Sell']}, 
                                      'Platform': {'$in': ['FSM HK', 'FSM SG', 'TD UK']}
                                       }
                                       ,{'Platform':1,
                                         'BBGCode':1,
                                         'Date':1,
                                         'NoOfUnits':1,
                                         'Type':1
                                         }
                                       )))
    df.drop(['_id'], axis=1, inplace=True)
    df.sort_values(['Platform','Date'], inplace=True)
    
    # loop through the list and enrich columns
    for i in range(len(df)):
        row = df.loc[i]
        
        # get security name
        df.loc[i, 'SecurityName'] = _GetSecurityName(row.BBGCode)
        
        # buy/sell
        if row.Type=='Buy':
            df.loc[i, 'NoOfUnitsBought'] = row.NoOfUnits
            df.loc[i, 'NoOfUnitsSold'] = None
            
        elif row.Type=='Sell':
            df.loc[i, 'NoOfUnitsBought'] = None
            df.loc[i, 'NoOfUnitsSold'] = -row.NoOfUnits

        # get balance b/f
        df.loc[i,'BalanceBF'] = _GetBalBF(row.Platform, row.BBGCode, DateFrom)
        
        # get balance c/f
        df.loc[i,'BalanceCF'] = _GetBalCF(row.Platform, row.BBGCode, DateTo)
    
    # rearrange and export
    df = df[['Platform', 'BalanceBF', 'Date', 'SecurityName', 'NoOfUnitsBought', 'NoOfUnitsSold', 'BalanceCF']]
    return df


# get the security name from BBG Code
def _GetSecurityName(BBGCode):
    db = ConnectToMongoDB()
    coll = db['Security']
    df = pd.DataFrame(list(coll.find({'BBGCode': BBGCode})))
    if len(df) > 0:
        name = df.iloc[0].Name
    else:
        name = None
    return name


# get the balance brought forward (opening balance) for a security on a given date
def _GetBalBF(Platform, BBGCode, Date):
    #Platform = 'FSM SG'
    #BBGCode = 'ESJDASH LX'
    #Date = datetime.datetime(2017,4,1)
    db = ConnectToMongoDB()
    coll = db['Transactions']
    df = pd.DataFrame(list(coll.find({'Date': {'$lt': Date}, 
                                      'BBGCode': BBGCode,
                                      'Platform': Platform,
                                      'Type': {'$in': ['Buy', 'Sell']}
                                      })))
    if len(df)>0:
        bal_bf = round(df.NoOfUnits.sum(), 4)
    else:
        bal_bf = 0
    return bal_bf


# get the balance carried forward (closing balance) for a security on a given date
def _GetBalCF(Platform, BBGCode, Date):
    #BBGCode = 'ESJDASH LX'
    #Date = datetime.datetime(2017,9,20)
    db = ConnectToMongoDB()
    coll = db['Transactions']
    df = pd.DataFrame(list(coll.find({'Date': {'$lte': Date}, 
                                      'BBGCode': BBGCode,
                                      'Platform': Platform,
                                      'Type': {'$in': ['Buy', 'Sell']}
                                      })))
    if len(df)>0:
        bal_cf = round(df.NoOfUnits.sum(), 4)
    else:
        bal_cf = 0
    return bal_cf


def GetHoldingsSummary(DateFrom, DateTo):
    #DateFrom = datetime.datetime(2017,4,1)
    #DateTo = datetime.datetime(2017,9,30)
    db = ConnectToMongoDB()
    coll = db['Transactions']
    df = pd.DataFrame(list(coll.find({'Date': {'$gte': DateFrom, '$lt': DateTo}, 
                                      'Type': {'$in': ['Buy', 'Sell']}, 
                                      'Platform': {'$in': ['FSM HK', 'FSM SG', 'TD UK']}
                                       }
                                       ,{'Platform':1,
                                         'BBGCode':1,
                                         'Date':1,
                                         'NoOfUnits':1,
                                         'Type':1
                                         }
                                       )))
    df.drop(['_id'], axis=1, inplace=True)
    df.sort_values(['Platform','Date'], inplace=True)

    # sum up buy and sell
    for i in range(len(df)):
        row = df.loc[i]

        # buy/sell
        if row.Type=='Buy':
            df.loc[i, 'NoOfUnitsBought'] = row.NoOfUnits
            df.loc[i, 'NoOfUnitsSold'] = None
            
        elif row.Type=='Sell':
            df.loc[i, 'NoOfUnitsBought'] = None
            df.loc[i, 'NoOfUnitsSold'] = -row.NoOfUnits
    
    a = df.groupby(['Platform', 'BBGCode']).agg({'NoOfUnitsBought': sum, 'NoOfUnitsSold': sum})
    a = a.reset_index()

    # loop through the list and enrich columns
    for i in range(len(a)):
        row = a.loc[i]
    
        # get security name
        a.loc[i, 'SecurityName'] = _GetSecurityName(row.BBGCode)
        
#        # buy/sell
#        if row.Type=='Buy':
#            a.loc[i, 'NoOfUnitsBought'] = row.NoOfUnits
#            a.loc[i, 'NoOfUnitsSold'] = None
#            
#        elif row.Type=='Sell':
#            a.loc[i, 'NoOfUnitsBought'] = None
#            a.loc[i, 'NoOfUnitsSold'] = -row.NoOfUnits
            
        # get balance b/f
        a.loc[i,'BalanceBF'] = _GetBalBF(row.Platform, row.BBGCode, DateFrom)
        
        # get balance c/f
        a.loc[i,'BalanceCF'] = _GetBalCF(row.Platform, row.BBGCode, DateTo)

    # rearrange and export
    a['TradeDate'] = '-'
    a = a[['Platform', 'BalanceBF', 'TradeDate', 'SecurityName', 'NoOfUnitsBought', 'NoOfUnitsSold', 'BalanceCF']]
    return a



# # REPORTING for Singapore PAD (not a requirement for HK)
# DateFrom = datetime.datetime(2017, 4,1)
# DateTo = datetime.datetime(2017,9,30)
# Report_TransactionsList = GetTransactionList(DateFrom, DateTo)
# Report_HoldingsSummary = GetHoldingsSummary(DateFrom, DateTo)

