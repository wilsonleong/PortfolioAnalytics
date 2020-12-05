# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:53:05 2020

@author: Wilson Leong

This module does the following:
    - portfolio summary
    - top holdings

"""




import setup
import pandas as pd
import calc_val
import calc_fx


# store a copy of the output on MongoDB for future reference
def UploadLatestPortfolioSummary(ps):
    print ('\nUpdating Portfolio Summary on MongoDB...')
    db = ConnectToMongoDB()
    coll = db['LatestPortfolioSummary']
    coll.delete_many({})
    coll.insert_many(ps.to_dict('records'))
    print ('(update completed)')


# generate a data table summary of the portfolio based on transactions and referential data
def GetPortfolioSummary():
    # get a summary of transactions in the portfolio
    tn = setup.GetAllTransactions()
    tn.CostInPlatformCcy = tn.CostInPlatformCcy.round(2)
    tn.drop('_id', axis=1, inplace=True)
    sec = setup.GetSecurities()
    sec.drop('_id', axis=1, inplace=True)

    # enrich transactions with Security metadata
    tn = pd.merge(tn, sec, how='left', left_on='BBGCode', right_on='BBGCode')

    agg = {'NoOfUnits':sum, 'CostInPlatformCcy':sum, 'RealisedPnL':sum} 
    summary = tn.groupby(['Platform','Name','BBGCode','BBGPriceMultiplier','Currency']).agg(agg)
    summary.reset_index(inplace=True)
    summary.rename(columns={'Currency':'SecurityCurrency'},inplace=True)

    # enrich with platforms
    db = setup.ConnectToMongoDB()
    platforms = pd.DataFrame(list(db.Platform.find()))
    summary = pd.merge(summary, platforms, how='left', left_on='Platform', right_on='PlatformName')
    summary.drop(['PlatformName','_id','SecurityCurrency'], axis=1, inplace=True)

    # enrich transactions with the latest price
    lastnav = calc_val.GetLastNAV()
    
    summary = summary.merge(lastnav[['BBGCode','LastNAV','SecurityCurrency']], how='left', left_on='BBGCode', right_on='BBGCode')
    
    # added 22 Nov 2018 (remove unused stock code)
    summary = summary[summary.SecurityCurrency.notnull()]
    summary.reset_index(inplace=True, drop=True)

    for i in range(len(summary)):
        #summary.loc[i,'FXConversionRate'] = GetFXRate(summary.loc[i,'PlatformCurrency'], summary.loc[i,'SecurityCurrency'])
        summary.loc[i,'FXConversionRate'] = calc_fx.ConvertFX(summary.loc[i,'SecurityCurrency'], summary.loc[i,'PlatformCurrency'])
        summary['CurrentValue'] = summary.NoOfUnits * summary.LastNAV * summary.FXConversionRate / summary.BBGPriceMultiplier
    summary.CurrentValue = summary.CurrentValue.round(2)
    summary['PnL'] = summary.CurrentValue - summary.CostInPlatformCcy #- summary.RealisedPnL
    summary.PnL = summary.PnL.round(2)
    agg2 = {'NoOfUnits':sum, 'CostInPlatformCcy':sum, 'CurrentValue':sum, 'PnL':sum, 'RealisedPnL':sum}
    ps = summary.groupby(['Platform','PlatformCurrency','Name','BBGCode','LastNAV']).agg(agg2)
    ps.reset_index(inplace=True)

    ps['PnLPct'] = ps.PnL / ps.CostInPlatformCcy

    # added 2 Dec 2020    
    sec = sec[['BBGCode','AssetType','Currency']]
    sec.rename(columns={'AssetType':'SecurityType','Currency':'SecurityCcy'}, inplace=True)
    ps = ps.merge(sec, how='left', left_on='BBGCode', right_on='BBGCode')

    # add current value in HKD
    for i in range(len(ps)):
        row = ps.loc[i]
        # get HKD equivalent amount
        ccy = row.PlatformCurrency
        value_ccy = row.CurrentValue
        ps.loc[i, 'CurrentValueInHKD'] = calc_fx.ConvertTo('HKD', ccy, value_ccy)
        # get Category
        sec_name = row.Name
        ps.loc[i, 'Category'] = _GetSecurityCategory(sec_name)

    # # export Portfolio Summary for later use
    # ps.to_csv('PortfolioSummary.csv', index=False)
    # setup.UploadLatestPortfolioSummary(ps)

    return ps


def _GetPlatformDef():
    db = setup.ConnectToMongoDB()
    coll = db['Platform']
    df = pd.DataFrame(list(coll.find()))
    df.drop('_id', axis=1, inplace=True)
    return df


def GetHistoricalRealisedPnL():
    tn = setup.GetAllTransactions()
    tn.CostInPlatformCcy = tn.CostInPlatformCcy.round(2)
    tn.drop('_id', axis=1, inplace=True)
    
    # enrich transactions with Platform metadata
    pl = _GetPlatformDef()
    tn = pd.merge(tn, pl, how='left', left_on='Platform', right_on='PlatformName')
    

    # enrich transactions with Security metadata
    sec = setup.GetSecurities()
    sec.drop(['_id', 'Currency'], axis=1, inplace=True)
    tn = pd.merge(tn, sec, how='left', left_on='BBGCode', right_on='BBGCode')

    hist = tn[tn.RealisedPnL.notnull()]
    pnl = hist.groupby(['Platform','Name','Type','PlatformCurrency']).RealisedPnL.sum()
    return hist, pnl


def _GetSecurityCategory(name):
    #name='Cash NZD'
    db = setup.ConnectToMongoDB()
    coll = db['Security']
    df = pd.DataFrame(list(coll.find()))
    match = df[df.Name==name]
    category = match.iloc[0].Category
    return category


def GetPnLUnrealised():
    ps = GetPortfolioSummary()
    ps_active = ps[ps.CurrentValue!=0]
    PnLByPlatformAndAccount = ps_active.groupby(['Platform','Name','PlatformCurrency']).agg({'CurrentValue':sum,'PnL':sum})
    PnLByPlatform = ps_active.groupby(['PlatformCurrency','Platform']).agg({'CostInPlatformCcy':sum,'CurrentValue':sum,'PnL':sum})

    obj = {}
    obj['PnLByPlatformAndAccount'] = PnLByPlatformAndAccount
    obj['PnLByPlatform'] = PnLByPlatform
    return obj


# return the top holdings in the portfolio
def TopHoldings(ps):
    df = ps.copy()
    # need to convert all to HKD first
    for i in range(len(df)):
        row = df.iloc[i]
        if row.PlatformCurrency=='HKD':
            df.loc[i, 'CurrentValueHKD'] = df.loc[i, 'CurrentValue']
        else:
            df.loc[i, 'CurrentValueHKD'] = calc_fx.ConvertTo('HKD', df.loc[i, 'PlatformCurrency'], df.loc[i, 'CurrentValue'])
    df.loc[:,'PortfolioPct'] = df.loc[:,'CurrentValueHKD'] / df.CurrentValueHKD.sum()
    df = df.sort_values(['CurrentValueHKD'], ascending=False)[['Name','CurrentValueHKD','PortfolioPct']].head(10)
    df = df.reset_index(drop=True)
    return df


# Portfolio Summary, including uninvested cash balances
def GetPortfolioSummaryIncCash():
    cash = setup.GetBankAndCashBalances()
    ps_IncCash = ps.copy()
    for i in range(len(cash)):
        row = cash.iloc[i]
        current_value_in_HKD = calc_fx.ConvertTo('HKD', row.Currency, row.Balance)
        dic = {'Platform':'Cash',
               'Name':row.AccountName,
               'CurrentValue':row.Balance,
               'SecurityType':'FX & cash',
               'SecurityCcy':row.Currency,
               'CurrentValueInHKD':current_value_in_HKD,
               'Category':row.Category
               }
        ps_IncCash = ps_IncCash.append(dic, ignore_index=True)
    return ps_IncCash


# get calculations for other modules to use
ps = GetPortfolioSummary()
top_holdings = TopHoldings(ps)
pnl_unrealised = GetPnLUnrealised()
