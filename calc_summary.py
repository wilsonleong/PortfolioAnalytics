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
import mdata
_output_dir = r'D:\Wilson\Documents\Personal Documents\Investments\PortfolioTracker\output'


# # store a copy of the output on MongoDB for future reference
# def UploadLatestPortfolioSummary(ps):
#     print ('\nUpdating Portfolio Summary on MongoDB...')
#     db = ConnectToMongoDB()
#     coll = db['LatestPortfolioSummary']
#     coll.delete_many({})
#     coll.insert_many(ps.to_dict('records'))
#     print ('(update completed)')


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
    summary = tn.groupby(['Platform','Name','FundHouse','AssetClass','BBGCode','BBGPriceMultiplier','Currency']).agg(agg)
    summary.reset_index(inplace=True)
    summary.rename(columns={'Currency':'SecurityCurrency'},inplace=True)

    # enrich with platforms
    db = setup.ConnectToMongoDB()
    platforms = pd.DataFrame(list(db.Platform.find()))
    summary = pd.merge(summary, platforms, how='left', left_on='Platform', right_on='PlatformName')
    summary.drop(['PlatformName','_id','SecurityCurrency'], axis=1, inplace=True)

    # enrich transactions with the latest price (ARKG has 2 FX rates USDHKD USDSGD that can cause duplicates)
    lastnav = calc_val.GetLastNAV()
    lastnav = lastnav.groupby(['BBGCode','LastNAV','SecurityCurrency']).agg({'LastUpdated':'min'})
    lastnav.reset_index(inplace=True)
    
    ### bug fixed 26 Dec 2020: left join caused duplicates
    summary = summary.merge(lastnav[['BBGCode','LastNAV','LastUpdated','SecurityCurrency']], how='left', left_on='BBGCode', right_on='BBGCode')
    
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
    ps = summary.groupby(['Platform','PlatformCurrency','FundHouse','AssetClass','Name','BBGCode','LastNAV','LastUpdated']).agg(agg2)
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

    # calculate Cost and PnL in HKD
    ps.loc[:,'CostInHKD'] = ps.loc[:,'CurrentValueInHKD']/ps.loc[:,'CurrentValue'] * ps.loc[:,'CostInPlatformCcy']
    ps.loc[:,'PnLInHKD'] = ps.loc[:,'CurrentValueInHKD']/ps.loc[:,'CurrentValue'] * ps.loc[:,'PnL']

    # 22 Dec 2020: add SecCcy to HKD rate, add WA cost in Security Ccy
    for i in [x for x in ps.index]:
        row = ps.loc[i]
        ps.loc[i,'WAC'] = setup.GetWeightedAvgCostPerUnitInSecCcy(row.BBGCode, row.Platform)
    
    # total PnL = realised + unrealised
    # (should I add or not? TO BE DECIDED)

    # special treatment to breakdown Allianz Income & Growth funds
    # divide by 3 separate rows and allocate different asset classes
    allianz_bbgcodes = ['ALIGH2S LX','ALLGAME LX']
    allianz_allocations = [{'Equity': 0.33},
                           {'Credit': 0.33},
                           {'Convertibles': 0.34}
                           ]
    # generate the new rows based on allocations
    dfAllianz = ps[ps.BBGCode.isin(allianz_bbgcodes)].copy()
    dfAllianzNew = pd.DataFrame(columns=dfAllianz.columns)
    for i in range(len(dfAllianz)):
        row = dfAllianz.iloc[i]
        for j in range(len(allianz_allocations)):
            new_row = row.copy()
            new_row['AssetClass'] = list(allianz_allocations[j].keys())[0]
            new_row['NoOfUnits'] = row.NoOfUnits * list(allianz_allocations[j].values())[0]
            new_row['CostInPlatformCcy'] = row.CostInPlatformCcy * list(allianz_allocations[j].values())[0]
            new_row['CurrentValue'] = row.CurrentValue * list(allianz_allocations[j].values())[0]
            new_row['PnL'] = row.PnL * list(allianz_allocations[j].values())[0]
            new_row['RealisedPnL'] = row.RealisedPnL * list(allianz_allocations[j].values())[0]
            new_row['CurrentValueInHKD'] = row.CurrentValueInHKD * list(allianz_allocations[j].values())[0]
            dfAllianzNew = dfAllianzNew.append(new_row)
    # replace the original rows with the new rows
    ps2 = ps[~ps.BBGCode.isin(allianz_bbgcodes)].copy()
    ps2 = ps2.append(dfAllianzNew)
    
    # can't assign Portfolio % when Allianz is broken down into separate asset classes
    ps.loc[:,'PortfolioPct'] = ps.loc[:,'CurrentValueInHKD'] / ps.CurrentValueInHKD.sum()

    # remove rows with 0 holdings
    ps = ps[ps.NoOfUnits!=0]
    ps2 = ps2[ps2.NoOfUnits!=0]

    PortfolioSummary = {'Original':ps,
                        'Adjusted':ps2}

    return PortfolioSummary


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
    ps = GetPortfolioSummaryFromDB(summary_type='Original')
    ps_active = ps[ps.CurrentValue!=0]
    PnLByPlatformAndAccount = ps_active.groupby(['Platform','Name','PlatformCurrency']).agg({'CurrentValue':sum,'PnL':sum})
    PnLByPlatform = ps_active.groupby(['PlatformCurrency','Platform']).agg({'CostInPlatformCcy':sum,'CurrentValue':sum,'PnL':sum})

    obj = {}
    obj['PnLByPlatformAndAccount'] = PnLByPlatformAndAccount
    obj['PnLByPlatform'] = PnLByPlatform
    return obj


# return the top holdings in the portfolio
def TopHoldings():
    df = GetPortfolioSummaryFromDB(summary_type='Original')
    # group by bbgcode (because same ETF can be held on different platforms)
    g = df.groupby(['BBGCode','Name','Category']).agg({'CurrentValueInHKD':'sum'})
    g.reset_index(inplace=True)
    g.loc[:,'PortfolioPct'] = g.loc[:,'CurrentValueInHKD'] / g.CurrentValueInHKD.sum()
    g = g.sort_values(['CurrentValueInHKD'], ascending=False)[['BBGCode','Name','Category','CurrentValueInHKD','PortfolioPct']].head(10)
    g = g.reset_index(drop=True)
    return g


# Portfolio Summary, including uninvested cash balances
def GetPortfolioSummaryIncCash():
    cash = setup.GetBankAndCashBalances()
    ps = GetPortfolioSummary()
    ps_adjusted = ps['Adjusted']
    ps_IncCash = ps_adjusted.copy()
    for i in range(len(cash)):
        row = cash.iloc[i]
        current_value_in_HKD = calc_fx.ConvertTo('HKD', row.Currency, row.Balance)
        dic = {'Platform':'Cash',
               'AssetClass':row.Category[1:],
               'Name':row.AccountName,
               'CurrentValue':row.Balance,
               'SecurityType':row.Category[1:],
               'SecurityCcy':row.Currency,
               'CurrentValueInHKD':current_value_in_HKD,
               'Category':row.Category
               }
        ps_IncCash = ps_IncCash.append(dic, ignore_index=True)
    ps_IncCash.loc[:,'PortfolioPct'] = ps_IncCash.loc[:,'CurrentValueInHKD'] / ps_IncCash.CurrentValueInHKD.sum()
    return ps_IncCash


# calculate portfolio summary and cache on DB
def CalcPortfolioSummaryAndCacheOnDB():
    print ('\nComputing portfolio summary...')
    # DB
    db = setup.ConnectToMongoDB()
    coll = db['PortfolioSummary']
    
    # Get calculations
    ps = GetPortfolioSummary()
    ps_original = ps['Original']
    ps_adjusted = ps['Adjusted']
    ps_adjustedIncCash = GetPortfolioSummaryIncCash() # inc cash is adjusted
    
    # fill blanks
    ps_adjusted['PortfolioPct'] = None
    
    # add summary type
    ps_original['SummaryType'] = 'Original'
    ps_adjusted['SummaryType'] = 'Adjusted'
    ps_adjustedIncCash['SummaryType'] = 'AdjustedIncCash'

    # cache on DB
    coll.delete_many({'SummaryType':'Original'})
    coll.insert_many(ps_original.to_dict('records'))    
    
    coll.delete_many({'SummaryType':'Adjusted'})
    coll.insert_many(ps_adjusted.to_dict('records'))    
    
    coll.delete_many({'SummaryType':'AdjustedIncCash'})
    coll.insert_many(ps_adjustedIncCash.to_dict('records')) 
    print ('(updated portfolio summary on mongodb)')
    
    # output to CSV file
    ps_original = ps_original[['Platform','Name','BBGCode','WAC','LastNAV','LastUpdated','CostInHKD','CurrentValueInHKD','PnLInHKD','PnLPct','PortfolioPct']].copy()
    ps_original.rename(columns={'LastNAV':'Last NAV',
                                'LastUpdated':'Last Updated',
                                'WAC':'WA cost',
                                'CostInHKD':'Cost (HKD)',
                                'CurrentValueInHKD':'Current Value (HKD)',
                                'PnLInHKD':'PnL (HKD)',
                                'PnLPct':'PnL (%)',
                                'PortfolioPct':'% of Ptf'
                                }, inplace=True)
    ps_original.to_csv(_output_dir + r'\ps_original.csv', index=False)
    print ('(exported portfolio summary as CSV)')


# get portfolio summary from cache (DB)
def GetPortfolioSummaryFromDB(summary_type='Original'):
    db = setup.ConnectToMongoDB()
    coll = db['PortfolioSummary']
    
    df = pd.DataFrame(list(coll.find({
        'SummaryType':summary_type
        })))
    df.drop(columns=['_id'], inplace=True)
    return df
