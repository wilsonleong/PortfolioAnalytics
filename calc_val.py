# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:48:17 2020

@author: Wilson Leong

This module does the following:
    - Yahoo Finance API (historical prices, FX rates)
    - Latest NAV

"""

import setup
import pandas as pd
import mdata
_last_nav_file = mdata.LastNavFilePath
import calc_fx



# Get latest NAV and last updated timestamp of existing holdings across all platforms, and cache on MongoDB
def ProcessLastNAV():
    # get all transactions from MongoDB
    tran = setup.GetAllTransactions()
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
    dfPlatforms['PlatformCcy'] = [setup.GetPlatformCurrency(x) for x in list(tran_summary.Platform.unique())]
    tran_summary = tran_summary.merge(dfPlatforms, how='left', left_on='Platform', right_on='Platform')
    secs = setup.GetSecurities()
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
            tran_summary.loc[i,'FXRate'] = calc_fx.GetFXRate(row.PlatformCcy, row.SecurityCcy)

    # format output columns
    tran_summary.rename(columns={'SecurityCcy':'SecurityCurrency'}, inplace=True)
    tran_summary.drop(columns=['PlatformCcy','NoOfUnits'], inplace=True)

    # save results on MongoDB
    db = setup.ConnectToMongoDB()
    LastNAV = db['LastNAV']
    LastNAV.delete_many({})
    LastNAV.insert_many(tran_summary[['BBGCode','LastNAV','SecurityCurrency','FXRate','LastUpdated']].to_dict('records'))
    #return tran_summary


# get the latest NAV cached on MongoDB
def GetLastNAV():
    db = setup.ConnectToMongoDB()
    LastNAV = db['LastNAV']
    LastNAV.find()
    df = pd.DataFrame(list(LastNAV.find()))
    df.drop(columns=['_id'], inplace=True)
    return df