# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 14:36:37 2020

@author: Wilson Leong

This module does the following:
    - Get FX rate using Yahoo Finance API (to be called during initiation only)
    - Get FX rate from cached copy on MongoDB
    - Converts XXX to YYY (source ccy, target ccy, amount in source ccy)

"""

_special_ccys = ['AUD','NZD','EUR','GBP']    # those ccys that don't use USD as base ccy

import mdata
import setup
import pandas as pd



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


# function to return the conversion rate (SGD to HKD=5.7922, HKD to SGD=0.1726)
# Source: Yahoo Finance API (live rate) - to be called only during initialisation
def ConvertFX(ccy_source, ccy_target):
    # usage: ccy_target = returned fx rate * ccy_target
    #ccy_source='EUR'
    #ccy_target='SGD'
    #ccy_source, ccy_target = 'MOP','HKD'
    #ccy_source, ccy_target = 'GBP','HKD'

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


# function to look up MongoDB for exchange rate
# Source: MongoDB (cached)
def GetFXRate(target_ccy, original_ccy):
    #target_ccy, original_ccy ='HKD','GBP'
    #target_ccy, original_ccy ='HKD','AUD'
    #target_ccy, original_ccy ='HKD','MOP'
    if target_ccy==original_ccy:
        fxrate = 1
    else:
        ccypair = original_ccy + target_ccy
        # use exchange rate in LastNAV
        db = setup.ConnectToMongoDB()
        coll = db['FX']
        df = pd.DataFrame(list(coll.find()))
        
        # check if there is data, if not try inverted
        if len(df[df.Ccypair==ccypair]) > 0:
            row = df[df.Ccypair==ccypair].iloc[0]
            fxrate = row.PX_LAST
        else:
            ccypair = ccypair[-3:] + ccypair[:3]
            
            # in case ccypair not cached, get live rate from Yahoo Finance API
            if len(df[df.Ccypair==ccypair])==0:
                fxrate = ConvertFX(original_ccy, target_ccy)
            else:
                row = df[df.Ccypair==ccypair].iloc[0]
                fxrate = 1/row.PX_LAST

        if original_ccy=='JPY':
            fxrate = 1/fxrate
    return fxrate


def ConvertTo(target_ccy, original_ccy, original_amount):
    #target_ccy, original_ccy = 'HKD', 'SGD'
    #target_ccy, original_ccy = 'HKD', 'GBP'
    #target_ccy, original_ccy = 'HKD', 'MOP'
    #original_amount = 100
    rate = GetFXRate(target_ccy, original_ccy)
    target_ccy_amount = original_amount * rate
    return target_ccy_amount
#ConvertTo('HKD','SGD',1000)

