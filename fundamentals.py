# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 22:53:45 2021

@author: Wilson Leong
"""

import datetime
import pandas as pd
import numpy as np
import yfinance as yf

stocks = ['AAPL','AMD','AMZN','NVDA','PYPL','PLTR','TSLA','SNPS','LRCX']
'''
ADBE    underperformer
GOOG    underperformer
MSFT    underperformer
ADI     underperformer
TER     underperformer
MU      underperformer

'''

df = pd.DataFrame()
for i in stocks:
    print ('Getting data for %s' % i)
    tmp = yf.Ticker(i).get_info()
    df = df.append(pd.DataFrame([tmp]))

cols = ['symbol',
        'shortName',
        
        # 'open',
        # 'dayHigh',
        # 'dayLow',
        # 'previousClose',
        
        # 'fiftyTwoWeekHigh',
        # 'fiftyTwoWeekLow',
        # '52WeekChange',
        # 'SandP52WeekChange',
        
        # 'bid',
        # 'ask',
        
        # 'regularMarketPreviousClose',
        # 'regularMarketOpen',
        # 'regularMarketDayHigh',
        # 'regularMarketDayLow',
        # 'regularMarketPrice',
        
        'twoHundredDayAverage',
        
        # 'dividendYield',
        # 'trailingAnnualDividendYield',
        # 'fiveYearAvgDividendYield',
        
        #'trailingAnnualDividendRate',
        #'dividendRate',
        #'lastDividendValue',
        
        # 'payoutRatio',
        
        'beta',
        # 'beta3Year',
        'forwardPE',
        'trailingPE',
        
        'forwardEps',
        'trailingEps',
        
        'priceToBook',
        'pegRatio',
        
        #'bookValue',
        #'priceToSalesTrailing12Months',
        #'enterpriseToRevenue',
        'profitMargins'
        ]
df2 = df[cols]
df2 = df2.fillna(np.nan)
