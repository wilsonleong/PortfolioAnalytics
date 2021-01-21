# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:21:16 2017
@author: Wilson

This module tracks the portfolio of my investments
Transactions stored on mongodb

Note:
    Accounting for reinvestment:
    FSM considers reinvestment as 0 cost
    while I consider reinvestment as realised PnL and reinvested (cost increased)

---

https://towardsdatascience.com/30-python-best-practices-tips-and-tricks-caefb9f8c5f5

use emoji
pip3 install emoji

use data class
from dataclasses import dataclass
https://realpython.com/python-data-classes/

"""



# Start timer
import datetime
import os, sys
print ("\n" + datetime.datetime.now().strftime('%d %b %Y %H:%M:%S') + ' | Running "' + os.path.basename(sys.argv[0]) + '"')
timer_start = datetime.datetime.now()


##### START OF PROGRAMMING CODE #####
import setup

import mdata
import calc_summary
import calc_returns
import calc_val


def main():
    # 1) Setup: update platform and security referential, cash balances -> cache on DB
    setup.InitialSetup()  # initial setup and other hardcoded transactions (exc import from FSM SG)

    # 2) Setup: process new transactions
    setup.InsertHistTransactions(datetime.datetime(2021,1,1))

    # 3) Market data: collect the latest (including intra-day) NAV of supported stocks, ETFs and mutual funds with existing holdings -> cache on DB
    mdata.UpdateLastNAV()
    calc_val.ProcessLastNAV()

    # 4) Market data: collect the latest (including intra-day) FX rates -> cache on DB
    setup.UpdateLatestFXrates()
    setup.UpdateLatestBullionRates()

    # 5) Market data: collect historical EOD USDHKD exchange rates -> cache on DB
    mdata.ProcessHistoricalUSDHKD()

    # 6) Market data: collect historical EOD market data of supported stocks, ETFs and mutual funds including those already sold -> cache on DB
    mdata.ProcessHistoricalMarketData()
    mdata.ProcessHistoricalSPX()

    # 7) Calculations: compute portfolio summary and IRR%s -> cache on DB
    calc_summary.CalcPortfolioSummaryAndCacheOnDB()
    calc_returns.CalcIRRAndCacheOnDB()
    calc_summary.StorePortfolioSummaryHistoryOnDB()     # inc. IRR%


def display_viz():
    # 8) Visualisations: load cached data from DB and plot various charts
    import viz
    
    # text display in console
    viz.DisplayPnL()
    viz.DisplayPortfolioSummary()

    # charts
    viz.DisplayReturnPct()
    viz.PlotLeadersAndLaggers()
    viz.PlotPerformanceOfHoldings(period='3M')
    viz.PlotCostvsVal(period='6M')
    viz.PlotRealisedPnLOverTime(period='6M')
    viz.PlotPortfolioComposition()
    viz.PlotAssetAllocationCurrencyExposure()
    viz.PlotTopHoldings()
    viz.PlotPortfolioCompositionBy(by='SecurityType',inc_cash=True)
    viz.PlotPortfolioCompositionBy(by='FundHouse')
    viz.PlotHistoricalSnapshot(period='6M')
    


if __name__ == "__main__":
    main()
    #display_viz()


##### END OF PROGRAMMING CODE #####

# End timer
timer_end = datetime.datetime.now()
TimeTaken = timer_end - timer_start
print ('\nModule "%s" finished running in %s.\n' % (os.path.basename(sys.argv[0]), TimeTaken))

# delete unused variables
del timer_start, timer_end, TimeTaken
