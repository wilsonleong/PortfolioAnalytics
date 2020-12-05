# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:21:16 2017
@author: Wilson

This module tracks the portfolio of my investments
Transactions stored on mongodb

Note:
    Accounting for reinvestment:
    FSM SG considers reinvestment as 0 cost
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
import viz
import mdata


def main():
    # refresh market data
    mdata.UpdateLastNAV()
    
    # update latest FX
    setup.UpdateLatestFXrates()
    
    # one-off first time setup
    setup.InitialSetup()  # initial setup and other hardcoded transactions (exc import from FSM SG)
    #setup.InsertHistTransactions()
    
    # process historical market data
    #mdata.ProcessHistoricalMarketData()
    
    # display visualisation
    viz.DisplaySummary()
    viz.DisplayReturnPct()
    viz.PlotPortfolioComposition()
    viz.PlotCostvsVal(period='6M', platform='FSM HK')
    #viz.PlotCostvsVal(period='6M', platform='FSM SG') # BUGGED - valuation still including XLE VWO in HK acc
    viz.PlotTopHoldings()
    viz.PlotCurrecnyExposureAssetAllocation()
    viz.PlotRealisedPnLOverTime(period='1Y')


if __name__ == "__main__":
    main()


##### END OF PROGRAMMING CODE #####

# End timer
timer_end = datetime.datetime.now()
TimeTaken = timer_end - timer_start
print ('\nModule "%s" finished running in %s.\n' % (os.path.basename(sys.argv[0]), TimeTaken))

# delete unused variables
del timer_start, timer_end, TimeTaken
