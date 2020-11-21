# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 15:21:16 2017
@author: Wilson

This module tracks the portfolio of my investments
Transactions stored on mongodb

Things to do before running the module:
    - MongoDB server started
    - update values in "fx" sheet on "My Portfolio" spreadsheet (manual from BBG)
    - update values in "FX" sheet on "Setup" spreadsheet (manual from BBG)

Note:
    Accounting for reinvestment:
    FSM SG considers reinvestment as 0 cost
    while I consider reinvestment as realised PnL and reinvested (cost increased)

Calculation of portfolio returns - Modified Dietz Return:
    https://einvestingforbeginners.com/calculating-portfolio-return-cfa-csmit/
    https://www.investopedia.com/ask/answers/062215/how-do-i-calculate-my-portfolios-investment-returns-and-performance.asp
    https://www.fool.com/about/how-to-calculate-investment-returns/
    https://corporatefinanceinstitute.com/resources/knowledge/trading-investing/modified-dietz-return/
    https://en.wikipedia.org/wiki/Modified_Dietz_method


TO DO LIST:

1) plot nicer charts showing composition (requires remapping of categories by region & sector)
    https://matplotlib.org/gallery/pie_and_polar_charts/nested_pie.html#sphx-glr-gallery-pie-and-polar-charts-nested-pie-py

2) add returns of YTD 1W 1M 3M 6M 1Y 3Y 5Y (requires valuation at each of these beginning dates)

3) add annualised returns for every year 2015, 2016, 2017, 2018, 2019, 2020 (requires valuation at start/end dates)

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
    
    # display visualisation
    viz.DisplaySummary()
    viz.PlotCostvsVal()
    viz.PlotTopHoldings()


if __name__ == "__main__":
    main()


##### END OF PROGRAMMING CODE #####

# End timer
timer_end = datetime.datetime.now()
TimeTaken = timer_end - timer_start
print ('\nModule "%s" finished running in %s.\n' % (os.path.basename(sys.argv[0]), TimeTaken))

# delete unused variables
del timer_start, timer_end, TimeTaken
