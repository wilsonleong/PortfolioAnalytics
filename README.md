# Consolidated Investment Portfolio Analytics

#Python #pandas #matplotlib #API #MongoDB #NoSQL #FX #PortfolioAnalytics #MarketData

## Problem statement
1. My investment accounts are scattered across different brokers in multiple currencies.
2. PAD (Personal Account Dealing) reporting to my employer's Compliance department is manual and very time-consuming.
3. There is no holistic view of overall performance or composition of my portfolio to aid my investment decisions.

## Solution
I built my own portfolio analytics tool that consolidates all my transactions across countries and brokers, connects to Yahoo Finance API for historical market data (NAV and FX rates), handles the FX conversions, and produces the following output:
1. consolidated investments with NAV, realised & unrealised PnL in a single currency
2. calculates returns % (both TWRR and MWRR) including withdrawals and deposits
3. calculates composition of portfolio by predefined categories/static referential data
4. displays top holdings
5. plots evolution of investment cost vs valuation over time in a single currency
6. automated PAD reporting

## Libraries used
* numpy
* pandas
* datetime
* dateutil
* pymongo
* matplotlib
* yfinance

## Modules

**main.py**
* Runs the main application

**setup.py**
* Connects to MongoDB
* Platform and securities referential
* Transactions processing
* Weighted average cost calculation (when selling)

**mdata.py**
* Yahoo Finance API
* Get historical NAV
* Get historical FX rates

**calc.py**
* FX conversions
* Latest NAV
* Portfolio summary
* Realised and unrealised PnL
* TWRR: Modified Dietz Return
* MWRR: Internal Rate of Return
* Yahoo Finance API (historical prices, FX rates)
* Historical cost vs valuation

**viz.py**
* Displays summary (PnL, returns %)
* Historical cost vs valuation (supported instruments only: stocks, ETFs, some mutual funds)
* Portfolio composition
* Top holdings

**pad.py**
* PAD reporting
* Transactions list
* Holdings summary (balance b/f and c/f)
