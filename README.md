# Consolidated Investment Portfolio Analytics

#PortfolioAnalytics #MarketData #FX #Equities #ETFs #MutualFunds
#Python #pandas #matplotlib #API 
#MongoDB #NoSQL 

## Problem statement
1. My investment accounts are scattered across different brokers in multiple currencies (GBP, SGD, HKD) due to my employment history.
2. PAD (Personal Account Dealing) reporting to my employer's Compliance department is manual and very time-consuming.
3. There is no holistic view of overall performance or composition of my portfolio to aid my investment decisions.

## Solution
I built my own portfolio analytics tool that consolidates all my transactions across countries and brokers, connects to Yahoo Finance via API for historical market data (NAV and FX rates), handles the FX conversions, and produces the following output:
1. consolidated investments with NAV, realised & unrealised PnL in a single currency
2. calculates IRR (internal rate of return) and benchmarks my portfolio returns vs S&P 500
3. calculates composition of portfolio by predefined categories, asset currency, and asset type
4. displays top holdings
5. plots evolution of investment cost vs valuation over time in a single currency
6. automated PAD reporting

## URLs
* Example output: https://github.com/wilsonleong/PortfolioAnalytics/wiki/Consolidated-Investment-Portfolio-Analytics
* Project kanban: https://github.com/wilsonleong/PortfolioAnalytics/projects/1

## Libraries used
* numpy
* scipy
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
* Historical NAV processing, cache on MongoDB

**calc_fx.py**
* FX conversions

**calc_val.py**
* Latest NAV processing (both Yahoo Finance API and manual input of unsupported mutual funds)

**calc_summary.py**
* Portfolio summary, inc. uninvested cash
* Realised and unrealised PnL

**calc_returns.py**
* Modified Dietz Return (time-weighted): suitable measurement for a fund manager (emphasizes on overall trading decisions)
* Internal Rate of Return (money-weighted): suitable measurement for an individual investor (emphasizes on timing of in/out flows)
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
