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
2. calculates my portfolio IRR (internal rate of return) and benchmarks it against S&P 500 index returns over different date ranges
3. displays consolidated investment cost vs valuation over time in a single currency
4. displays consolidated PnL over time, break down by dividends and trading PnL
5. displays composition of portfolio, asset allocation, currency exposure
6. displays top holdings, their individual performance; and top gainers & losers
7. automated PAD reporting

## URLs
* Example output: https://github.com/wilsonleong/PortfolioAnalytics/wiki/Consolidated-Investment-Portfolio-Analytics
* Project kanban board: https://github.com/wilsonleong/PortfolioAnalytics/projects/1

## Libraries used (Python 3.7.6 - 64 bits)
* numpy (1.19.1)
* scipy (1.5.2)
* pandas (1.0.0)
* datetime 
* dateutil (2.8.1)
* pymongo (3.10.1)
* matplotlib (3.3.1)
* yfinance (0.1.54)

## Modules

**main.py**
* Runs the main application

**setup.py**
* Connects to MongoDB
* Platform and securities referential
* Transactions processing
* Weighted average cost calculation (when selling)
* Handling of uninvested cash balances

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
* Adjustment: 1 fund -> allocation to 3 asset classes

**calc_returns.py**
* Modified Dietz Return (time-weighted): suitable measurement for a fund manager (emphasizes on overall trading decisions)
* Internal Rate of Return (money-weighted): suitable measurement for an individual investor (emphasizes on impacts of in/out flows)
* Yahoo Finance API (historical prices, FX rates)
* Historical cost vs valuation
* Calculates returns of benchmark (S&P 500)

**viz.py**
* Portfolio Performance (IRR) vs Benchmark (S&P 500)
* Cost vs Latest Valuation
* Realised PnL over time, breakdown by dividends & trading PnL
* Portfolio Composition
* Asset Allocation
* Currency Exposure
* Top Holdings
* Product Type Breakdown
* Holdings by Fund House

**pad.py**
* PAD reporting
* Transactions list
* Holdings summary (balance b/f and c/f)

**util.py**
* handling of date range periods
