# PortfolioAnalytics
Portfolio Analytics

*Problem statement*
Due to my work history in multiple locations, my investment accounts are scattered across different brokers in multiple currencies. As such, there are 2 pain points:
1) PAD (Personal Account Dealing) reporting to my employer's Compliance department is manual and very time-consuming.
2) There is no holistic view of overall performance or composition of my portfolio to aid my investment decisions.

*Solution*
I built my own portfolio analytics tool that consolidates all my transactions across countries and brokers, connects to Yahoo Finance API for historical market data (NAV and FX rates), handles the FX conversions, and produces the following output:
1) consolidated investments with NAV, realised & unrealised PnL in a single currency
2) calculates returns % (both TWRR and MWRR) including withdrawals and deposits
2) calculates composition of portfolio by predefined categories/static referential data
3) displays top holdings
5) plots evolution of investment cost vs valuation over time in a single currency
6) automated PAD reporting
