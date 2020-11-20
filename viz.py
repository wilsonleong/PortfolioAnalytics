# -*- coding: utf-8 -*-
"""
PORTFOLIO TRACKER - VISUALISATION MODULE


Created on Sat Sep 26 13:43:39 2020

@author: Wilson Leong
"""


import pandas as pd
pd.options.display.float_format = '{:,.2f}'.format
import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib as mpl
import matplotlib.dates as mdates

from setup import *
from calc import *


def DisplaySummary():
    # get the calculations
    pnl_unrealised = GetPnLUnrealised()
    # print PnL by platform and Currency
    print ('\n*** PnL by Platform and Account ***')
    print (pnl_unrealised['PnLByPlatformAndAccount'])
    print ('\n*** %s ***' % datetime.datetime.today())
    print ('\n*** PnL by Platform ***')
    print (pnl_unrealised['PnLByPlatform'])

    # historical realised PnL
    print ('\n*** Realised PnL (not inc. platform fees) ***')
    pnl = GetHistoricalRealisedPnL()
#    print (pnl)
    
#    print ('\n *** Total Realised PnL ***')
    pnl = pnl.reset_index()
    print ('')
    print (pnl.groupby(['Platform','PlatformCurrency']).sum())


    # Portfolio composition report
    pcr = GetPortfolioComposition()
    pcr.to_csv('PortfolioSummaryInHKD.csv', index=False)
    pct = pcr.groupby('Category').agg({'CurrentValueInHKD':'sum'})
    pct.reset_index(inplace=True)
    total = pct.CurrentValueInHKD.sum()
    
    print ('\nTotal investments by category in HKD equivalent:')
    for i in range(len(pct)):
        row = pct.iloc[i]
        cat = row.Category
        cat_withspace = cat + ' ' * (15-len(cat))
        value = row.CurrentValueInHKD
        pc = '%s' % '{:,.2f}'.format(value / total * 100) + '%'
        print ('%s \t %s' % (cat_withspace, pc))
    print ('Total investments: %s (HKD equivalent)' % '{:,.0f}'.format(pct.CurrentValueInHKD.sum()))
    
    # print annualised returns on FSM HK & SG accounts
    ar_fsmhk = CalcModDietzReturn('FSM HK')
    ar_fsmsg = CalcModDietzReturn('FSM SG')
    ar_etf = CalcIRR(platform='FSM HK')
    ar_etf_1W = CalcIRR(platform='FSM HK', period='1W')
    ar_etf_1M = CalcIRR(platform='FSM HK', period='1M')
    ar_etf_3M = CalcIRR(platform='FSM HK', period='3M')
    ar_etf_6M = CalcIRR(platform='FSM HK', period='6M')
    ar_etf_1Y = CalcIRR(platform='FSM HK', period='1Y')
    ar_etf_3Y = CalcIRR(platform='FSM HK', period='3Y')
    print ('')
    print ('Annualised returns from inception (TWRR / Modified Dietz Return):')
    print ('> FSM HK: \t\t' + '{:,.2%}'.format(ar_fsmhk['AnnualisedReturn']))
    print ('> FSM SG: \t\t' + '{:,.2%}'.format(ar_fsmsg['AnnualisedReturn']))
    print ('')
    print ('Performance of US ETFs (MWRR / IRR):')
    print ('> Since Inception: \t\t' + '{:,.2%}'.format(ar_etf))
    print ('> 1W:              \t\t' + '{:,.2%}'.format(ar_etf_1W))
    print ('> 1M:              \t\t' + '{:,.2%}'.format(ar_etf_1M))
    print ('> 3M:              \t\t' + '{:,.2%}'.format(ar_etf_3M))
    print ('> 6M:              \t\t' + '{:,.2%}'.format(ar_etf_6M))
    print ('> 1Y:              \t\t' + '{:,.2%}'.format(ar_etf_1Y))
    print ('> 3Y:              \t\t' + '{:,.2%}'.format(ar_etf_3Y))
    print ('')

    # plot chart: portfolio composition
    pct['Percentage']=pct['CurrentValueInHKD']/pct.CurrentValueInHKD.sum()
    labels = list(pct.Category)
    #sizes = list(pct.CurrentValueInHKD)
    sizes = list(pct.Percentage)
    plt.rcdefaults()
    fig, ax1 = plt.subplots()
    y_pos = np.arange(len(labels))
    ax1.barh(y_pos, sizes)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_xlabel('Percentage of Portfolio')
    ax1.set_ylabel('Category')
    vals = ax1.get_xticks()
    ax1.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    title = 'Portfolio Composition as of %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')
    ax1.set(title=title)
    for index, value in enumerate(sizes):
        ax1.text(value, index, str('{:,.2%}'.format(value)), color='black', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.show()


    # fig, axs = plt.subplots(ncols=1, figsize=(10,5))
    # import seaborn as sns
    # sns.set(style='whitegrid')
    # ax = sns.barplot(y='Category',
    #                  x='Percentage',
    #                  data=pct[['Category','Percentage']],
    #                  orient='h',
    #                  hue=None
    #                  )
    # ax.set_title('Total investments by category in HKD equivalent')

    
    # # pcts as pie
    # labels = list(pct.Category)
    # sizes = list(pct.pct)
    # explode = [0.1] * len(labels)
    # fig1, ax1 = plt.subplots()
    # ax1.pie(sizes, labels=labels, autopct='%1.2f%%', explode=explode)
    # ax1.axis('equal')
    # ax1.set(title='Portfolio Composition (in HKD equivalent)')
    # plt.show()
    
    
    # # example pie chart
    # fig, ax = plt.subplots()
    
    # size = 0.3
    # vals = np.array([[60., 32.], [37., 40.], [29., 10.]])
    
    # cmap = plt.get_cmap("tab20c")
    # outer_colors = cmap(np.arange(3)*4)
    # inner_colors = cmap([1, 2, 5, 6, 9, 10])
    
    # ax.pie(vals.sum(axis=1), radius=1, colors=outer_colors,
    #        wedgeprops=dict(width=size, edgecolor='w'))
    
    # ax.pie(vals.flatten(), radius=1-size, colors=inner_colors,
    #        wedgeprops=dict(width=size, edgecolor='w'))
    
    # ax.set(aspect="equal", title='Portfolio Composition (in HKD equivalent)')
    # plt.show()



# plot line of costs for US ETF portfolio
def PlotCostvsVal():
    # collect the data
    hist_cost = CalcPortfolioHistoricalCost()
    hist_valuation = CalcPortfolioHistoricalValuation()
    df = hist_valuation.merge(hist_cost, how='left', on='Date')
    df = df.fillna(method='ffill')
    
    # # optional filter: start date
    # df = df[df.Date>=datetime.datetime(2020,9,3)]
    # df = df.reset_index(drop=True)
    
    # create the plots
    fig, ax = plt.subplots()    # can set dpi=150 or 200 for bigger image; figsize=(8,6)
    #ax.set_title('US ETF Portfolio: Investment Amount vs Current Value')
    
    title = 'US ETFs: Investment Cost vs Current Value as of %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')
    ax.set(title=title)
    
    ax.set_ylabel('Amount (HKD)')
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    
    # plot the cost
    x1 = df.Date
    y1 = df.AccumCostHKD
    #ax.plot(x1, y1, marker='.', linestyle='-')
    ax.plot(x1, y1, linestyle='-')
    
    # plot the valuation
    x2 = df.Date
    y2 = df.ValuationHKD
    ax.plot(x2, y2, linestyle='-', color='orange')
    
    # add annotation: 01 Sep 2020 transfer of XLE VWO from Singapore account
    x2_pos = x2[x2 == datetime.datetime(2020, 9, 1)].index
    ax.annotate('1 Sep 2020: Transfer in VWO and XLE',
                #(mdates.date2num(x[1]), y[1]),
                xy=('2020-09-01', y2.iloc[x2_pos]),
                xytext=(-200, 10),
                textcoords='offset points', 
                arrowprops=dict(arrowstyle='-|>')
                )
    
    # add annotation: 03 Sep 2020 built up position
    x2_pos = x2[x2 == datetime.datetime(2020, 9, 3)].index
    ax.annotate('3 Sep 2020: Built up positions',
                #(mdates.date2num(x[1]), y[1]),
                xy=('2020-09-03', y2.iloc[x2_pos]),
                xytext=(-200, 0),
                textcoords='offset points', 
                arrowprops=dict(arrowstyle='-|>')
                )
    
    # # add annotation: 10 Sep 2020 invest redundancy payout lumpsum
    # x2_pos = x2[x2 == datetime.datetime(2020, 9, 10)].index
    # ax.annotate('10 Sep 2020: Lumpsum from redundancy',
    #             #(mdates.date2num(x[1]), y[1]),
    #             xy=('2020-09-10', y2.iloc[x2_pos]),
    #             xytext=(-200, -30),
    #             textcoords='offset points', 
    #             arrowprops=dict(arrowstyle='-|>')
    #             )
    
    #fig.autofmt_xdate(rotation=45)
    plt.xticks(rotation=45, ha='right')
    plt.show()


def PlotTopHoldings():
    #variable with information: top_holdings
    labels = list(top_holdings.Name)
    #sizes = list(top_holdings.CurrentValueHKD)
    #sizes = list(top_holdings.PortfolioPct)
    sizes_pct = list(top_holdings.PortfolioPct)
    plt.rcdefaults()
    fig, ax1 = plt.subplots()
    y_pos = np.arange(len(labels))
    #ax1.barh(y_pos, sizes)
    ax1.barh(y_pos, sizes_pct)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    ax1.set_ylabel('Security')
    vals = ax1.get_xticks()
    ax1.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    #ax1.set_xticklabels(['{:,.0f}'.format(x) for x in vals])
    ax1.set_xlabel('Percentage of Portfolio')
    #ax1.set_xlabel('Current Value (HKD)')
    #plt.xticks(rotation=45, ha='right')
    title = 'Top Holdings as of %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')
    ax1.set(title=title)
    for index, value in enumerate(sizes_pct):
        ax1.text(value, index, str('{:,.2%}'.format(value)), color='black', fontweight='bold')
    plt.gca().invert_yaxis()
    plt.show()












