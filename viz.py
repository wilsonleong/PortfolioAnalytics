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
    print ('Annualised returns from inception (time-weighted):')
    print ('> FSM HK: \t\t' + '{:,.2%}'.format(ar_fsmhk['AnnualisedReturn']))
    print ('> FSM SG: \t\t' + '{:,.2%}'.format(ar_fsmsg['AnnualisedReturn']))
    print ('')
    print ('*** BUGGED (need to include b/f) *** Performance of Yahoo Finance supported instruments (money-weighted):')
    print ('> YTD: \t\t' + '{:,.2%}'.format(ar_etf))
    print ('> 1W:  \t\t' + '{:,.2%}'.format(ar_etf_1W))
    print ('> 1M:  \t\t' + '{:,.2%}'.format(ar_etf_1M))
    print ('> 3M:  \t\t' + '{:,.2%}'.format(ar_etf_3M))
    print ('> 6M:  \t\t' + '{:,.2%}'.format(ar_etf_6M))
    print ('> 1Y:  \t\t' + '{:,.2%}'.format(ar_etf_1Y))
    print ('> 3Y:  \t\t' + '{:,.2%}'.format(ar_etf_3Y))
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


    # 2020-12-01: plot donut chart by category
    labels_with_pct = []
    for i in range(len(labels)):
        labels_with_pct.append(labels[i][1:] + ' (%s)' % '{:,.2%}'.format(sizes[i]))
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(aspect="equal"))
    wedges, texts = ax.pie(sizes, wedgeprops=dict(width=0.3), startangle=-40)
    #bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.5)
    kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})

        #ax.annotate(labels_with_pct[i], xy=(x, y), xytext=(1*np.sign(x), 1.1*y),
        #            horizontalalignment=horizontalalignment, **kw, fontsize=7)

    ax.set_title(title)
    #plt.legend(wedges, labels_with_pct, loc='center', bbox_to_anchor=(-0.1, 1.), fontsize=8)
    plt.legend(wedges, labels_with_pct, loc='center', fontsize=8)
    plt.show()
    
    PlotPortfolioComposition('SecurityCcy')
    PlotPortfolioComposition('SecurityType')


# 2020-12-02: plot donut chart by security currency
def PlotPortfolioComposition(by='SecurityCcy'):
    #by='SecurityType'
    if by=='SecurityCcy':
        title = 'Currency Exposure'
    elif by=='SecurityType':
        title = 'Asset Allocation'
    title = title + ' as of %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')
    
    pcr = GetPortfolioComposition()
    pct = pcr.groupby(by).agg({'CurrentValueInHKD':'sum'})
    pct.reset_index(inplace=True)
    total = pct.CurrentValueInHKD.sum()
    pct['Percentage'] = pct['CurrentValueInHKD']/pct.CurrentValueInHKD.sum()
    
    categories = pct[by]
    values = pct.Percentage
    categories_with_pct = []
    for i in range(len(categories)):
        categories_with_pct.append(categories[i] + ' (%s)' % '{:,.2%}'.format(values[i]))
    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw=dict(aspect="equal"))
    wedges, texts = ax.pie(values, wedgeprops=dict(width=0.3), startangle=-40)
    #bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.5)
    kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")
    for i, p in enumerate(wedges):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax.annotate(categories_with_pct[i], xy=(x, y), xytext=(1*np.sign(x), 1.1*y),horizontalalignment=horizontalalignment, **kw, fontsize=10)
    ax.set_title(title)
    #plt.legend(wedges, categories_with_pct, loc='center', fontsize=8)
    plt.show()

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
    title = 'Investment Cost vs Valuation as of %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')
    
    # add subtitle with return %
    ar_etf = CalcIRR(platform='FSM HK')
    subtitle = 'Performance YTD: %s' % '{0:.2%}'.format(ar_etf)
    
    fig.suptitle(title, fontsize=12)
    ax.set(title=subtitle)
    
    ax.set_ylabel('Amount (HKD)')
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    
    # plot the cost
    x1 = df.Date
    y1 = df.AccumCostHKD
    #ax.plot(x1, y1, marker='.', linestyle='-')
    ax.plot(x1, y1, linestyle='-', label='Investment cost')
    
    # plot the valuation
    x2 = df.Date
    y2 = df.ValuationHKD
    ax.plot(x2, y2, linestyle='-', color='orange', label='Valuation')
    
    # add legend
    ax.legend(frameon=False, loc='lower center', ncol=2)
    
    # add annotation: 01 Sep 2020 transfer of XLE VWO from Singapore account
    x2_pos = x2[x2 == datetime.datetime(2020, 9, 1)].index
    ax.annotate('Transfer in; built up positions',
                xy=('2020-09-01', y2.iloc[x2_pos]),
                xytext=(-200, 20),
                textcoords='offset points', color='gray',
                arrowprops=dict(arrowstyle='-|>', color='gray')
                )

    # add annotation: 24 Nov 2020 took profit from Airlines, reinvested in Tech
    x2_pos = x2[x2 == datetime.datetime(2020, 11, 24)].index
    ax.annotate('Took profit from Airlines, reinvested in Innovation',
                xy=('2020-11-24', y2.iloc[x2_pos]),
                xytext=(-275, 0),
                textcoords='offset points', color='gray',
                arrowprops=dict(arrowstyle='-|>', color='gray')
                )
    
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
