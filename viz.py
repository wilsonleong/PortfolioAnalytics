# -*- coding: utf-8 -*-
"""
PORTFOLIO TRACKER - VISUALISATION MODULE


Created on Sat Sep 26 13:43:39 2020

@author: Wilson Leong


reference viz: https://wyn.grapecity.com/blogs/how-to-optimize-your-stock-market-and-sector-performance-dashboard


"""


import pandas as pd
import numpy as np
pd.options.display.float_format = '{:,.2f}'.format
import datetime
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.colors as mcolors

import setup
import calc_summary
import calc_returns
import calc_fx
import util
import mdata
_output_dir = r'D:\Wilson\Documents\Personal Documents\Investments\PortfolioTracker\sample screenshots'


# displays a summary in the console (text)
def DisplayPnL():
    # get the calculations
    pnl_unrealised = calc_summary.GetPnLUnrealised()
    # print PnL by platform and Currency
    print ('\n*** PnL by Platform and Account ***')
    print (pnl_unrealised['PnLByPlatformAndAccount'])
    print ('\n*** %s ***' % datetime.datetime.today())
    print ('\n*** PnL by Platform ***')
    print (pnl_unrealised['PnLByPlatform'])

    # historical realised PnL
    print ('\n*** Realised PnL (not inc. platform fees) ***')
    pnl_obj = calc_summary.GetHistoricalRealisedPnL()
    pnl = pnl_obj[1]
    
    pnl = pnl.reset_index()
    print ('')
    print (pnl.groupby(['Platform','PlatformCurrency']).sum())


# display portfolio summary
def DisplayPortfolioSummary():
    # Portfolio composition report
    pcr = calc_summary.GetPortfolioSummaryFromDB(summary_type='Original')
    pct = pcr.groupby('Category').agg({'CurrentValueInHKD':'sum'})
    pct.reset_index(inplace=True)
    total = pct.CurrentValueInHKD.sum()
    total2 = calc_fx.ConvertTo('USD','HKD',total)

    print ('\nTotal investments (exc. cash) by category in HKD equivalent:')
    for i in range(len(pct)):
        row = pct.iloc[i]
        cat = row.Category
        cat_withspace = cat + ' ' * (15-len(cat))
        value = row.CurrentValueInHKD
        pc = '%s' % '{:,.2f}'.format(value / total * 100) + '%'
        print ('%s \t %s' % (cat_withspace, pc))
    print ('Total investments: %s HKD | %s USD' % ('{:,.0f}'.format(total), 
                                                   '{:,.0f}'.format(total2)))

    # calculate the equivalent in other currencies
    ps_inc_cash = calc_summary.GetPortfolioSummaryFromDB(summary_type='AdjustedIncCash')
    total_inc_cash = ps_inc_cash.CurrentValueInHKD.sum()
    total_USD = calc_fx.ConvertTo('USD','HKD',total_inc_cash)
    total_EUR = calc_fx.ConvertTo('EUR','HKD',total_inc_cash)
    total_GBP = calc_fx.ConvertTo('GBP','HKD',total_inc_cash)
    total_SGD = calc_fx.ConvertTo('SGD','HKD',total_inc_cash)
    
    print ('\nTotal portfolio value including cash:')
    print ('>> %s HKD' % '{:,.0f}'.format(total_inc_cash))
    print ('or %s USD' % '{:,.0f}'.format(total_USD))
    print ('or %s EUR' % '{:,.0f}'.format(total_EUR))
    print ('or %s GBP' % '{:,.0f}'.format(total_GBP))
    print ('or %s SGD' % '{:,.0f}'.format(total_SGD))
    
    # # print annualised returns on FSM HK & SG accounts
    # ar_fsmhk = calc_returns.CalcModDietzReturn('FSM HK')
    # ar_fsmsg = calc_returns.CalcModDietzReturn('FSM SG')
    # print ('')
    # print ('Annualised returns from inception (time-weighted):')
    # print ('> FSM HK: \t\t' + '{:,.2%}'.format(ar_fsmhk['AnnualisedReturn']))
    # print ('> FSM SG: \t\t' + '{:,.2%}'.format(ar_fsmsg['AnnualisedReturn']))
    # print ('')


# display return %
def DisplayReturnPct():    
    # IRR
    date_ranges = util.date_ranges
    
    # # get the IRR for the date ranges
    # returns = {}
    # for i in range(len(date_ranges)):
    #     returns[date_ranges[i]] = calc_returns.CalcIRR(period=date_ranges[i])

    # # get the IRR % only
    # returns_irr = {}
    # for i in range(len(date_ranges)):
    #     returns_irr[date_ranges[i]] = returns[date_ranges[i]]['IRR']
    
    # get IRR from cache (DB)
    returns_irr = calc_returns.GetIRRFromDB()
    returns_irr = returns_irr[(returns_irr.Platform.isnull()) & (returns_irr.BBGCode.isnull())]
    #returns_irr = returns_irr[['Period','IRR']]
    #returns_irr.set_index('Period', inplace=True)
    dic = {}
    for i in range(len(returns_irr)):
        dic[returns_irr.loc[i,'Period']] = returns_irr.loc[i,'IRR']
    returns_irr = dic
    
    supported_instruments = setup.GetListOfSupportedInstruments()
    ps = calc_summary.GetPortfolioSummaryFromDB()
    ps_supported_instruments = ps[ps.BBGCode.isin(supported_instruments)]
    print ('\nPerformance of Yahoo Finance supported instruments (%s of total):' % '{:,.2%}'.format(ps_supported_instruments.CurrentValueInHKD.sum()/ps.CurrentValueInHKD.sum()))
    for i in range(len(returns_irr)):
        print ('> %s: \t\t' % list(returns_irr.keys())[i] + '{:,.2%}'.format(list(returns_irr.values())[i]))
        #row = returns_irr.iloc[i]
        #print ('> %s: \t\t' % row.Period + '{:,.2%}'.format(row.IRR))
    print ('Total value of supported instruments: %s HKD' % ('{:,.0f}'.format(ps_supported_instruments.CurrentValueInHKD.sum())))
    print ('')

    # plot the returns on a bar chart
    # prepare the data
    date_ranges = np.array(date_ranges)
    values = np.array(list(returns_irr.values()))

    # get SPX returns as benchmark
    spx = calc_returns.GetSPXReturns()
    #spx_returns = np.array(spx.Returns)
    
    has_negative_values = False
    if len(spx[spx.AnnualisedReturn < 0]) > 0:
        has_negative_values = True
    if np.sum(values < 0):
        has_negative_values = True

    # compare porfolio returns vs SPX (YTD)
    YTD_spx_diff = returns_irr['YTD'] - spx.loc['YTD','AnnualisedReturn']
    if YTD_spx_diff >= 0:
        comp_label = 'Outperformed'
        annotate_colour = 'tab:green'
    else:
        comp_label = 'Underperformed'
        annotate_colour = 'tab:red'
    comp_full_text = '%s\nSPX\nby %s' % (comp_label, '{:.2%}'.format(YTD_spx_diff))
    #comp_full_text = '%s SPX by %s bps' % (comp_label, int((YTD_spx_diff*10000)))

    # compare porfolio returns vs SPX (5Y)
    spx_diff_5Y = returns_irr['5Y'] - spx.loc['5Y','AnnualisedReturn']
    if spx_diff_5Y >= 0:
        comp_label2 = 'Outperformed'
        annotate_colour2 = 'tab:green'
    else:
        comp_label2 = 'Underperformed'
        annotate_colour2 = 'tab:red'
    comp_full_text2 = '%s\nSPX\nby %s' % (comp_label2, '{:.2%}'.format(spx_diff_5Y))


    # plot the chart
    fig, ax = plt.subplots()
    return_positive = values > 0
    return_negative = values < 0
    # plot the date ranges with empty values first (to set the order)
    ax.bar(date_ranges, [0]*len(date_ranges))
    # then plot postive first, and then negative
    ax.bar(date_ranges[return_positive], values[return_positive], color='tab:green')
    ax.bar(date_ranges[return_negative], values[return_negative], color='tab:red')

    # add SPX as benchmark
    ax.plot(date_ranges, list(spx.AnnualisedReturn), color='tab:blue', 
            marker='_', markeredgewidth=2, markersize=20,
            label='S&P500 Index', lw=0)

    # add annotate text (YTD)
    ax.annotate(comp_full_text,
            xy=(list(date_ranges).index('YTD'), returns_irr['YTD']),
            #xytext=(-25, 50),
            xytext=(0, 25),
            textcoords='offset points', 
            color=annotate_colour,
            #weight='bold',
            fontsize=7, ha='center',
            arrowprops=dict(arrowstyle='-|>', color=annotate_colour)
            )

    # add annotate text (5Y)
    ax.annotate(comp_full_text2,
            xy=(list(date_ranges).index('5Y'), returns_irr['5Y']),
            xytext=(0, 25),
            textcoords='offset points', 
            color=annotate_colour2,
            fontsize=7, ha='center',
            arrowprops=dict(arrowstyle='-|>', color=annotate_colour2)
            )

    # finalise chart
    ax.set_ylabel('Performance % (>1Y annualised)')
    ax.set_xlabel('Date Range')
    #ax.set_ylabel('Annualised Return % for date range above 1Y')
    title = 'Portfolio Returns vs S&P 500 - %s' % (datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S'))
    ax.set_title(title)
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    # this is bugged when there is negative value
    # for ymaj in ax.yaxis.get_majorticklocs():
    #     ax.axhline(y=ymaj, ls='-', lw=0.25, color='black')
    if has_negative_values:
        ax.axhline(y=0, ls='-', lw=0.25, color='black')

    # save output as PNG
    output_filename = 'PortfolioPerformance.png'
    output_fullpath = '%s/%s' % (_output_dir, output_filename)
    ax.legend(loc='upper right', bbox_to_anchor=(1,-0.1))
    fig.savefig(output_fullpath, format='png', dpi=150, bbox_inches='tight')
    plt.show()


# plot chart: portfolio composition
def PlotPortfolioComposition():
    # prepare data
    pcr = calc_summary.GetPortfolioSummaryFromDB(summary_type='Original')
    pct = pcr.groupby('Category').agg({'CurrentValueInHKD':'sum'})
    pct.reset_index(inplace=True)

    # plot bar chart
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
    title = 'Portfolio Composition - %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')
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
    wedges, texts = ax.pie(sizes, wedgeprops=dict(width=0.3), startangle=-60)
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
        ax.annotate('{:,.2%}'.format(sizes[i]),
                    xy=(x, y), 
                    xytext=(1*np.sign(x), 1.1*y),
                    horizontalalignment=horizontalalignment, 
                    **kw, 
                    fontsize=7)
    ax.set_title(title)
    plt.legend(wedges,
               labels_with_pct,
               loc='upper center',
               bbox_to_anchor=(0.5, -0.05),
               ncol=3,
               fontsize=8)
    #plt.legend(wedges, labels_with_pct, loc='center', fontsize=8)

    # save output as PNG
    output_filename = 'PortfolioComposition.png'
    output_fullpath = '%s/%s' % (_output_dir, output_filename)
    fig.savefig(output_fullpath, format='png', dpi=150, bbox_inches='tight')
    plt.show()
    
    
# 2020-12-02: plot donut chart by security currency
def PlotAssetAllocationCurrencyExposure():
    # get the data
    #pcr = calc_summary.GetPortfolioSummaryIncCash()
    pcr = calc_summary.GetPortfolioSummaryFromDB(summary_type='AdjustedIncCash')
    
    # prepare the titles
    by1 = 'AssetClass'
    title1 = 'Asset Allocation (inc. FX & cash)'
    by2 = 'SecurityCcy'
    title2 = 'Currency Exposure (inc. FX & cash)'
    title1 = title1 + ' - %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')
    title2 = title2 + ' - %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')

    # plot the figure & axis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), subplot_kw=dict(aspect="equal"))

    # Asset Allocation
    pct1 = pcr.groupby(by1).agg({'CurrentValueInHKD':'sum'})
    pct1.reset_index(inplace=True)
    total1 = pct1.CurrentValueInHKD.sum()
    pct1['Percentage'] = pct1['CurrentValueInHKD']/pct1.CurrentValueInHKD.sum()
    
    categories1 = pct1[by1]
    values1 = pct1.Percentage
    categories_with_pct1 = []
    for i in range(len(categories1)):
        categories_with_pct1.append(categories1[i] + ' (%s)' % '{:,.2%}'.format(values1[i]))
        
    wedges1, texts1 = ax1.pie(values1, wedgeprops=dict(width=0.3), startangle=180)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.5)
    kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")
    for i, p in enumerate(wedges1):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax1.annotate(categories_with_pct1[i], xy=(x, y), 
                     xytext=(1*np.sign(x), 1.1*y),
                     horizontalalignment=horizontalalignment, 
                     **kw, fontsize=10)
    ax1.set_title(title1)
    
    # Currency Exposure
    pct2 = pcr.groupby(by2).agg({'CurrentValueInHKD':'sum'})
    pct2.reset_index(inplace=True)
    total2 = pct2.CurrentValueInHKD.sum()
    pct2['Percentage'] = pct2['CurrentValueInHKD']/pct2.CurrentValueInHKD.sum()
    
    categories2 = pct2[by2]
    values2 = pct2.Percentage
    categories_with_pct2 = []
    for i in range(len(categories2)):
        categories_with_pct2.append(categories2[i] + ' (%s)' % '{:,.2%}'.format(values2[i]))

    
    
    wedges2, texts2 = ax2.pie(values2, wedgeprops=dict(width=0.3), startangle=20)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.5)
    kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")
    for i, p in enumerate(wedges2):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax2.annotate(categories_with_pct2[i], xy=(x, y), 
                     xytext=(1*np.sign(x), 1.1*y),
                     horizontalalignment=horizontalalignment, 
                     **kw, fontsize=10)
    ax2.set_title(title2)
    
    # save output as PNG
    output_filename = 'AssetAllocationAndCurrencyExposure.png'
    output_fullpath = '%s/%s' % (_output_dir, output_filename)
    fig.savefig(output_fullpath, format='png', dpi=150, bbox_inches='tight')
    plt.show()


# # 2020-12-02: plot donut chart for portfolio composition
def PlotPortfolioCompositionBy(by='SecurityType'):
    #by='SecurityType'
    if by=='SecurityCcy': # NOT IN USE
        title = 'Currency Exposure'
    elif by=='SecurityType':
        title = 'Product Type Breakdown'
    elif by=='AssetClass': # NOT IN USE
        title = 'Asset Class Breakdown'
    if by=='FundHouse':
        title = 'Holdings by Fund House'
    title = title + ' - %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')
    
    pcr = calc_summary.GetPortfolioSummaryFromDB(summary_type='Adjusted')
    #pcr = calc_summary.GetPortfolioSummary()
    #pcr = pcr['Adjusted']
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
    wedges, texts = ax.pie(values, wedgeprops=dict(width=0.3), startangle=0)
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

    # save output as PNG
    output_filename = 'PortfolioCompositionBy' + by + '.png'
    output_fullpath = '%s/%s' % (_output_dir, output_filename)
    fig.savefig(output_fullpath, format='png', dpi=150, bbox_inches='tight')
    plt.show()


# plot line of costs for US ETF portfolio
def PlotCostvsVal(period='6M', platform=None):
    #period,platform='6M','FSM SG'
    # collect the data
    hist_cost = calc_returns.CalcPortfolioHistoricalCost(platform=platform)
    hist_valuation = calc_returns.CalcPortfolioHistoricalValuation(platform=platform)
    
    df = hist_valuation.merge(hist_cost, how='left', on='Date')
    df = df.fillna(method='ffill')
    
    # optional filter: start date
    if period is not None:
        start_date = util.GetStartDate(period)
        start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
        df = df[df.Date>=start_date]
        df = df.reset_index(drop=True)

    # get the IRR performance % for the chart title
    #ar_etf = calc_returns.CalcIRR(period=period, platform=platform)
    returns = calc_returns.GetIRRFromDB()
    ar_etf = {
        'Period':period,
        'IRR': returns[returns.Period==period].IRR.iloc[0]
        }

    # create the plots
    fig, ax = plt.subplots()    # can set dpi=150 or 200 for bigger image; figsize=(8,6)
    title = 'Investment Cost vs Valuation' 
    if platform is not None:
        title = title + ' (%s)' % platform
    title = title + ' - %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')

    # add subtitle with return %
    subtitle = 'Performance %s: %s' % (period, '{0:.2%}'.format(ar_etf['IRR']))
    
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
    #ax.legend(frameon=False, loc='lower center', ncol=2)
    ax.legend(frameon=True, loc='best', ncol=1)
    
    if platform is None or platform=='FSM HK':
        # add annotation: 24 Nov 2020 took profit from Airlines, reinvested in ARKK
        x2_pos = x2[x2 == datetime.datetime(2020, 11, 24)].index
        ax.annotate('Took profit from JETS, reinvested in ARKK',
                    xy=('2020-11-24', y2.iloc[x2_pos]),
                    xytext=(-250, 0),
                    textcoords='offset points', color='gray',
                    arrowprops=dict(arrowstyle='-|>', color='gray')
                    )
        
        # add annotation: 4 Dec 2020 took profit from Tech
        x2_pos = x2[x2 == datetime.datetime(2020, 12, 4)].index
        ax.annotate('Took profit from Tech',
                    xy=('2020-12-04', y2.iloc[x2_pos]),
                    xytext=(-150, 0),
                    textcoords='offset points', color='gray',
                    arrowprops=dict(arrowstyle='-|>', color='gray')
                    )
    
    #fig.autofmt_xdate(rotation=45)
    plt.xticks(rotation=45, ha='right')

    # save output as PNG
    output_filename = 'CostVsValuation.png'
    output_fullpath = '%s/%s' % (_output_dir, output_filename)
    fig.savefig(output_fullpath, format='png', dpi=150, bbox_inches='tight')
    plt.show()
#PlotCostvsVal(period='6M', platform='FSM HK')
#PlotCostvsVal(period='3M', platform='FSM HK') #BUGGED (chart code)
#PlotCostvsVal(period='6M', platform='FSM SG')


# plot the top 10 holdings
def PlotTopHoldings():
    #variable with information: top_holdings
    top_holdings = calc_summary.TopHoldings()
    labels = list(top_holdings.Name)
    sizes_pct = list(top_holdings.PortfolioPct)
    plt.rcdefaults()
    fig, ax1 = plt.subplots()
    y_pos = np.arange(len(labels))
    #ax1.barh(y_pos, sizes)
    ax1.barh(y_pos, sizes_pct)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(labels)
    #ax1.set_ylabel('Security')
    vals = ax1.get_xticks()
    ax1.set_xticklabels(['{:,.0%}'.format(x) for x in vals])
    #ax1.set_xticklabels(['{:,.0f}'.format(x) for x in vals])
    ax1.set_xlabel('Percentage of Portfolio')
    #ax1.set_xlabel('Current Value (HKD)')
    #plt.xticks(rotation=45, ha='right')
    title = 'Top Holdings - %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')
    ax1.set(title=title)
    for index, value in enumerate(sizes_pct):
        ax1.text(value, index, str('{:,.2%}'.format(value)), color='black', fontweight='bold')
    plt.gca().invert_yaxis()

    # save output as PNG
    output_filename = 'TopHoldings.png'
    output_fullpath = '%s/%s' % (_output_dir, output_filename)
    fig.savefig(output_fullpath, format='png', dpi=150, bbox_inches='tight')
    plt.show()


# plots a stacked bar chart of realised PnL over time
def PlotRealisedPnLOverTime(period='6M'):
    # set the date range
    if period is not None:
        start_date = util.GetStartDate(period)
        start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())

    # prepare the data
    pnl_obj = calc_summary.GetHistoricalRealisedPnL()
    chart_data = pnl_obj[0][['Platform','Date','Type','RealisedPnL','PlatformCurrency','AssetType','Category']].copy()
    chart_data = chart_data[chart_data.Date > start_date]
    ccys = list(chart_data.PlatformCurrency.unique())
    GBPHKD = calc_fx.GetFXRate('HKD','GBP')
    SGDHKD = calc_fx.GetFXRate('HKD','SGD')
    ToHKD = {'HKD':1, 'SGD':SGDHKD, 'GBP':GBPHKD}
    chart_data['PnLInHKD'] = chart_data.PlatformCurrency.map(ToHKD) * chart_data.RealisedPnL
    chart_data.drop(['RealisedPnL'], axis=1, inplace=True)
    chart_data_grouped = chart_data.groupby([pd.Grouper(key='Date', freq='MS'), 'Type']).sum()
    chart_data_grouped = chart_data_grouped.reset_index()
    
    #labels = chart_data_grouped.index.get_level_values('Date')
    months = list(chart_data_grouped.Date.unique())
    
    df_dividends = chart_data_grouped[chart_data_grouped.Type=='Dividend'].copy()
    df_dividends.rename(columns={'PnLInHKD':'Dividend'}, inplace=True)
    df_dividends.drop(['Type'], axis=1, inplace=True)
    df_tradingpnl = chart_data_grouped[chart_data_grouped.Type=='Sell'].copy()
    df_tradingpnl.rename(columns={'PnLInHKD':'TradingPnL'}, inplace=True)
    df_tradingpnl.drop(['Type'], axis=1, inplace=True)
    
    df = pd.DataFrame(index=months)
    df = df.reset_index()
    df.rename(columns={'index':'Date'}, inplace=True)
    df = df.merge(df_dividends, how='left', left_on='Date', right_on='Date')
    df = df.merge(df_tradingpnl, how='left', left_on='Date', right_on='Date')
    df = df.fillna(0)
    
    width = 20

    # plot the chart
    fig, ax = plt.subplots()
    ax.bar(df.Date, df.Dividend, width, label='Dividends')
    ax.bar(df.Date, df.TradingPnL, width, bottom=df.Dividend, label='Trading PnL')
    ax.set_ylabel('Realised PnL (HKD)')
    title = 'Last %s Realised PnL - %s' % (period, datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S'))
    ax.set_title(title)
    ax.legend()
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    #ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(y=ymaj, ls='-', lw=0.25, color='black')

    #fig.autofmt_xdate(rotation=45)
    plt.xticks(rotation=45, ha='right')

    # save output as PNG
    output_filename = 'Last_%s_RealisedPnL.png' % period
    output_fullpath = '%s/%s' % (_output_dir, output_filename)
    fig.savefig(output_fullpath, format='png', dpi=150, bbox_inches='tight')
    plt.show()


# # XY plot with bubbles of PnL, PnL%, Portfolio % as size (IRR won't work because I don't hold funds long enough)
# def PlotXYBubbles(period='1M'):
#     # get the data
#     ps = calc_summary.GetPortfolioSummaryFromDB('Original')
#     ps = ps[ps.NoOfUnits!=0]

#     # # filter by supported instruments with market data (too many newly bought are NA)
#     # ps = ps[ps.BBGCode.isin(setup.GetListOfSupportedInstruments())]
#     # ps.reset_index(inplace=True)
#     # # calc IRR for holdings
#     # for i in range(len(ps)):
#     #     row = ps.iloc[i]
#     #     IRR = calc_returns.CalcIRR(platform=row.Platform,
#     #                                             bbgcode=row.BBGCode,
#     #                                             period=period)
#     #     ps.loc[i,'IRR'] = IRR['IRR']
    
#     # chart data
#     x = list(ps.PnLInHKD)
#     y = list(ps.PnLPct)
#     size = ps.PortfolioPct*1000
#     color = list(mcolors.TABLEAU_COLORS)[:len(x)]
#     labels = list(ps.Name)
    
#     # assign colour to Category
#     cats = list(ps.Category.unique())
#     cats_colour = {cats[i]: color[i] for i in range(len(cats))} 
#     ps['CategoryColour'] = ps.Category.map(cats_colour)

#     # plot the chart
#     fig, ax = plt.subplots()
    
#     scatter = ax.scatter(x, y, s=size, c=ps.CategoryColour, alpha=0.5)
    
#     # draw lines on the axis
#     ax.axhline(y=0, xmin=0, xmax=1, color='black', lw=0.5)
#     ax.axvline(x=0, ymin=0, ymax=1, color='black', lw=0.5)
    
#     # add labels to points
#     for i, txt in enumerate(labels):
#         ax.annotate(txt, 
#                     (x[i], y[i]),
#                     xytext=(10,0),
#                     textcoords='offset points', color='gray', fontsize=8
#                     )
    
#     title = 'Scatter Plot of Investments - %s' % (datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S'))
#     ax.set_title(title)
#     ax.xaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
#     ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.2f}'))
#     plt.xlabel('Unrealised PnL (HKD)', size=10)
#     plt.ylabel('Unrealised PnL %', size=10)
#     #ax.legend(title='Category', frameon=True, loc='best', ncol=1, bbox_to_anchor=(1,1))
    
#     # save output as PNG
#     output_filename = 'ScatterPlotOfInvestments.png'
#     output_fullpath = '%s/%s' % (_output_dir, output_filename)
#     fig.savefig(output_fullpath, format='png', dpi=300, bbox_inches='tight')
#     plt.show()


# plot performance of holdings over time
def PlotPerformanceOfHoldings(period='3M'):
    # get the start date
    start_date = util.GetStartDate(period)
    
    # get the list of instruments
    ps = calc_summary.GetPortfolioSummaryFromDB()
    ps = ps[ps.NoOfUnits>0]
    ps = ps[ps.BBGCode.isin(setup.GetListOfSupportedInstruments())]
    top10tickers = list(ps.sort_values('CurrentValueInHKD', ascending=False).head(10).BBGCode)
    
    # get the historical market data
    hp = mdata.GetHistoricalData()
    hp = hp[hp.BBGCode.isin(top10tickers)]
    hp = hp[hp.Date>=start_date]
    
    # plot the chart
    title = 'Top Holdings Performance (%s)' % period
    title = title + ' - %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')
    fig, ax = plt.subplots()
    ax.set_ylabel('Price Index')
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    
    # plot the cost
    for i in range(len(top10tickers)):
        tmp = hp[hp.BBGCode==top10tickers[i]].copy()
        base = tmp.Close.iloc[0]
        tmp.loc[:,'AdjustedIndex'] = tmp.loc[:,'Close'] / base * 100
        label = tmp.BBGCode.iloc[0]
        x = tmp.Date
        y = tmp.AdjustedIndex
        colour = list(mcolors.TABLEAU_COLORS.keys())[i]
        ax.plot(x, y, linestyle='-', label=label,color=colour)
    
    # add legend and other formatting
    ax.legend(title='Bloomberg ticker', frameon=True, loc='best', ncol=1, bbox_to_anchor=(1,1))
    plt.xticks(rotation=45, ha='right')
    ax.set_title(title)
    ax.axhline(y=100, xmin=0, xmax=1, color='black', lw=0.5)
    
    # save output as PNG
    output_filename = 'TopHoldingsPerformance.png'
    output_fullpath = '%s/%s' % (_output_dir, output_filename)
    fig.savefig(output_fullpath, format='png', dpi=300, bbox_inches='tight')
    plt.show()


# plot Leaders & Laggers (accurate for overnight; longer date range assumes no buying/selling)
def PlotLeadersAndLaggers(period=None):
    # get market data
    hp = mdata.GetHistoricalData()
    
    # filter by existing holdings
    ps = calc_summary.GetPortfolioSummaryFromDB('Original')
    ps = ps[ps.NoOfUnits>0]
    ps = ps[ps.BBGCode.isin(setup.GetListOfSupportedInstruments())]
    
    tickers = list(ps.BBGCode)
    
    # create a table to store the league table
    df = pd.DataFrame(tickers, columns=['BBGCode'])
    
    # calculate percentage change based on period
    def _CalculatePctChg(bbgcode, period=None):
        md = hp[hp.BBGCode==bbgcode].copy()
        if period is None:
            md = md.tail(2)
            
        else:
            start_date = util.GetStartDate(period)
            md = md[~(md.Date<start_date)]
        pct_chg = md.Close.iloc[-1] / md.Close.iloc[0] - 1
        return pct_chg
    
    # apply the calculation
    for i in range(len(df)):
        df.loc[i,'PctChg'] = _CalculatePctChg(df.BBGCode.iloc[i], period)
    
    # get existing holdings and calculate amount change
    df = df.merge(ps[['BBGCode','Name','CurrentValueInHKD']], how='left', on='BBGCode')
    df['AmountChgInHKD'] = df.CurrentValueInHKD - (df.CurrentValueInHKD / (1+df.PctChg))
    df = df[df.PctChg!=0]
    leaders = df.sort_values(['AmountChgInHKD'], ascending=False)#.head(5)
    leaders = leaders[leaders.PctChg>0]
    leaders.reset_index(inplace=True)
    laggers = df.sort_values(['AmountChgInHKD'], ascending=True)#.head(5)
    laggers = laggers[laggers.PctChg<0]
    laggers.reset_index(inplace=True)


    # plot the charts
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
    plt.rcdefaults()
    
    # Top 5 gainers
    labels1 = leaders.BBGCode
    sizes1 = leaders.AmountChgInHKD
    y_pos1 = np.arange(len(labels1))
    ax1.set_yticks(y_pos1)
    ax1.set_yticklabels(labels1)
    ax1.barh(y_pos1, sizes1, color='tab:green')
    vals1 = ax1.get_xticks()
    ax1.set_xticklabels(['{:,.0f}'.format(x) for x in vals1])
    ax1.set_xlabel('Gains (HKD)')
    title1 = 'Top gainers (+%s HKD)' % '{:,.0f}'.format(sum(sizes1))
    # add labels
    for i in range(len(sizes1)):
        ax1.text(x=sizes1[i],
                 y=y_pos1[i],
                 s=str('+'+'{:,.2%}'.format(leaders.PctChg.iloc[i])),
                 color='tab:green')
    ax1.set(title=title1)
    ax1.invert_yaxis()
    

    # Top 5 losers
    labels2 = laggers.BBGCode
    sizes2 = laggers.AmountChgInHKD*-1
    y_pos2 = np.arange(len(labels2))
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels(labels2)
    ax2.barh(y_pos2, sizes2, color='tab:red')
    vals2 = ax2.get_xticks()
    ax2.set_xticklabels(['{:,.0f}'.format(x) for x in vals2])
    ax2.set_xlabel('Losses (HKD)')
    title2 = 'Top losers (%s HKD)' % '{:,.0f}'.format(sum(sizes2)*-1)
    # add labels
    for i in range(len(sizes2)):
        ax2.text(x=sizes2[i],
                 y=y_pos2[i],
                 s=str('{:,.2%}'.format(laggers.PctChg.iloc[i])),
                 color='tab:red')
    ax2.set(title=title2)
    ax2.invert_yaxis()

    #plt.gca().invert_yaxis()
    if period is None:
        dr = 'overnight'
    else:
        dr = period
    title = 'Gainers and Losers (%s) - %s' % (dr, datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S'))
    plt.suptitle(title, fontsize=12)
    plt.subplots_adjust(top=0.85, wspace=0.4)

    # save output as PNG
    output_filename = 'GainersAndLosers.png'
    output_fullpath = '%s/%s' % (_output_dir, output_filename)
    fig.savefig(output_fullpath, format='png', dpi=150, bbox_inches='tight')
    plt.show()

