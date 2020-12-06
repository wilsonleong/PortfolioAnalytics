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

import calc_summary
import calc_returns
import calc_fx
import util
_output_dir = r'D:\Wilson\Documents\Personal Documents\Investments\PortfolioTracker\sample screenshots'


# displays a summary in the console (text)
def DisplaySummary():
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


    # Portfolio composition report
    #pcr = calc_summary.GetPortfolioSummary()
    pcr = calc_summary.ps
    pcr.to_csv('PortfolioSummaryInHKD.csv', index=False)
    pct = pcr.groupby('Category').agg({'CurrentValueInHKD':'sum'})
    pct.reset_index(inplace=True)
    total = pct.CurrentValueInHKD.sum()

    print ('\nTotal investments (exc. cash) by category in HKD equivalent:')
    for i in range(len(pct)):
        row = pct.iloc[i]
        cat = row.Category
        cat_withspace = cat + ' ' * (15-len(cat))
        value = row.CurrentValueInHKD
        pc = '%s' % '{:,.2f}'.format(value / total * 100) + '%'
        print ('%s \t %s' % (cat_withspace, pc))
    print ('Total investments: %s HKD' % '{:,.0f}'.format(total))

    # calculate the equivalent in other currencies
    ps_inc_cash = calc_summary.GetPortfolioSummaryIncCash()
    total_inc_cash = ps_inc_cash.CurrentValueInHKD.sum()
    total_USD = calc_fx.ConvertTo('USD','HKD',total_inc_cash)
    total_EUR = calc_fx.ConvertTo('EUR','HKD',total_inc_cash)
    total_GBP = calc_fx.ConvertTo('GBP','HKD',total_inc_cash)
    total_SGD = calc_fx.ConvertTo('SGD','HKD',total_inc_cash)
    
    print ('\nTotal portfolio value including cash:')
    print ('> %s HKD' % '{:,.0f}'.format(total_inc_cash))
    print ('> %s USD' % '{:,.0f}'.format(total_USD))
    print ('> %s EUR' % '{:,.0f}'.format(total_EUR))
    print ('> %s GBP' % '{:,.0f}'.format(total_GBP))
    print ('> %s SGD' % '{:,.0f}'.format(total_SGD))
    # print annualised returns on FSM HK & SG accounts
    ar_fsmhk = calc_returns.CalcModDietzReturn('FSM HK')
    ar_fsmsg = calc_returns.CalcModDietzReturn('FSM SG')
    print ('')
    print ('Annualised returns from inception (time-weighted):')
    print ('> FSM HK: \t\t' + '{:,.2%}'.format(ar_fsmhk['AnnualisedReturn']))
    print ('> FSM SG: \t\t' + '{:,.2%}'.format(ar_fsmsg['AnnualisedReturn']))
    print ('')


# display return %
def DisplayReturnPct():    
    # IRR
    date_ranges = util.date_ranges
    # get the IRR for the date ranges
    returns = {}
    for i in range(len(date_ranges)):
        returns[date_ranges[i]] = calc_returns.CalcIRR(period=date_ranges[i])

    # get the IRR % only
    returns_irr = {}
    for i in range(len(date_ranges)):
        returns_irr[date_ranges[i]] = returns[date_ranges[i]]['IRR']
    
    print ('Performance of Yahoo Finance supported instruments (money-weighted):')
    for i in range(len(returns_irr)):
        print ('> %s: \t\t' % list(returns_irr.keys())[i] + '{:,.2%}'.format(list(returns_irr.values())[i]))
    print ('')

    # plot the returns on a bar chart
    # prepare the data
    date_ranges = np.array(date_ranges)
    values = np.array(list(returns_irr.values()))

    # get SPX returns as benchmark
    spx = calc_returns.GetSPXReturns()
    #spx_returns = np.array(spx.Returns)

    # compare porfolio returns vs SPX    
    YTD_spx_diff = returns_irr['YTD'] - spx.loc['YTD','AnnualisedReturn']
    if YTD_spx_diff >= 0:
        comp_label = 'Beats'
        annotate_colour = 'tab:green'
    else:
        comp_label = 'Under-performs'
        annotate_colour = 'tab:red'
    comp_full_text = '%s SPX by %s' % (comp_label, '{:.2%}'.format(YTD_spx_diff))


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
            label='SPX', lw=0)

    # add annotate text
    ax.annotate(comp_full_text,
            xy=(0, returns_irr['YTD']),
            xytext=(0, 50),
            textcoords='offset points', 
            color=annotate_colour,
            weight='bold',
            arrowprops=dict(arrowstyle='-|>', color=annotate_colour)
            )

    # finalise chart
    ax.set_ylabel('Performance % (>1Y annualised)')
    #ax.set_ylabel('Annualised Return % for date range above 1Y')
    title = 'Portfolio Performance (IRR) - %s' % (datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S'))
    ax.set_title(title)
    #ax.legend()
    vals = ax.get_yticks()
    ax.set_yticklabels(['{:,.0%}'.format(x) for x in vals])
    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(y=ymaj, ls='-', lw=0.25, color='black')

    # save output as PNG
    output_filename = 'Performance.png'
    output_fullpath = '%s/%s' % (_output_dir, output_filename)
    fig.savefig(output_fullpath, format='png', dpi=150, bbox_inches='tight')
    plt.legend()
    plt.show()


# plot chart: portfolio composition
def PlotPortfolioComposition():
    # prepare data
    #pcr = calc_summary.GetPortfolioSummary()
    pcr = calc_summary.ps
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
    
    # save output as PNG
    output_filename = 'PortfolioComposition.png'
    output_fullpath = '%s/%s' % (_output_dir, output_filename)
    fig.savefig(output_fullpath, format='png', dpi=150, bbox_inches='tight')
    plt.show()
    
    
# 2020-12-02: plot donut chart by security currency
def PlotCurrecnyExposureAssetAllocation():
    by1 = 'SecurityCcy'
    title = 'Currency Exposure (inc. FX & cash)'
    by2 = 'SecurityType'
    title2 = 'Asset Allocation (inc. FX & cash)'
    
    title = title + ' - %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')
    title2 = title2 + ' - %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')
    
    pcr = calc_summary.GetPortfolioSummaryIncCash()
    
    # Currency Exposure
    pct = pcr.groupby(by1).agg({'CurrentValueInHKD':'sum'})
    pct.reset_index(inplace=True)
    total = pct.CurrentValueInHKD.sum()
    pct['Percentage'] = pct['CurrentValueInHKD']/pct.CurrentValueInHKD.sum()
    
    categories = pct['SecurityCcy']
    values = pct.Percentage
    categories_with_pct = []
    for i in range(len(categories)):
        categories_with_pct.append(categories[i] + ' (%s)' % '{:,.2%}'.format(values[i]))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8), subplot_kw=dict(aspect="equal"))
    wedges, texts = ax1.pie(values, wedgeprops=dict(width=0.3), startangle=-40)
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
        ax1.annotate(categories_with_pct[i], xy=(x, y), xytext=(1*np.sign(x), 1.1*y),horizontalalignment=horizontalalignment, **kw, fontsize=10)
    ax1.set_title(title)
    
    # Asset Allocation
    pct2 = pcr.groupby(by2).agg({'CurrentValueInHKD':'sum'})
    pct2.reset_index(inplace=True)
    total2 = pct2.CurrentValueInHKD.sum()
    pct2['Percentage'] = pct2['CurrentValueInHKD']/pct2.CurrentValueInHKD.sum()
    
    categories2 = pct2['SecurityType']
    values2 = pct2.Percentage
    categories_with_pct2 = []
    for i in range(len(categories2)):
        categories_with_pct2.append(categories2[i] + ' (%s)' % '{:,.2%}'.format(values2[i]))
        
    wedges2, texts2 = ax2.pie(values2, wedgeprops=dict(width=0.3), startangle=-40)
    #bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
    bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.5)
    kw = dict(arrowprops=dict(arrowstyle="-"), bbox=bbox_props, zorder=0, va="center")
    for i, p in enumerate(wedges2):
        ang = (p.theta2 - p.theta1)/2. + p.theta1
        y = np.sin(np.deg2rad(ang))
        x = np.cos(np.deg2rad(ang))
        horizontalalignment = {-1: "right", 1: "left"}[int(np.sign(x))]
        connectionstyle = "angle,angleA=0,angleB={}".format(ang)
        kw["arrowprops"].update({"connectionstyle": connectionstyle})
        ax2.annotate(categories_with_pct2[i], xy=(x, y), xytext=(1*np.sign(x), 1.1*y),horizontalalignment=horizontalalignment, **kw, fontsize=10)
    ax2.set_title(title2)
    
    #fig.tight_layout()
    
    # save output as PNG
    output_filename = 'CurrencyExposureAndAssetAllocation.png'
    output_fullpath = '%s/%s' % (_output_dir, output_filename)
    fig.savefig(output_fullpath, format='png', dpi=150, bbox_inches='tight')
    plt.show()


# # 2020-12-02: plot donut chart by security currency
def PlotPortfolioCompositionBy(by='FundHouse'):
    #by='SecurityType'
    if by=='SecurityCcy':
        title = 'Currency Exposure'
    elif by=='SecurityType':
        title = 'Asset Allocation'
    if by=='FundHouse':
        title = 'Holdings by Fund House'
    title = title + ' - %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')
    
    pcr = calc_summary.GetPortfolioSummaryIncCash()
    #pcr = calc_summary.ps
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
    #hist_valuation = calc_returns.hist_valuation
    hist_valuation = calc_returns.CalcPortfolioHistoricalValuation(platform=platform)
    
    df = hist_valuation.merge(hist_cost, how='left', on='Date')
    df = df.fillna(method='ffill')
    
    # optional filter: start date
    if period is not None:
        start_date = calc_returns.GetStartDate(period)
        start_date = datetime.datetime.combine(start_date, datetime.datetime.min.time())
        df = df[df.Date>=start_date]
        df = df.reset_index(drop=True)
    
    # create the plots
    fig, ax = plt.subplots()    # can set dpi=150 or 200 for bigger image; figsize=(8,6)
    title = 'Investment Cost vs Valuation' 
    if platform is not None:
        title = title + ' (%s)' % platform
    title = title + ' - %s' % datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S')
    
    # add subtitle with return %
    ar_etf = calc_returns.CalcIRR(period=period, platform=platform)
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
    ax.legend(frameon=False, loc='lower center', ncol=2)
    
    if platform is None or platform=='FSM HK':
        # add annotation: 01 Sep 2020 transfer of XLE VWO from Singapore account
        x2_pos = x2[x2 == datetime.datetime(2020, 9, 1)].index
        ax.annotate('Transfer in; built up positions',
                    xy=('2020-09-01', y2.iloc[x2_pos]),
                    xytext=(20, 20),
                    textcoords='offset points', color='gray',
                    arrowprops=dict(arrowstyle='-|>', color='gray')
                    )
    
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
    labels = list(calc_summary.top_holdings.Name)
    sizes_pct = list(calc_summary.top_holdings.PortfolioPct)
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
        start_date = calc_returns.GetStartDate(period)
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
    ax.set_ylabel('PnL in HKD')
    title = 'Last %s PnL -%s' % (period, datetime.datetime.strftime(datetime.datetime.today(), '%Y-%m-%d %H:%M:%S'))
    ax.set_title(title)
    ax.legend()
    ax.yaxis.set_major_formatter(mpl.ticker.StrMethodFormatter('{x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
    for ymaj in ax.yaxis.get_majorticklocs():
        ax.axhline(y=ymaj, ls='-', lw=0.25, color='black')

    # save output as PNG
    output_filename = 'Last_%s_PnL.png' % period
    output_fullpath = '%s/%s' % (_output_dir, output_filename)
    fig.savefig(output_fullpath, format='png', dpi=150, bbox_inches='tight')
    plt.show()

