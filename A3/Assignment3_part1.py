#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  6 20:14:53 2021

@author: sebastiengorgoni
"""

import pandas as pd
import numpy as np
import os 
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns
from numpy.matlib import repmat
import random

random.seed(10)

sns.set(style="whitegrid")

#Set the working directory
#os.chdir("/Users/sebastiengorgoni/Documents/HEC Master/Semester 4.2/Quantitative Asset & Risk Management/Assignments/Assignment 2")
print("Current working directory: {0}".format(os.getcwd()))

#Create files in the working directory
if not os.path.isdir('Plot'):
    os.makedirs('Plot')
    
if not os.path.isdir('Output'):
    os.makedirs('Output')

#Download the data of prices
df_price = pd.read_excel('Data_QAM2.xlsx', 'Prices',engine="openpyxl")
df_price.set_index('Dates', inplace=True)

#Create the client's benchmark (50% World Equities, 50% World Bonds)
benchmark = pd.DataFrame(data=(0.5*df_price['World Equities'] + 0.5*df_price['World Bonds']))

#In/Out Sample Prices
in_sample_price = df_price.loc[(df_price.index <= pd.to_datetime('2010-12-31'))].iloc[:,:]
out_sample_price = df_price.loc[(df_price.index > pd.to_datetime('2010-12-31'))].iloc[:,:]

#Compute the simple returns (IS: In-sample, OS: Out-Sample)
returns_price = np.log(df_price/df_price.shift(1)).replace(np.nan, 0)
returns_IS_price = np.log(in_sample_price/in_sample_price.shift(1)).replace(np.nan, 0)
returns_OS_price = returns_price.loc[(returns_price.index > pd.to_datetime('2010-12-31'))].iloc[:,:]

#Plot the overall cumulative returns of our assets
plt.figure(figsize=(12,7))
plt.plot((returns_price + 1).cumprod()*100)
plt.legend(returns_price.columns, loc='upper left', frameon=False)
plt.title('Cumulative Returns All Assets')
#plt.savefig('Plot/returns_asset.png')
#plt.show()
plt.close()

#Compute the IS and OS returns of the benchmark
benchmark_return_IS = pd.DataFrame({'IS Benchmark': 0.5*returns_IS_price['World Equities'] + 0.5*returns_IS_price['World Bonds']})
benchmark_return_OS = pd.DataFrame({'OS Benchmark': 0.5*returns_OS_price['World Equities'] + 0.5*returns_OS_price['World Bonds']})

#Compute the IS and OS weights of the benchmark (50% World Equities, 50% World Bonds)
benchmark_weights_IS = pd.DataFrame({'World Equities': np.ones(returns_IS_price.shape[0])*0.5,
                                     'World Bonds': np.ones(returns_IS_price.shape[0])*0.5,
                                     'US Investment Grade': np.zeros(returns_IS_price.shape[0]),
                                     'US High Yield': np.zeros(returns_IS_price.shape[0]),
                                     'Gold': np.zeros(returns_IS_price.shape[0]),
                                     'Energy': np.zeros(returns_IS_price.shape[0]),
                                     'Copper': np.zeros(returns_IS_price.shape[0])}, index = returns_IS_price.index)

benchmark_weights_OS = pd.DataFrame({'World Equities': np.ones(returns_OS_price.shape[0])*0.5,
                                     'World Bonds': np.ones(returns_OS_price.shape[0])*0.5,
                                     'US Investment Grade': np.zeros(returns_OS_price.shape[0]),
                                     'US High Yield': np.zeros(returns_OS_price.shape[0]),
                                     'Gold': np.zeros(returns_OS_price.shape[0]),
                                     'Energy': np.zeros(returns_OS_price.shape[0]),
                                     'Copper': np.zeros(returns_OS_price.shape[0])}, index = returns_OS_price.index)

# =============================================================================
# =============================================================================
# #Part 1: SAA
# =============================================================================
# =============================================================================

# =============================================================================
# 1.1
# =============================================================================

def MCR_calc(alloc, Returns):
    """ 
    This function computes the marginal contribution to risk (MCR), which 
    determine how much the portfolio volatility would change if we increase
    the weight of a particular asset.
    
    Parameters
    ----------
    alloc : TYPE
        Weights in the investor's portfolio
    Returns : TYPE
        The returns of the portfolio's assets
    Returns
    -------
    MCR : Object
        Marginal contribution to risk (MCR)
    """
    ptf=np.multiply(Returns,alloc);
    ptfReturns=np.sum(ptf,1); # Summing across columns
    vol_ptf=np.std(ptfReturns);
    Sigma=np.cov(np.transpose(Returns))
    MCR=np.matmul(Sigma,np.transpose(alloc))/vol_ptf;
    return MCR

###ERC Allocation###
def ERC(alloc,Returns):
    """ 
    This function computes the Equally-Weighted Risk Contribution Portfolio (ERC),
    which attributes the same risk contribution to all the assets.
    
    Parameters
    ----------
    alloc : TYPE
        Weights in the investor's portfolio
    Returns : TYPE
        The returns of the portfolio's assets
    Returns
    -------
    criterions : Object
        Optimal weights of assets in the portfolio.
    """
    ptf=np.multiply(Returns.iloc[:,:],alloc);
    ptfReturns=np.sum(ptf,1); # Summing across columns
    vol_ptf=np.std(ptfReturns);
    indiv_ERC=alloc*MCR_calc(alloc,Returns);
    criterion=np.power(indiv_ERC-vol_ptf/len(alloc),2)
    criterion=np.sum(criterion)*1000000000
    return criterion


x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001 #Set the first weights of the Gradient Descent

cons=({'type':'eq', 'fun': lambda x:sum(x)-1}, #Sum of weights is equal to 1
      {'type':'ineq', 'fun': lambda x: x[1]-0.01}, #Minimum of 1% in World Bonds
      {'type':'ineq', 'fun': lambda x: x[2]-0.01}) #Minimum of 1% in US investment grades

Bounds= [(0 , 1) for i in range(0,7)] #Long only positions

#Optimisation
res_ERC = minimize(ERC, x0, method='SLSQP', args=(returns_IS_price),bounds=Bounds,constraints=cons,options={'disp': True})

labels=list(returns_IS_price)

plt.figure(figsize=(15,7))

#Plot the optimal weights using ERC
weight_to_chart=np.array(res_ERC.x)
plt.subplot(131)
plt.plot(labels,weight_to_chart,'ro',labels,weight_to_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('Optimal Allocation ERC')
plt.ylabel('Ptf Weight')
plt.tight_layout()

#MCR for Max ERC
MCR_chart=MCR_calc(res_ERC.x, returns_IS_price)
MCR_chart=np.array(MCR_chart)
plt.subplot(132)
plt.plot(labels,MCR_chart,'ro',labels,MCR_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('MCR of ERC')
plt.ylabel('MCR')
plt.tight_layout()

#Test for the alpha x MCR ratio
ratio=res_ERC.x*MCR_chart
plt.subplot(133)
plt.plot(labels,ratio,'ro',labels,ratio*0,'b-')
plt.xticks(rotation=90)
plt.title('Alpha x MCR')
plt.ylabel('Ratio')
plt.tight_layout()
#plt.savefig('Plot/ERC_output.png')
#plt.show()
plt.close()

###Sharp Ratio Allocation###
def Max_SR(alloc, Returns, Rf=0):
    """
    This function computes the Maximum Sharpe Ratio Portfolio (SR),
    which attributes the weights that maximize the Sharpe Ratio.
    
    Parameters
    ----------
    alloc : TYPE
        Weights in the investor's portfolio
    Returns : TYPE
        The returns of the portfolio's assets
    Rf : TYPE
        The risk-free rate, assumed to be 0.
        
    Returns
    -------
    SR : Object
        Optimal weights of assets in the portfolio.
    """
    ptf=np.multiply(Returns.iloc[:,:],alloc);
    ptfReturns=np.sum(ptf,1); # Summing across columns
    mu_bar=np.mean(ptfReturns)-Rf;
    vol_ptf=np.std(ptfReturns);
    SR=-mu_bar/vol_ptf;
    return SR

x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001 #Set the first weights of the Gradient Descent

cons=({'type':'eq', 'fun': lambda x:sum(x)-1}, #Sum of weights is equal to 1
      {'type':'ineq', 'fun': lambda x: x[1]-0.01}, #Minimum of 1% in World Bonds
      {'type':'ineq', 'fun': lambda x: x[2]-0.01}) #Minimum of 1% in US investment grades

Bounds= [(0 , 1) for i in range(0,7)] #Long only positions

#Optimisation
res_SR = minimize(Max_SR, x0, method='SLSQP', args=(returns_IS_price),bounds=Bounds,constraints=cons,options={'disp': True})

plt.figure(figsize=(15,7))

#Plot the optimal weights using Sharpe Ratio
weight_to_chart=np.array(res_SR.x)
plt.subplot(131)
plt.plot(labels,weight_to_chart,'ro',labels,weight_to_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('Optimal Allocation SR')
plt.ylabel('Ptf Weight')
plt.tight_layout()

#MCR for Max SR
MCR_chart=MCR_calc(res_SR.x, returns_IS_price)
MCR_chart=np.array(MCR_chart)
plt.subplot(132)
plt.plot(labels,MCR_chart,'ro',labels,MCR_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('MCR of SR')
plt.ylabel('MCR')
plt.tight_layout()

#Test for excess return to MCR Ratio
reward=np.mean(returns_IS_price, 0)
ratio=MCR_chart/reward
plt.subplot(133)
plt.plot(labels,ratio,'ro',labels,ratio*0,'b-')
plt.xticks(rotation=90)
plt.title('Excess Return to MCR Ratio')
plt.ylabel('Ratio')
plt.tight_layout()
#plt.savefig('Plot/SR_output.png')
#plt.show()
plt.close()

###Most-Diversified Portfolio###
def Max_DP(alloc, Returns):
    """ 
    This function computes the Most-Diversified Portfolio (MDP),
    which attributes the same relative marginal volatility to all the assets.
    
    Parameters
    ----------
    alloc : TYPE
        Weights in the investor's portfolio
    Returns : TYPE
        The returns of the portfolio's assets
    Returns
    -------
    Div_ratio : Object
        Optimal weights of assets in the portfolio.
    """
    ptf=np.multiply(Returns.iloc[:,:],alloc);
    ptfReturns=np.sum(ptf,1); # Summing across columns
    vol_ptf=np.std(ptfReturns);
    
    numerator=np.multiply(np.std(Returns),alloc);
    numerator=np.sum(numerator);
    
    Div_Ratio=-numerator/vol_ptf;
    return Div_Ratio

x0 = np.array([0, 0, 0, 0, 0, 0, 0])+0.00001 #Set the first weights of the Gradient Descent

cons=({'type':'eq', 'fun': lambda x:sum(x)-1}, #Sum of weights is equal to 1
      {'type':'ineq', 'fun': lambda x: x[1]-0.01}, #Minimum of 1% in World Bonds
      {'type':'ineq', 'fun': lambda x: x[2]-0.01}) #Minimum of 1% in US investment grades

Bounds= [(0 , 1) for i in range(0,7)] #Long only positions

#Optimisation
res_MDP = minimize(Max_DP, x0, method='SLSQP', args=(returns_IS_price), bounds=Bounds,constraints=cons, options={'disp': True}) #options={'disp': True} means I want feedbacks in my optimization

plt.figure(figsize=(15,7))

#Plot the optimal weights using Most-Diversified Portfolio (MDP)
weight_to_chart=np.array(res_MDP.x)
plt.subplot(131)
plt.plot(labels,weight_to_chart,'ro',labels,weight_to_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('Optimal Allocation MDP')
plt.ylabel('Ptf Weight')
plt.tight_layout()

# MCR for Max SR
MCR_chart=MCR_calc(res_MDP.x, returns_IS_price);
MCR_chart=np.array(MCR_chart)
plt.subplot(132)
plt.plot(labels,MCR_chart,'ro',labels,MCR_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('MCR of MDP')
plt.ylabel('MCR')
plt.tight_layout()

#Test for MCR/Risk ratio
MCR_risk = MCR_chart/np.std(returns_IS_price)
plt.subplot(133)
plt.plot(labels, MCR_risk, 'ro', labels, MCR_chart*0,'b-')
plt.xticks(rotation=90)
plt.title('MCR-per-Risk Ratio')
plt.ylabel('Ratio')
plt.tight_layout()
#plt.savefig('Plot/MDP_output.png')
#plt.show()
plt.close()

#Compute the returns of the ERC, SR, MDP portfolio IS
portfolio_IS_ERC = pd.DataFrame({'ERC': np.sum(np.multiply(returns_IS_price,np.transpose(res_ERC.x)), 1)})
portfolio_IS_SR = pd.DataFrame({'SR': np.sum(np.multiply(returns_IS_price,np.transpose(res_SR.x)), 1)})
portfolio_IS_MDP = pd.DataFrame({'MDP': np.sum(np.multiply(returns_IS_price,np.transpose(res_MDP.x)), 1)})

def hit_ratio(return_dataset):
    """
    This function determine the hit ratio of any time series returns

    Parameters
    ----------
    return_dataset : TYPE
        The returns of the asset.

    Returns
    -------
    TYPE
        It returns the hit ratio.

    """
    return len(return_dataset[return_dataset >= 0]) / len(return_dataset)

def max_drawdown(cum_returns):
    """
    It determines the maximum drawdown over the cumulative returns
    of a time series.

    Parameters
    ----------
    cum_returns : TYPE
        Cumulative Return.

    Returns
    -------
    max_monthly_drawdown : TYPE
        Evolution of the max drawdown (negative output).

    """
    roll_max = cum_returns.cummax()
    monthly_drawdown = cum_returns/roll_max - 1
    max_monthly_drawdown = monthly_drawdown.cummin()
    return max_monthly_drawdown

#Determine the maximum drawdown (minimum as the output is negative)  
max_drawdown((portfolio_IS_ERC['ERC']+1).cumprod()).min()

def perf(data, benchmark, name, name_plt):
    """
    This function compute all the required performances of a time series.
    It also plot the monthly returns, the evolution of the mayx drawdown and 
    the cumulative return of the portfolio vs. benchmark

    Parameters
    ----------
    data : TYPE
        Returns of a given portfolio.
    benchmark : TYPE
        Returns of the benchmark.
    name : TYPE
        Name of the dataframe.
    name_plt : TYPE
        Name given to the plot.

    Returns
    -------
    df : TYPE
        Return a dataframe that contains the annualized returns, volatility,
        Sharpe ratio, max drawdown and hit ratio.

    """
    plt.figure(figsize=(20,7))
    plt.subplot(131)
    plt.plot(data, 'b')
    plt.title("Monthly Returns", fontsize=15)
    exp = np.mean(data,0)*12
    vol = np.std(data,0)*np.power(12,0.5)
    sharpe = exp/vol
    max_dd = max_drawdown((data+1).cumprod())
    plt.subplot(132)
    plt.plot(max_dd, 'g')
    plt.title("Evolution of Max Drawdown", fontsize=15)
    hit = hit_ratio(data)
    df = pd.DataFrame({name: [exp, vol, sharpe, max_dd.min(), hit]}, index = ['Mean', 'Volatility', 'SharpeRatio', 'MaxDrawdown', 'HitRatio'])
    plt.subplot(133)
    plt.plot((data + 1).cumprod()*100, 'b', label='Portfolio')
    plt.plot((benchmark + 1).cumprod()*100, 'r', label='Benchmark')
    plt.title(name_plt, fontsize=15)
    plt.legend(loc='upper left', frameon=True)
    #plt.savefig('Plot/'+name_plt+'.png')
    #plt.show()
    plt.close()
    return df

#Determine the perfomances of the ERC, SR, MDP portfolio IS
ERC_IS_results = perf(portfolio_IS_ERC['ERC'], benchmark_return_IS['IS Benchmark'], 'ERC', 'ERC Cumulative Returns In-Sample')
SR_IS_results = perf(portfolio_IS_SR['SR'], benchmark_return_IS['IS Benchmark'], 'SR', 'SR Cumulative Returns In-Sample')
MDP_IS_results = perf(portfolio_IS_MDP['MDP'], benchmark_return_IS['IS Benchmark'], 'MDP', 'MDP Cumulative Returns In-Sample')

#Determine the perfomances of thebenchmark portfolio IS
benchmark_IS_results = perf(benchmark_return_IS['IS Benchmark'], benchmark_return_IS['IS Benchmark'], 'Benchmark IS', 'Benchmark Cumulative Returns In-Sample')

#Merge all results 
SAA_IS_results = pd.concat([benchmark_IS_results, ERC_IS_results, SR_IS_results, MDP_IS_results], axis=1)
SAA_IS_results.to_latex('Output/SAA_IS_results.tex')

###Tracking Error###
""" Check the end of code. """

###Information Ratio###
""" Check the end of code. """

# =============================================================================
# 1.2
# =============================================================================

#Compute the returns of the ERC, SR, MDP portfolio OS
portfolio_OS_ERC = pd.DataFrame({'ERC': np.sum(np.multiply(returns_OS_price,np.transpose(res_ERC.x)), 1)})
portfolio_OS_SR = pd.DataFrame({'SR': np.sum(np.multiply(returns_OS_price,np.transpose(res_SR.x)), 1)})
portfolio_OS_MDP = pd.DataFrame({'MDP': np.sum(np.multiply(returns_OS_price,np.transpose(res_MDP.x)), 1)})

#Determine the perfomances of the ERC, SR, MDP portfolio OS
ERC_OS_results = perf(portfolio_OS_ERC['ERC'], benchmark_return_OS['OS Benchmark'], 'ERC', 'ERC Cumulative Returns Out-Sample')
SR_OS_results = perf(portfolio_OS_SR['SR'], benchmark_return_OS['OS Benchmark'], 'SR', 'SR Cumulative Returns Out-Sample')
MDP_OS_results = perf(portfolio_OS_MDP['MDP'], benchmark_return_OS['OS Benchmark'], 'MDP', 'MDP Cumulative Returns Out-Sample')

#Determine the perfomances of thebenchmark portfolio OS
benchmark_OS_results = perf(benchmark_return_OS['OS Benchmark'], benchmark_return_OS['OS Benchmark'], 'Benchmark OS', 'Benchmark Cumulative Returns Out-Sample')

#Merge all results 
SAA_OS_results = pd.concat([benchmark_OS_results, ERC_OS_results, SR_OS_results, MDP_OS_results], axis=1)
SAA_OS_results.to_latex('Output/SAA_OS_results.tex')

###Tracking Error###
""" Check the end of code. """

###Information Ratio###
""" Check the end of code. """

# =============================================================================
# =============================================================================
# Part 2: TAA
# =============================================================================
# =============================================================================

# =============================================================================
# 2.1
# =============================================================================

#Download the data of carry
df_carry = pd.read_excel("Data_QAM2.xlsx",sheet_name='Carry',engine="openpyxl")
df_carry.index = df_carry['Unnamed: 0']
df_carry.index.name = 'date'
del df_carry['Unnamed: 0']

###Value In-Sample###

#Take the IS carry
df_carry_insample = df_carry[df_carry.index <= pd.to_datetime('2010-12-31')]

#Standardize the dataframe 
df_carry_insample_Z = (df_carry_insample - np.mean(df_carry_insample))/np.std(df_carry_insample)
df_carry_insample_Z['median'] = df_carry_insample_Z.median(axis=1)
df_carry_insample_Z

#The value factor will oppose the assets with a z-score higher than the median of that month to the assets with a score lower than that median value. 
df_carry_insample_Z_pos = df_carry_insample_Z.copy()
for value in df_carry_insample_Z.iloc[:,0:7].columns:
    list_value = []
    for i in range(len(df_carry_insample_Z)):
        if df_carry_insample_Z[value][i] > df_carry_insample_Z['median'][i]:
            list_value.append(1)
        else:
            list_value.append(-1)
    df_carry_insample_Z_pos[f'{value}'] = list_value
#df_carry_insample_Z_pos.to_csv('Carry_postion.csv')
df_carry_insample_Z_pos

position = df_carry_insample_Z_pos

position[position[position.columns] > 0]

del position['median']

#Attribue the same weights to all long positions (short position respectively).
weight_final = {}
for i in range(len(position)):
    num_pos = np.sum(np.array(position.values[i]) > 0)
    num_neg = np.sum(np.array(position.values[i]) < 0)
    array = []
    for j in range(len(position.values[i])):
        #print(position.values[i][j]/num_neg)
        if position.iloc[i,j] > 0:
           position.iloc[i,j] = position.iloc[i,j]/num_pos
        else:
            position.iloc[i,j] = position.iloc[i,j]/num_neg
    #print(array)
    #position.loc[i] = array
    #print(position.values[i])
#position.to_csv('value_position.csv')

position_IS = position.copy()

#Compute the returns of the value factor IS
portofolio_value = position.mul(returns_IS_price).sum(axis=1)

#Allocate a 2% ex ante volatility budget 
vol_budget = 0.02
vol_value_scaled_IS = vol_budget/portofolio_value.std()

value_IS_scaled = pd.DataFrame({'Value': vol_value_scaled_IS * portofolio_value})

print(value_IS_scaled)

###Momentum In-Sample###

#Source: https://www.youtube.com/watch?v=dnrJ4zwCADM

#Calculate the returns over the past 11 months
returns_IS_past11 = (returns_IS_price+1).rolling(11).apply(np.prod) - 1
returns_IS_past11 = returns_IS_past11.dropna()

#Compute the quintile  of each asset each periods
returns_IS_quantile = returns_IS_past11.T.apply(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"), axis=0).T

#Long asset with the highest quantile (winners), short asset with the lowest quantile (losers)
for i in returns_IS_quantile.columns:
    returns_IS_quantile.loc[returns_IS_quantile[i] == 4, i] = 0.5
    returns_IS_quantile.loc[returns_IS_quantile[i] == 0, i] = -0.5
    returns_IS_quantile.loc[returns_IS_quantile[i] == 1, i] = 0
    returns_IS_quantile.loc[returns_IS_quantile[i] == 2, i] = 0
    returns_IS_quantile.loc[returns_IS_quantile[i] == 3, i] = 0

#Check if the wum of weights is equal to zero 
np.sum(returns_IS_quantile, axis=1) == 0

#Shift the weights as we ignore the last month
weights_IS_mom = returns_IS_quantile.shift(1, axis = 0).dropna()

#Compute the returns of the momentum factor OS
portofolio_mom = weights_IS_mom.multiply(returns_IS_price).sum(axis=1)

#Allocate a 2% ex ante volatility budget 
vol_budget = 0.02
vol_mom_scaled_IS = vol_budget/portofolio_mom.std()

mom_IS_scaled = pd.DataFrame({'Momentum': vol_mom_scaled_IS * portofolio_mom})

print(mom_IS_scaled)

###Collect the Results###
#Compute the performances of value and momentum
value_IS_results = perf(value_IS_scaled['Value'], benchmark_return_IS['IS Benchmark'], 'Value', 'Value Cumulative Returns In-Sample')
mom_IS_results = perf(mom_IS_scaled['Momentum'], benchmark_return_IS['IS Benchmark'], 'Momentum', 'Momentum Cumulative Returns In-Sample')

#Merge Value and Momentum IS
ValMom_IS_results = pd.concat([benchmark_IS_results, value_IS_results, mom_IS_results], axis=1)
ValMom_IS_results.to_latex('Output/ValMom_IS_results.tex')

#Merge Value and Benchmark IS
ValBen_IS_results = pd.concat([benchmark_IS_results, value_IS_results], axis=1)
ValBen_IS_results.to_latex('Output/ValBen_IS_results.tex')

#Merge Momentum and Benchmark IS
MomBen_IS_results = pd.concat([benchmark_IS_results, mom_IS_results], axis=1)
MomBen_IS_results.to_latex('Output/MomBen_IS_results.tex')

# =============================================================================
# 2.2
# =============================================================================

#Download the VIX Index 
df_vix = pd.read_excel("Data_QAM2.xlsx",sheet_name='VIX',engine="openpyxl")
df_vix.set_index('Dates', inplace=True)

#Plot the VIX
plt.figure(figsize=(10,7))
#plt.plot(df_vix)
plt.title('Level of VIX Index')
#plt.savefig('Plot/VIX.png')

#Standardize the VIX
df_vix = (df_vix - df_vix.mean())/df_vix.std()

#Compute the percentage change of VIX
df_vix['Percentage Change'] = np.log(df_vix['VIX']/df_vix['VIX'].shift(1))
df_vix['Percentage Change'] = df_vix['Percentage Change'].replace(np.nan, 0)

#Set the IS and OS VIX 
df_vix_IS = df_vix.loc[(df_vix.index <= pd.to_datetime('2010-12-31'))]
df_vix_OS = df_vix.loc[(df_vix.index > pd.to_datetime('2010-12-31'))]

#Compute the 90% quantile of VIX
quantile = df_vix_IS.quantile(q=0.90)
df_vix_IS['Quantile'] = np.ones(df_vix_IS.shape[0])*quantile.VIX

#Plot the IS VIX level and its percentage change
plt.figure(figsize=(15,7))
plt.subplot(121)
plt.plot(df_vix_IS['Percentage Change'], 'r')
plt.title('Percentage Change of VIX In-Sample')
plt.subplot(122)
plt.plot(df_vix_IS['VIX'], label = 'Standardized VIX')
plt.plot(df_vix_IS['Quantile'], label = '90% Quantile')
plt.title('Standardized VIX In-Sample')
plt.legend(loc='upper left', frameon=True)
#plt.savefig('Plot/VIX_IS.png')
#plt.show()
plt.close()

#Plot the covariance matrices and 10-month rolling covariances between VIX and value/momentum factor
plt.figure(figsize=(15,10))
plt.subplot(221)
corr_value_IS = sns.heatmap(pd.concat([df_vix_IS['VIX'], value_IS_scaled], axis=1).corr(), annot=True)
plt.title('Full Period Covarianve In-Sample')
plt.subplot(222)
corr_mom_IS = sns.heatmap(pd.concat([df_vix_IS['VIX'], mom_IS_scaled], axis=1).corr(), annot=True)
plt.title('Full Period Covarianve In-Sample')
plt.subplot(223)
plt.plot(df_vix_IS['VIX'].rolling(10).corr(value_IS_scaled))
plt.title('Rolling Covariance Value-VIX (10 Months) In-sample')
plt.subplot(224)
plt.plot(df_vix_IS['VIX'].rolling(10).corr(mom_IS_scaled))
plt.title('Rolling Covariance Momentum-VIX (10 Months) In-sample')
#plt.savefig('Plot/VIX_analysis_IS.png')
#plt.show()
plt.close()

###Parametric###
df_vix_IS.loc[df_vix_IS['VIX'] <= df_vix_IS['Quantile'], 'Long/Short'] = 1 #Expansion if VIX is lower than the 90% quantile
df_vix_IS.loc[df_vix_IS['VIX'] > df_vix_IS['Quantile'], 'Long/Short'] = -1 #Recession if VIX is higher than the 90% quantile

ValMom_returns_IS = pd.concat([value_IS_scaled['Value'], mom_IS_scaled['Momentum']], axis=1)

#Take the data from 2001-08-31 as the momentum factor is realized from this date 
ValMom_returns_IS_adj = ValMom_returns_IS.loc[ValMom_returns_IS.index >= '2001-09-28']
df_vix_IS_adj = df_vix_IS.loc[df_vix_IS.index >= '2001-08-31', 'Long/Short']

#Compute the parametric weights between value and momentum factor (to determine the TAA)
lambda_ra=3
numerator = []
denominator = []
n = 0
for i in range (0, len(ValMom_returns_IS_adj)):
        temp_num = df_vix_IS['Long/Short'][i] * ValMom_returns_IS_adj.iloc[i]
        temp_den = (df_vix_IS['Long/Short'][i]*df_vix_IS['Long/Short'][i]) * np.multiply(ValMom_returns_IS_adj.iloc[i], ValMom_returns_IS_adj.iloc[i].transpose())
        numerator.append(temp_num)
        denominator.append(temp_den)
        n += 1
    
alpha = (1/lambda_ra)*(np.sum(numerator, axis=0)/np.sum(denominator))
alpha_para_IS = alpha/(alpha[0] + alpha[1])

#Determine the returns of the portfolio when using only parametrics
TAA_IS_parametrics_returns = value_IS_scaled['Value']*alpha_para_IS[0] + mom_IS_scaled['Momentum']*alpha_para_IS[1]
TAA_IS_parametrics_results = perf(TAA_IS_parametrics_returns, benchmark_return_IS['IS Benchmark'], 'TAA_IS_Parametrics', 'TAA (Parametrics) Cumulative Returns In-Sample')

###Complex Strategy###
"""
- Assign the parametric weights to the value and momentum factor when the standardized VIX is lower than its 90% quantile
- Assign 100% to the Momentum and 0% to value factor standardized VIX is higher than its 90% quantile and increasing.
- Assign -100% to the Momentum and 200% to value factor standardized VIX is higher than its 90% quantile and decreasing.
"""
df_vix_IS.loc[(df_vix_IS['VIX'] <= df_vix_IS['Quantile']), 'Value Position'] = alpha_para_IS[0] #Long Value
df_vix_IS.loc[(df_vix_IS['VIX'] > df_vix_IS['Quantile'])  & (df_vix_IS['Percentage Change'] >= 0), 'Value Position'] = 0 #Short Value
df_vix_IS.loc[(df_vix_IS['VIX'] > df_vix_IS['Quantile'])  & (df_vix_IS['Percentage Change'] < 0), 'Value Position'] = 2 #Long Value

df_vix_IS.loc[df_vix_IS['VIX'] <= df_vix_IS['Quantile'], 'Mom Position'] = alpha_para_IS[1] #Long Mom
df_vix_IS.loc[(df_vix_IS['VIX'] > df_vix_IS['Quantile'])  & (df_vix_IS['Percentage Change'] >= 0), 'Mom Position'] = 1 #Long Value
df_vix_IS.loc[(df_vix_IS['VIX'] > df_vix_IS['Quantile']) & (df_vix_IS['Percentage Change'] < 0), 'Mom Position'] = -1 #Long Value

TAA_IS_VIX = pd.DataFrame({'Returns Strategy': value_IS_scaled['Value']*df_vix_IS['Value Position'] + mom_IS_scaled['Momentum']*df_vix_IS['Mom Position']}).replace(np.nan, 0)

TAA_IS_VIX_results = perf(TAA_IS_VIX['Returns Strategy'], benchmark_return_IS['IS Benchmark'], 'TAA_IS_VIX', 'TAA (Own Strategy) Cumulative Returns In-Sample')

TAA_IS = pd.concat([benchmark_IS_results, value_IS_results, mom_IS_results, TAA_IS_VIX_results, TAA_IS_parametrics_results], axis=1)
TAA_IS.to_latex('Output/TAA_IS.tex')

###Tracking Error###
""" Check the end of code. """

###Information Ratio###
""" Check the end of code. """

# =============================================================================
# 2.3
# =============================================================================

###Value###

#Take the OS carry
df_carry_outsample = df_carry[df_carry.index > pd.to_datetime('2010-12-31')]

#Standardize the carry
df_carry_outsample_Z = (df_carry_outsample - np.mean(df_carry_outsample))/np.std(df_carry_outsample)
df_carry_outsample_Z['median'] = df_carry_outsample_Z.median(axis=1)
df_carry_outsample_Z

#The value factor will oppose the assets with a z-score higher than the median of that month to the assets with a score lower than that median value. 
df_carry_outsample_Z_pos = df_carry_outsample_Z.copy()
for value in df_carry_outsample_Z.iloc[:,0:7].columns:
    list_value = []
    for i in range(len(df_carry_outsample_Z)):
        if df_carry_outsample_Z[value][i] > df_carry_outsample_Z['median'][i]:
            list_value.append(1)
        else:
            list_value.append(-1)
    df_carry_outsample_Z_pos[f'{value}'] = list_value
#df_carry_outsample_Z_pos.to_csv('Carry_postion.csv')
df_carry_outsample_Z_pos

position = df_carry_outsample_Z_pos
position[position[position.columns] > 0]

del position['median']

#Attribue the same weights to all long positions (short position respectively).
weight_final = {}
for i in range(len(position)):
    num_pos = np.sum(np.array(position.values[i]) > 0)
    num_neg = np.sum(np.array(position.values[i]) < 0)
    array = []
    for j in range(len(position.values[i])):
        #print(position.values[i][j]/num_neg)
        if position.iloc[i,j] > 0:
           position.iloc[i,j] = position.iloc[i,j]/num_pos
        else:
            position.iloc[i,j] = position.iloc[i,j]/num_neg
    #print(array)
    #position.loc[i] = array
    #print(position.values[i])
position.to_csv('value_position.csv')

position_OS = position.copy()

#Compute the returns of the value factor OS
portofolio_value = position.mul(returns_OS_price).sum(axis=1)

#Allocate a 2% ex ante volatility budget (IS volatility)

value_OS_scaled = pd.DataFrame({'Value': vol_value_scaled_IS * portofolio_value})
#value_OS_scaled = pd.DataFrame({'Value': portofolio_value})

print(value_OS_scaled)

###Momentum###

#Source: https://www.youtube.com/watch?v=dnrJ4zwCADM

#Calculate the returns over the past 11 months
returns_past11 = (returns_price+1).rolling(11).apply(np.prod) - 1
returns_past11 = returns_past11.dropna()

#Compute the quintile  of each asset each periods
returns_quantile = returns_past11.T.apply(lambda x: pd.qcut(x, 5, labels=False, duplicates="drop"), axis=0).T

#Long asset with the highest quantile (winners), short asset with the lowest quantile (losers)
for i in returns_quantile.columns:
    returns_quantile.loc[returns_quantile[i] == 4, i] = 0.5
    returns_quantile.loc[returns_quantile[i] == 0, i] = -0.5
    returns_quantile.loc[returns_quantile[i] == 1, i] = 0
    returns_quantile.loc[returns_quantile[i] == 2, i] = 0
    returns_quantile.loc[returns_quantile[i] == 3, i] = 0

#Check if the wum of weights is equal to zero 
np.sum(returns_quantile, axis=1) == 0

#Shift the weights as we ignore the last month
weights_mom = returns_quantile.shift(1, axis = 0).dropna()

#Compute the returns of the momentum factor OS
portofolio_mom_OS = weights_mom.multiply(returns_OS_price).sum(axis=1)
portofolio_mom_OS = pd.DataFrame({'Momentum': portofolio_mom_OS})
portofolio_mom_OS =  portofolio_mom_OS.loc[(portofolio_mom_OS != 0).any(1)]

#Allocate a 2% ex ante volatility budget (IS volatility)

mom_OS_scaled = vol_mom_scaled_IS * portofolio_mom_OS
#mom_OS_scaled = portofolio_mom_OS

print(mom_OS_scaled)

###Collect the Results###
#Compute the performances of value and momentum
value_OS_results = perf(value_OS_scaled['Value'], benchmark_return_OS['OS Benchmark'], 'Value', 'Value Cumulative Returns Out-Sample')
mom_OS_results = perf(mom_OS_scaled['Momentum'], benchmark_return_OS['OS Benchmark'], 'Momentum', 'Momentum Cumulative Returns Out-Sample')

#Merge Value and Momentum OS
ValMom_OS_results = pd.concat([benchmark_OS_results, value_OS_results, mom_OS_results], axis=1)
ValMom_OS_results.to_latex('Output/ValMom_OS_results.tex')

#Merge Value and Benchmark OS
ValBen_OS_results = pd.concat([benchmark_OS_results, value_OS_results], axis=1)
ValBen_OS_results.to_latex('Output/ValBen_OS_results.tex')

#Merge Momentum and Benchmark OS
MomBen_OS_results = pd.concat([benchmark_OS_results, mom_OS_results], axis=1)
MomBen_OS_results.to_latex('Output/MomBen_OS_results.tex')

###Out-Sample VIX###

#Compute the quantile
quantile = df_vix_IS.quantile(q=0.90)
df_vix_OS['Quantile'] = np.ones(df_vix_OS.shape[0])*quantile.VIX

"""
plt.figure(figsize=(15,7))
plt.subplot(121)
plt.plot(df_vix_OS['Percentage Change'], 'r')
plt.title('Percentage Change of VIX Out-Sample')
plt.subplot(122)
plt.plot(df_vix_OS['VIX'], 'b', label = 'Standardized VIX')
#plt.plot(df_vix_IS['Quantile'], label = '90% Quantile')
plt.title('Standardized VIX Out-Sample')
#plt.legend(loc='upper left', frameon=True)
plt.savefig('Plot/VIX_OS.png')
plt.show()
plt.close()
"""
#Plot the covariance matrices and 10-month rolling covariances between VIX and value/momentum factor
plt.figure(figsize=(15,10))
plt.subplot(221)
corr_value_OS = sns.heatmap(pd.concat([df_vix_OS['VIX'], value_OS_scaled], axis=1).corr(), annot=True)
plt.title('Full Period Covarianve Out-Sample')
plt.subplot(222)
corr_mom_OS = sns.heatmap(pd.concat([df_vix_OS['VIX'], mom_OS_scaled], axis=1).corr(), annot=True)
plt.title('Full Period Covarianve Out-Sample')
plt.subplot(223)
plt.plot(df_vix_OS['VIX'].rolling(10).corr(value_OS_scaled))
plt.title('Rolling Covariance Value-VIX (10 Months) Out-sample')
plt.subplot(224)
plt.plot(df_vix_OS['VIX'].rolling(10).corr(mom_OS_scaled))
plt.title('Rolling Covariance Momentum-VIX (10 Months) Out-sample')
#plt.savefig('Plot/VIX_analysis_OS.png')
#plt.show()
plt.close()


###Parametric 
TAA_OS_parametrics_returns = value_OS_scaled['Value']*alpha_para_IS[0] + mom_OS_scaled['Momentum']*alpha_para_IS[1]
TAA_OS_parametrics_results = perf(TAA_OS_parametrics_returns, benchmark_return_OS['OS Benchmark'], 'TAA_OS_Parametrics', 'TAA (Parametrics) Cumulative Returns Out-Sample')


###Complex Strategy (Adaptive Quantile)### 
"""
- Assign the parametric weights to the value and momentum factor when the standardized VIX is lower than its 90% quantile
- Assign 100% to the Momentum and 0% to value factor standardized VIX is higher than its 90% quantile and increasing.
- Assign -100% to the Momentum and 200% to value factor standardized VIX is higher than its 90% quantile and decreasing.
- The quantile is determined from the first period of the IS up to the last data observed (updated each months)
"""
df_vix_OS['Value Position'] = 0
df_vix_OS['Mom Position'] = 0
quantile_tot = []
for i in range(0, df_vix_OS.shape[0]):
    quantile = df_vix.iloc[:df_vix_IS.shape[0]+1+i, 0].quantile(q=0.90)
    quantile_tot.append(quantile)
    print(quantile)
    if (df_vix_OS.iloc[i, 0] <= quantile):
        df_vix_OS.iloc[i, 3] = alpha_para_IS[0]
    elif (df_vix_OS.iloc[i, 0] > quantile) & (df_vix_OS.iloc[i, 1] >= 0):
        df_vix_OS.iloc[i, 3] = 0
    elif (df_vix_OS.iloc[i, 0] > quantile) & (df_vix_OS.iloc[i, 1] < 0):
        df_vix_OS.iloc[i, 3] = 2
    else:
        pass
    
    if (df_vix_OS.iloc[i, 0] <= quantile):
        df_vix_OS.iloc[i, 4] = alpha_para_IS[1]
    elif (df_vix_OS.iloc[i, 0] > quantile) & (df_vix_OS.iloc[i, 1] >= 0):
        df_vix_OS.iloc[i, 4] = 1
    elif (df_vix_OS.iloc[i, 0] > quantile) & (df_vix_OS.iloc[i, 1] < 0):
        df_vix_OS.iloc[i, 4] = -1 
    else:
        pass       

quantile_OS = pd.DataFrame({'Quantile': quantile_tot}, index = df_vix_OS.index)

#Plot the OS VIX level and its percentage change
plt.figure(figsize=(15,7))
plt.subplot(121)
plt.plot(df_vix_OS['Percentage Change'], 'r')
plt.title('Percentage Change of VIX Out-Sample')
plt.subplot(122)
plt.plot(df_vix_OS['VIX'], 'b', label = 'Standardized VIX')
plt.plot(quantile_OS, 'orange', label = '90% Quantile (In-Sample & Out-Sample)')
plt.title('Standardized VIX Out-Sample')
plt.legend(loc='upper left', frameon=True)
#plt.savefig('Plot/VIX_OS.png')
#plt.show()
plt.close()

#Compute the returns of the TAA    
TAA_OS_VIX = pd.DataFrame({'Returns Strategy': value_OS_scaled['Value']*df_vix_OS['Value Position'] + mom_OS_scaled['Momentum']*df_vix_OS['Mom Position']}).replace(np.nan, 0)

#Compute the performances of the TAA   
TAA_OS_VIX_results = perf(TAA_OS_VIX['Returns Strategy'], benchmark_return_OS['OS Benchmark'], 'TAA_OS_VIX', 'TAA (Own Strategy) Cumulative Returns Out-Sample')

#Merge all performances 
TAA_OS = pd.concat([benchmark_OS_results, value_OS_results, mom_OS_results, TAA_OS_VIX_results, TAA_OS_parametrics_results], axis=1)
TAA_OS.to_latex('Output/TAA_OS.tex')

###Tracking Error###
""" Check the end of code. """

###Information Ratio###
""" Check the end of code. """

# =============================================================================
# =============================================================================
# Part 3
# =============================================================================
# =============================================================================

###Collecting All Weights###
SAA_weights_IS = pd.DataFrame({'World Equities': np.ones(returns_IS_price.shape[0])*res_SR.x[0],
                                   'World Bonds': np.ones(returns_IS_price.shape[0])*res_SR.x[1],
                                   'US Investment Grade': np.ones(returns_IS_price.shape[0])*res_SR.x[2],
                                   'US High Yield': np.ones(returns_IS_price.shape[0])*res_SR.x[3],
                                   'Gold': np.ones(returns_IS_price.shape[0])*res_SR.x[4],
                                   'Energy': np.ones(returns_IS_price.shape[0])*res_SR.x[5],
                                   'Copper': np.ones(returns_IS_price.shape[0])*res_SR.x[6]
                                   }, index = returns_IS_price.index)

SAA_weights_OS = pd.DataFrame({'World Equities': np.ones(returns_OS_price.shape[0])*res_SR.x[0],
                                   'World Bonds': np.ones(returns_OS_price.shape[0])*res_SR.x[1],
                                   'US Investment Grade': np.ones(returns_OS_price.shape[0])*res_SR.x[2],
                                   'US High Yield': np.ones(returns_OS_price.shape[0])*res_SR.x[3],
                                   'Gold': np.ones(returns_OS_price.shape[0])*res_SR.x[4],
                                   'Energy': np.ones(returns_OS_price.shape[0])*res_SR.x[5],
                                   'Copper': np.ones(returns_OS_price.shape[0])*res_SR.x[6]
                                   }, index = returns_OS_price.index)

value_weights_IS = position_IS

value_weights_OS = position_OS

mom_weights_IS = weights_IS_mom

mom_weights_OS = weights_mom.iloc[(weights_mom.index > pd.to_datetime('2010-12-31'))].iloc[:,:]

VIX_weights_IS = pd.DataFrame({'Value Position': df_vix_IS['Value Position'], 
                               'Mom Position': df_vix_IS['Mom Position']})

VIX_weights_OS = pd.DataFrame({'Value Position': df_vix_OS['Value Position'], 
                               'Mom Position': df_vix_OS['Mom Position']})

value_weights_IS = value_weights_IS.multiply(VIX_weights_IS['Value Position'], axis='index').replace(np.nan, 0)

mom_weight_IS = mom_weights_IS.multiply(VIX_weights_IS['Mom Position'], axis='index').replace(np.nan, 0)

TAA_weights_IS = value_weights_IS.multiply(VIX_weights_IS['Value Position'], axis='index') +  mom_weights_IS.multiply(VIX_weights_IS['Mom Position'], axis='index').replace(np.nan, 0)

TAA_weights_OS = (value_weights_OS.multiply(VIX_weights_OS['Value Position'], axis='index') +  mom_weights_OS.multiply(VIX_weights_OS['Mom Position'], axis='index').replace(np.nan, 0))

weight_target_IS = SAA_weights_IS + TAA_weights_IS

weight_target_OS = SAA_weights_OS + TAA_weights_OS

#Determine the covariance matrix
sigma_IS = returns_IS_price.cov().values
sigma_OS = returns_OS_price.cov().values

###Tracking Error Formula###
def TE(weight_ptf, weight_target, sigma=sigma_IS):
    """
    This function computes the tracking error between
    a portfolio and a benchmark.

    Parameters
    ----------
    weight_ptf : TYPE
        Weight of our portfolio.
        
    weight_target : TYPE
        Weight the benchmark portfolio.
    
    sigma : TYPE
        Covariance matrix.

    Returns
    -------
    vol_month : TYPE
        Monthly Tracking Error.

    """

    diff_alloc = weight_ptf - weight_target
    temp =  np.matmul(diff_alloc.T, sigma)
    var_month = np.matmul(temp, diff_alloc)
    vol_month = np.power(var_month, 0.5)
    return vol_month 

###Ex Ante Tracking Error & Replication Portfolion In-Sample###

#The sum of weights is equal to 1, and no position to US investment grade
cons = ({'type':'eq', 'fun': lambda x: sum(x)-1},
        {'type':'eq', 'fun': lambda x: x[2]})

#We allow long/short positions (we put high bounds as we sometimes have position of 110-120%)
Bounds = [(-10 , 10) for i in range(0,7)] 

#Run the TE in each periods IS
weight_opt_IS = []
TE_ReplicationvsTarget_IS = []
for i in range(0, weight_target_IS.shape[0]):
    x0 = weight_target_IS.copy()*0 + (1/weight_target_IS.shape[1]) #Initial weights for gradient descent (equal weights)
    x0 = x0.iloc[0].values
    weight_target = weight_target_IS.iloc[i, :].values
    sigma = sigma_IS
    opt_TE_IS = minimize(TE, x0, method='SLSQP', bounds=Bounds, args=(weight_target), constraints=cons, options={'disp': True}) #bounds=Bounds, 
    weight_opt_IS.append(opt_TE_IS.x)
    TE_ReplicationvsTarget_IS.append(opt_TE_IS.fun)
 
#Store the TE of Replication vs Target IS
TE_ReplicationvsTarget_IS = pd.DataFrame({'TE_IS': TE_ReplicationvsTarget_IS}, index = weight_target_IS.index)
 
#Store the weights of the replication ptf IS                                      
weight_rep_IS = np.array(weight_opt_IS)
weight_rep_IS = pd.DataFrame({'World Equities': weight_rep_IS[:, 0],
                              'World Bonds': weight_rep_IS[:, 1],
                              'US Investment Grade': weight_rep_IS[:, 2],
                              'US High Yield': weight_rep_IS[:, 3],
                              'Gold': weight_rep_IS[:, 4],
                              'Energy': weight_rep_IS[:, 5],
                              'Copper': weight_rep_IS[:, 6]}, index = weight_target_IS.index)

#Determine the returns of the target and replication portolio
return_target_IS  = np.sum(np.multiply(returns_IS_price, weight_target_IS), axis=1)

return_rep_IS  = np.sum(np.multiply(returns_IS_price, weight_rep_IS), axis=1)

#Determine the performances of the target and replication portolio
perf_target_IS = perf(return_target_IS, benchmark_return_IS['IS Benchmark'], 'Model', 'Target Portfolio In-Sample')
perf_rep_IS = perf(return_rep_IS, benchmark_return_IS['IS Benchmark'], 'Replication', 'Replication Portfolio In-Sample')

#Merge the performances of target and replication ptf
perf_target_rep_IS = pd.concat([perf_target_IS, perf_rep_IS], axis=1)
perf_target_rep_IS.to_latex('Output/perf_target_rep_IS.tex')

#Plot the IS cumulative Returns and ex-ante TE between target and replication ptf
plt.figure(figsize=(14,7))
plt.subplot(121)
plt.plot((return_target_IS+1).cumprod()*100, 'b', label='Model Portfolio')
plt.plot((return_rep_IS+1).cumprod()*100, 'r', label='Replication Portfolio')
plt.title('In-Sample Cumulative Returns')
plt.legend(loc='upper left', frameon=True)
plt.subplot(122)
plt.plot(TE_ReplicationvsTarget_IS)
plt.title('Ex-Ante Tracking Error between Real Ptf vs. SAA+TAA')
#plt.savefig('Plot/target_replication_is.png')
#plt.show()
plt.close()

###Ex Post Tracking Error & Replication Portfolio Out-Sample###

#The sum of weights is equal to 1, and no position to US investment grade
cons = ({'type':'eq', 'fun': lambda x: sum(x)-1},
        {'type':'eq', 'fun': lambda x: x[2]})

#We allow long/short positions
Bounds = [(-10 , 10) for i in range(0,7)] #Long/short positions

#Run the TE in each periods OS
weight_opt_OS = []
TE_ReplicationvsTarget_OS = []
for i in range(0, weight_target_OS.shape[0]):
    x0 = weight_target_IS.copy()*0 + (1/weight_target_IS.shape[1]) #Initial weights for gradient descent (equal weights)
    x0 = x0.iloc[0].values
    weight_target = weight_target_OS.iloc[i, :].values
    sigma = sigma_IS
    opt_TE_OS = minimize(TE, x0, method='SLSQP', bounds=Bounds, args=(weight_target), constraints=cons, options={'disp': True}) #Powell works bounds=Bounds, 
    weight_opt_OS.append(opt_TE_OS.x)
    TE_ReplicationvsTarget_OS.append(opt_TE_OS.fun)
 
#Store the TE of Replication vs Target OS    
TE_ReplicationvsTarget_OS = pd.DataFrame({'TE_OS': TE_ReplicationvsTarget_OS}, index = weight_target_OS.index)

#Store the weights of the replication ptf OS  
weight_rep_OS = np.array(weight_opt_OS)
weight_rep_OS = pd.DataFrame({'World Equities': weight_rep_OS[:, 0],
                              'World Bonds': weight_rep_OS[:, 1],
                              'US Investment Grade': weight_rep_OS[:, 2],
                              'US High Yield': weight_rep_OS[:, 3],
                              'Gold': weight_rep_OS[:, 4],
                              'Energy': weight_rep_OS[:, 5],
                              'Copper': weight_rep_OS[:, 6]}, index = weight_target_OS.index)

#Determine the returns of the target and replication portolio
return_target_OS  = np.sum(np.multiply(returns_OS_price, weight_target_OS), axis=1)
return_rep_OS  = np.sum(np.multiply(returns_OS_price, weight_rep_OS), axis=1)

#Determine the performances of the target and replication portolio
perf_target_OS = perf(return_target_OS, benchmark_return_OS['OS Benchmark'], 'Model', 'Target Portfolio Out-Sample')
perf_rep_OS = perf(return_rep_OS, benchmark_return_OS['OS Benchmark'], 'Replication', 'Replication Portfolio Out-Sample')

#Merge the performances of target and replication ptf
perf_target_rep_OS = pd.concat([perf_target_OS, perf_rep_OS], axis=1)
perf_target_rep_OS.to_latex('Output/perf_target_rep_OS.tex')

#Plot the OS cumulative Returns and ex-ante TE between target and replication ptf
plt.figure(figsize=(14,7))
plt.subplot(121)
plt.plot((return_target_OS+1).cumprod()*100, 'b', label='Model Portfolio')
plt.plot((return_rep_OS+1).cumprod()*100, 'r', label='Replication Portfolio')
plt.title('Out-of-Sample Cumulative Returns')
plt.legend(loc='upper left', frameon=True)
plt.subplot(122)
plt.plot(TE_ReplicationvsTarget_OS)
plt.title('Ex-Post Tracking Error between Real Ptf vs. SAA+TAA')
#plt.savefig('Plot/target_replication_os.png')
#plt.show()
plt.close()

###Tracking Error###
"""Between SAA and Benchmark"""
#In-Sample
Tot_TE_SAAvsBenchmark_IS = []
for i in range(0, SAA_weights_IS.shape[0]):
    weight_target = benchmark_weights_IS.iloc[i, :].values
    sigma = sigma_IS
    TE_SAAvsBenchmark_IS = TE(SAA_weights_IS.iloc[i, :].values, weight_target)
    Tot_TE_SAAvsBenchmark_IS.append(TE_SAAvsBenchmark_IS)

#Out-Sample    
Tot_TE_SAAvsBenchmark_OS = []
for i in range(0, SAA_weights_OS.shape[0]):
    weight_target = benchmark_weights_OS.iloc[i, :].values
    sigma = sigma_IS
    TE_SAAvsBenchmark_OS = TE(SAA_weights_OS.iloc[i, :].values, weight_target)
    Tot_TE_SAAvsBenchmark_OS.append(TE_SAAvsBenchmark_OS)
 
TE_SAAvsBenchmark_IS = pd.DataFrame({'TE_IS': Tot_TE_SAAvsBenchmark_IS}, index = benchmark_return_IS.index)
TE_SAAvsBenchmark_OS = pd.DataFrame({'TE_OS': Tot_TE_SAAvsBenchmark_OS}, index = benchmark_return_OS.index)

plt.figure(figsize=(13,7))
plt.subplot(121)
plt.plot(TE_SAAvsBenchmark_IS, 'b')
plt.title('Ex-Ante Tracking Error between SAA vs. Benchmark')
plt.subplot(122)
plt.plot(TE_SAAvsBenchmark_OS, 'r')
plt.title('Ex-Post Tracking Error between SAA vs. Benchmark')
#plt.savefig('Plot/TE_SAAvsBenchmark.png')
#plt.show() 
plt.close()

"""Between TAA and Benchmark"""
#In-Sample
Tot_TE_TAAvsBenchmark_IS = []
for i in range(0, TAA_weights_IS.shape[0]):
    weight_target = benchmark_weights_IS.iloc[i, :].values
    sigma = sigma_IS
    TE_TAAvsBenchmark_IS = TE(TAA_weights_IS.iloc[i, :].values, weight_target)
    Tot_TE_TAAvsBenchmark_IS.append(TE_TAAvsBenchmark_IS)
 
#Out-Sample    
Tot_TE_TAAvsBenchmark_OS = []
for i in range(0, TAA_weights_OS.shape[0]):
    weight_target = benchmark_weights_OS.iloc[i, :].values
    sigma = sigma_IS
    TE_TAAvsBenchmark_OS = TE(TAA_weights_OS.iloc[i, :].values, weight_target)
    Tot_TE_TAAvsBenchmark_OS.append(TE_TAAvsBenchmark_OS)
 
TE_TAAvsBenchmark_IS = pd.DataFrame({'TE_IS': Tot_TE_TAAvsBenchmark_IS}, index = TAA_weights_IS.index)
TE_TAAvsBenchmark_OS = pd.DataFrame({'TE_OS': Tot_TE_TAAvsBenchmark_OS}, index = TAA_weights_OS.index)

plt.figure(figsize=(13,7))
plt.subplot(121)
plt.plot(TE_TAAvsBenchmark_IS , 'b')
plt.title('Ex-Ante Tracking Error between TAA vs. Benchmark')
plt.subplot(122)
plt.plot(TE_TAAvsBenchmark_OS , 'r')
plt.title('Ex-Post Tracking Error between TAA vs. Benchmark')
#plt.savefig('Plot/TE_TAAvsBenchmark.png')
#plt.show() 
plt.close()

"""Between TAA and SAA"""
#In-Sample
Tot_TE_SAAvsTAA_IS = []
for i in range(0, weight_target_IS.shape[0]):
    weight_target = SAA_weights_IS.iloc[i, :].values
    sigma = sigma_IS
    TE_SAAvsTAA_IS = TE(TAA_weights_IS.iloc[i, :].values, weight_target)
    Tot_TE_SAAvsTAA_IS.append(TE_SAAvsTAA_IS)

#Out-Sample 
Tot_TE_SAAvsTAA_OS = []
for i in range(0, weight_target_OS.shape[0]):
    weight_target = SAA_weights_OS.iloc[i, :].values
    sigma = sigma_IS
    TE_SAAvsTAA_OS = TE(TAA_weights_OS.iloc[i, :].values, weight_target)
    Tot_TE_SAAvsTAA_OS.append(TE_SAAvsTAA_OS)
 
TE_SAAvsTAA_IS = pd.DataFrame({'TE_IS': Tot_TE_SAAvsTAA_IS}, index = TAA_weights_IS.index)
TE_SAAvsTAA_OS = pd.DataFrame({'TE_OS': Tot_TE_SAAvsTAA_OS}, index = TAA_weights_OS.index)

plt.figure(figsize=(13,7))
plt.subplot(121)
plt.plot(TE_SAAvsTAA_IS, 'b')
plt.title('Ex-Ante Tracking Error between TAA vs. SAA')
plt.subplot(122)
plt.plot(TE_SAAvsTAA_OS, 'r')
plt.title('Ex-Post Tracking Error between TAA vs. SAA')
#plt.savefig('Plot/TE_SAAvsTAA.png')
#plt.show() 
plt.close()

"""Between SAA+TAA and SAA"""
#In-Sample
Tot_TE_SAAvsTarget_IS = []
for i in range(0, weight_target_IS.shape[0]):
    weight_target = SAA_weights_IS.iloc[i, :].values
    sigma = sigma_IS
    TE_SAAvsTarget_IS = TE(weight_target_IS.iloc[i, :].values, weight_target)
    Tot_TE_SAAvsTarget_IS.append(TE_SAAvsTarget_IS)

#Out-Sample 
Tot_TE_SAAvsTarget_OS = []
for i in range(0, weight_target_OS.shape[0]):
    weight_target = SAA_weights_OS.iloc[i, :].values
    sigma = sigma_IS
    TE_SAAvsTarget_OS = TE(weight_target_OS.iloc[i, :].values, weight_target)
    Tot_TE_SAAvsTarget_OS.append(TE_SAAvsTarget_OS)
 
TE_SAAvsTarget_IS = pd.DataFrame({'TE_IS': Tot_TE_SAAvsTarget_IS}, index = weight_target_IS.index)
TE_SAAvsTarget_OS = pd.DataFrame({'TE_OS': Tot_TE_SAAvsTarget_OS}, index = weight_target_OS.index)

plt.figure(figsize=(13,7))
plt.subplot(121)
plt.plot(TE_SAAvsTarget_IS, 'b')
plt.title('Ex-Ante Tracking Error between SAA+TAA vs. SAA')
plt.subplot(122)
plt.plot(TE_SAAvsTarget_OS, 'r')
plt.title('Ex-Post Tracking Error between SAA+TAA vs. SAA')
#plt.savefig('Plot/TE_SAAvsTarget.png')
#plt.show() 
plt.close()

"""Between Replication and SAA+TAA"""
plt.figure(figsize=(13,7))
plt.subplot(121)
plt.plot(TE_ReplicationvsTarget_IS, 'b')
plt.title('Ex-Ante Tracking Error between Replication vs. SAA+TAA')
plt.subplot(122)
plt.plot(TE_ReplicationvsTarget_OS, 'r')
plt.title('Ex-Post Tracking Error between Replication vs. SAA+TAA')
#plt.savefig('Plot/TE_ReplicationvsTarget.png')
#plt.show() 
plt.close()

"""Collect All Output SAA+TAA Gradually"""
TE_annualized_output_1 =  pd.DataFrame({'In-Sample': [TE_SAAvsBenchmark_IS['TE_IS'].mean(axis=0)*(12**0.5)],
                          'Out-Sample': [TE_SAAvsBenchmark_OS['TE_OS'].mean(axis=0)*(12**0.5)]},
                            index=['SAA vs. Benchmark'])
TE_annualized_output_1.to_latex('Output/TE_annualized_output_1.tex')

TE_annualized_output_2 =  pd.DataFrame({'In-Sample': [TE_SAAvsBenchmark_IS['TE_IS'].mean(axis=0)*(12**0.5), TE_TAAvsBenchmark_IS['TE_IS'].mean(axis=0)*(12**0.5)],
                          'Out-Sample': [TE_SAAvsBenchmark_OS['TE_OS'].mean(axis=0)*(12**0.5), TE_TAAvsBenchmark_OS['TE_OS'].mean(axis=0)*(12**0.5)]},
                            index=['SAA vs. Benchmark', 'TAA vs. Benchmark'])
TE_annualized_output_2.to_latex('Output/TE_annualized_output_2.tex')

TE_annualized_output_3 =  pd.DataFrame({'In-Sample': [TE_SAAvsBenchmark_IS['TE_IS'].mean(axis=0)*(12**0.5), TE_TAAvsBenchmark_IS['TE_IS'].mean(axis=0)*(12**0.5), TE_SAAvsTAA_IS['TE_IS'].mean(axis=0)*(12**0.5)],
                          'Out-Sample': [TE_SAAvsBenchmark_OS['TE_OS'].mean(axis=0)*(12**0.5), TE_TAAvsBenchmark_OS['TE_OS'].mean(axis=0)*(12**0.5), TE_SAAvsTAA_OS['TE_OS'].mean(axis=0)*(12**0.5)]},
                            index=['SAA vs. Benchmark', 'TAA vs. Benchmark', 'TAA vs. SAA'])
TE_annualized_output_3.to_latex('Output/TE_annualized_output_3.tex')

TE_annualized_output_4 =  pd.DataFrame({'In-Sample': [TE_SAAvsBenchmark_IS['TE_IS'].mean(axis=0)*(12**0.5), TE_TAAvsBenchmark_IS['TE_IS'].mean(axis=0)*(12**0.5), TE_SAAvsTAA_IS['TE_IS'].mean(axis=0)*(12**0.5), TE_SAAvsTarget_IS['TE_IS'].mean(axis=0)*(12**0.5)],
                          'Out-Sample': [TE_SAAvsBenchmark_OS['TE_OS'].mean(axis=0)*(12**0.5), TE_TAAvsBenchmark_OS['TE_OS'].mean(axis=0)*(12**0.5), TE_SAAvsTAA_OS['TE_OS'].mean(axis=0)*(12**0.5), TE_SAAvsTarget_OS['TE_OS'].mean(axis=0)*(12**0.5)]},
                            index=['SAA vs. Benchmark', 'TAA vs. Benchmark', 'TAA vs. SAA', 'TAA+SAA vs. SAA'])
TE_annualized_output_4.to_latex('Output/TE_annualized_output_4.tex')

TE_annualized_output_5 =  pd.DataFrame({'In-Sample': [TE_SAAvsBenchmark_IS['TE_IS'].mean(axis=0)*(12**0.5), TE_TAAvsBenchmark_IS['TE_IS'].mean(axis=0)*(12**0.5), TE_SAAvsTAA_IS['TE_IS'].mean(axis=0)*(12**0.5), TE_SAAvsTarget_IS['TE_IS'].mean(axis=0)*(12**0.5), TE_ReplicationvsTarget_IS['TE_IS'].mean(axis=0)*(12**0.5)],
                          'Out-Sample': [TE_SAAvsBenchmark_OS['TE_OS'].mean(axis=0)*(12**0.5), TE_TAAvsBenchmark_OS['TE_OS'].mean(axis=0)*(12**0.5), TE_SAAvsTAA_OS['TE_OS'].mean(axis=0)*(12**0.5), TE_SAAvsTarget_OS['TE_OS'].mean(axis=0)*(12**0.5), TE_ReplicationvsTarget_OS['TE_OS'].mean(axis=0)*(12**0.5)]},
                            index=['SAA vs. Benchmark', 'TAA vs. Benchmark', 'TAA vs. SAA', 'TAA+SAA vs. SAA', 'Replication vs. TAA+SAA'])
TE_annualized_output_5.to_latex('Output/TE_annualized_output_5.tex')

###Information Ratio###

def info_ratio(return_p, return_b):
    """
    This function determine the information ratio of an investment.
    Source: https://en.wikipedia.org/wiki/Information_ratio

    Parameters
    ----------
    return_p : TYPE
        Returns of the actual portfolio.
    return_b : TYPE
        Returns of the benchmark.

    Returns
    -------
    TYPE
        It returns the annualized info. ratio.

    """
    excess = return_p - return_b
    return (excess.mean(axis=0)*12)/(excess.std(axis=0)*(12**0.5))

"""Between SAA and Benchmark"""
IR_SAAvsBenchmark_IS = info_ratio(portfolio_IS_SR['SR'], benchmark_return_IS['IS Benchmark'])
IR_SAAvsBenchmark_OS = info_ratio(portfolio_OS_SR['SR'], benchmark_return_OS['OS Benchmark'])

"""Between TAA and Benchmark"""
IR_TAAvsBenchmark_IS = info_ratio(TAA_IS_VIX['Returns Strategy'], benchmark_return_IS['IS Benchmark'])
IR_TAAvsBenchmark_OS = info_ratio(TAA_OS_VIX['Returns Strategy'], benchmark_return_OS['OS Benchmark'])

"""Between TAA and SAA"""
IR_TAAvsSAA_IS = info_ratio(TAA_IS_VIX['Returns Strategy'], portfolio_IS_SR['SR'])
IR_TAAvsSAA_OS = info_ratio(TAA_OS_VIX['Returns Strategy'], portfolio_OS_SR['SR'])

"""Between SAA+TAA and SAA"""
IR_TargetvsSAA_IS = info_ratio(return_target_IS, portfolio_IS_SR['SR'])
IR_TargetvsSAA_OS = info_ratio(return_target_OS, portfolio_OS_SR['SR'])

"""Between Replication Portfolion and SAA+TAA (i.e. Target Portfolio)"""
IR_ReplicationvsTarget_IS = info_ratio(return_rep_IS, return_target_IS)
IR_ReplicationvsTarget_OS = info_ratio(return_rep_OS, return_target_OS)

"""Collect All Output Gradually"""
IR_output_1 = pd.DataFrame({'In-Sample': [IR_SAAvsBenchmark_IS],
                            'Out-Sample': [IR_SAAvsBenchmark_OS]},
                         index=['SAA vs. Benchmark'])
IR_output_1.to_latex('Output/IR_output_1.tex')

IR_output_2 = pd.DataFrame({'In-Sample': [IR_SAAvsBenchmark_IS, IR_TAAvsBenchmark_IS],
                            'Out-Sample': [IR_SAAvsBenchmark_OS, IR_TAAvsBenchmark_OS]},
                         index=['SAA vs. Benchmark', 'TAA vs. Benchmark'])
IR_output_2.to_latex('Output/IR_output_2.tex')

IR_output_3 = pd.DataFrame({'In-Sample': [IR_SAAvsBenchmark_IS, IR_TAAvsBenchmark_IS, IR_TAAvsSAA_IS],
                          'Out-Sample': [IR_SAAvsBenchmark_OS, IR_TAAvsBenchmark_OS, IR_TAAvsSAA_OS]},
                         index=['SAA vs. Benchmark', 'TAA vs. Benchmark', 'TAA vs. SAA'])
IR_output_3.to_latex('Output/IR_output_3.tex')

IR_output_4 = pd.DataFrame({'In-Sample': [IR_SAAvsBenchmark_IS, IR_TAAvsBenchmark_IS, IR_TAAvsSAA_IS, IR_TargetvsSAA_IS],
                          'Out-Sample': [IR_SAAvsBenchmark_OS, IR_TAAvsBenchmark_OS, IR_TAAvsSAA_OS, IR_TargetvsSAA_OS]},
                         index=['SAA vs. Benchmark', 'TAA & SAA vs. SAA', 'TAA vs. SAA', 'TAA+SAA vs. SAA'])
IR_output_4.to_latex('Output/IR_output_4.tex')

IR_output_5 = pd.DataFrame({'In-Sample': [IR_SAAvsBenchmark_IS, IR_TAAvsBenchmark_IS, IR_TAAvsSAA_IS, IR_TargetvsSAA_IS, IR_ReplicationvsTarget_IS],
                          'Out-Sample': [IR_SAAvsBenchmark_OS, IR_TAAvsBenchmark_OS, IR_TAAvsSAA_OS, IR_TargetvsSAA_OS, IR_ReplicationvsTarget_OS]},
                         index=['SAA vs. Benchmark', 'TAA & SAA vs. SAA', 'TAA vs. SAA', 'TAA+SAA vs. SAA', 'Replication vs. TAA & SAA'])
IR_output_5.to_latex('Output/IR_output_5.tex')

###Allocation/Selection Performance Attribution In-Sample###

#Weight for each type of asset (target)
weights_bonds_target_IS = weight_target_IS['World Bonds'] + weight_target_IS['US Investment Grade'] + weight_target_IS['US High Yield']
weights_equities_target_IS = weight_target_IS['World Equities']
weights_commodities_target_IS = weight_target_IS['Gold'] +  weight_target_IS['Energy'] + weight_target_IS['Copper']

#Weight for each type of asset (replication portfolio)
weights_bonds_rep_IS = weight_rep_IS['World Bonds'] + weight_rep_IS['US Investment Grade'] + weight_rep_IS['US High Yield']
weights_equities_rep_IS = weight_rep_IS['World Equities']
weights_commodities_rep_IS = weight_rep_IS['Gold'] +  weight_rep_IS['Energy'] + weight_rep_IS['Copper']

#Performance for each type of asset (target)
perf_bonds_target_IS = (np.multiply(returns_IS_price['World Bonds'], weight_target_IS['World Bonds']) + np.multiply(returns_IS_price['US Investment Grade'], weight_target_IS['US Investment Grade']) + np.multiply(returns_IS_price['US High Yield'], weight_target_IS['US High Yield']))/weights_bonds_target_IS
perf_equities_target_IS = (np.multiply(returns_IS_price['World Equities'], weight_target_IS['World Equities']))/weights_equities_target_IS
perf_commodities_target_IS = (np.multiply(returns_IS_price['Gold'], weight_target_IS['Gold']) + np.multiply(returns_IS_price['Energy'], weight_target_IS['Energy']) + np.multiply(returns_IS_price['Copper'], weight_target_IS['Copper']))/weights_commodities_target_IS

#Performance for each type of asset (replication portfolio)
perf_bonds_rep_IS = (np.multiply(returns_IS_price['World Bonds'], weight_rep_IS['World Bonds']) + np.multiply(returns_IS_price['US Investment Grade'], weight_rep_IS['US Investment Grade']) + np.multiply(returns_IS_price['US High Yield'], weight_rep_IS['US High Yield']))/weights_bonds_rep_IS
perf_equities_rep_IS = (np.multiply(returns_IS_price['World Equities'], weight_rep_IS['World Equities']))/weights_equities_rep_IS
perf_commodities_rep_IS = (np.multiply(returns_IS_price['Gold'], weight_rep_IS['Gold']) + np.multiply(returns_IS_price['Energy'], weight_rep_IS['Energy']) + np.multiply(returns_IS_price['Copper'], weight_rep_IS['Copper']))/weights_commodities_rep_IS

#The portfolio return
R_IS = weights_bonds_rep_IS*perf_bonds_rep_IS +  weights_equities_rep_IS*perf_equities_rep_IS + weights_commodities_rep_IS*perf_commodities_rep_IS

#The target portfolio return
B_IS = weights_bonds_target_IS*perf_bonds_target_IS + weights_equities_target_IS*perf_equities_target_IS + weights_commodities_target_IS*perf_commodities_target_IS

#The Selection Notional Fund
R_S_IS = weights_bonds_target_IS*perf_bonds_rep_IS + weights_equities_target_IS*perf_equities_rep_IS + weights_commodities_target_IS*perf_commodities_rep_IS

#The Allocation Notional Fund
B_S_IS = weights_bonds_rep_IS*perf_bonds_target_IS + weights_equities_rep_IS*perf_equities_target_IS + weights_commodities_rep_IS*perf_commodities_target_IS

plt.figure(figsize=(16,7))
plt.subplot(131)
#Interaction effect
interaction_IS = R_IS - R_S_IS - B_S_IS + B_IS
plt.plot(interaction_IS, 'r')
plt.title('Monthly Interaction Effect In-Sample')

plt.subplot(132)
#Allocation effect
allocation_IS = B_S_IS - B_IS 
plt.plot(allocation_IS, 'b')
plt.title('Monthly Allocation Effect In-Sample')

plt.subplot(133)
#Selection effect
selection_IS = R_S_IS - B_IS 
plt.plot(selection_IS, 'g')
plt.title('Monthly Selection Effect In-Sample')
#plt.savefig('Plot/attribution_IS.png')
#plt.show()
plt.close()

###Allocation/Selection Performance Attribution Out-Of-Sample###

#Weight for each type of asset (target)
weights_bonds_target_OS = weight_target_OS['World Bonds'] + weight_target_OS['US Investment Grade'] + weight_target_OS['US High Yield']
weights_equities_target_OS = weight_target_OS['World Equities']
weights_commodities_target_OS = weight_target_OS['Gold'] +  weight_target_OS['Energy'] + weight_target_OS['Copper']

#Weight for each type of asset (replication portfolio)
weights_bonds_rep_OS = weight_rep_OS['World Bonds'] + weight_rep_OS['US Investment Grade'] + weight_rep_OS['US High Yield']
weights_equities_rep_OS = weight_rep_OS['World Equities']
weights_commodities_rep_OS = weight_rep_OS['Gold'] +  weight_rep_OS['Energy'] + weight_rep_OS['Copper']

#Performance for each type of asset (target)
perf_bonds_target_OS = (np.multiply(returns_OS_price['World Bonds'], weight_target_OS['World Bonds']) + np.multiply(returns_OS_price['US Investment Grade'], weight_target_OS['US Investment Grade']) + np.multiply(returns_OS_price['US High Yield'], weight_target_OS['US High Yield']))/weights_bonds_target_OS
perf_equities_target_OS = (np.multiply(returns_OS_price['World Equities'], weight_target_OS['World Equities']))/weights_equities_target_OS
perf_commodities_target_OS = (np.multiply(returns_OS_price['Gold'], weight_target_OS['Gold']) + np.multiply(returns_OS_price['Energy'], weight_target_OS['Energy']) + np.multiply(returns_OS_price['Copper'], weight_target_OS['Copper']))/weights_commodities_target_OS

#Performance for each type of asset (replication portfolio)
perf_bonds_rep_OS = (np.multiply(returns_OS_price['World Bonds'], weight_rep_OS['World Bonds']) + np.multiply(returns_OS_price['US Investment Grade'], weight_rep_OS['US Investment Grade']) + np.multiply(returns_OS_price['US High Yield'], weight_rep_OS['US High Yield']))/weights_bonds_rep_OS
perf_equities_rep_OS = (np.multiply(returns_OS_price['World Equities'], weight_rep_OS['World Equities']))/weights_equities_rep_OS
perf_commodities_rep_OS = (np.multiply(returns_OS_price['Gold'], weight_rep_OS['Gold']) + np.multiply(returns_OS_price['Energy'], weight_rep_OS['Energy']) + np.multiply(returns_OS_price['Copper'], weight_rep_OS['Copper']))/weights_commodities_rep_OS

#The portfolio return
R_OS = weights_bonds_rep_OS*perf_bonds_rep_OS +  weights_equities_rep_OS*perf_equities_rep_OS + weights_commodities_rep_OS*perf_commodities_rep_OS

#The target portfolio return
B_OS = weights_bonds_target_OS*perf_bonds_target_OS + weights_equities_target_OS*perf_equities_target_OS + weights_commodities_target_OS*perf_commodities_target_OS

#The Selection Notional Fund
R_S_OS = weights_bonds_target_OS*perf_bonds_rep_OS + weights_equities_target_OS*perf_equities_rep_OS + weights_commodities_target_OS*perf_commodities_rep_OS

#The Allocation Notional Fund
B_S_OS = weights_bonds_rep_OS*perf_bonds_target_OS + weights_equities_rep_OS*perf_equities_target_OS + weights_commodities_rep_OS*perf_commodities_target_OS

plt.figure(figsize=(16,7))
plt.subplot(131)
#Interaction effect
interaction_OS = R_OS - R_S_OS - B_S_OS + B_OS
plt.plot(interaction_OS, 'r')
plt.title('Monthly Interaction Effect Out-of-Sample')

plt.subplot(132)
#Allocation effect
allocation_OS = B_S_OS - B_OS
plt.plot(allocation_OS, 'b')
plt.title('Monthly Allocation Effect Out-of-Sample')

plt.subplot(133)
#Selection effect
selection_OS = R_S_OS - B_OS
plt.plot(selection_OS, 'g')
plt.title('Monthly Selection Effect Out-of-Sample')
#plt.savefig('Plot/attribution_OS.png')
#plt.show()
plt.close()

###Collect all annualized Performance Attribution###
performance_attribution = pd.DataFrame({'In-Sample': [np.mean(R_IS, 0)*12, np.mean(B_IS, 0)*12, np.mean(R_S_IS, 0)*12, np.mean(B_S_IS, 0)*12, np.mean(interaction_IS, 0)*12, np.mean(allocation_IS, 0)*12, np.mean(selection_IS, 0)*12],
                                        'Out-of-Sample': [np.mean(R_OS, 0)*12, np.mean(B_OS, 0)*12, np.mean(R_S_OS, 0)*12, np.mean(B_S_OS, 0)*12, np.mean(interaction_OS, 0)*12, np.mean(allocation_OS, 0)*12, np.mean(selection_OS, 0)*12]},
                                       index=['R', 'B', 'R_S', 'B_S', 'Interaction', 'Allocation Effect', 'Selection Effect'])
performance_attribution.to_latex('Output/performance_attribution.tex')
